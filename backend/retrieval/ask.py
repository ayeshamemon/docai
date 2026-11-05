"""
Step 2.3 — Answer questions with Gemini, using retrieved chunks, with page citations.

Usage (run from backend/):
    python -m retrieval.ask "What is the complexity of self-attention?"
    python -m retrieval.ask "How many employees does Apple have?" --doc f9a3e5b32cb7cbf2

Needs a .env file in backend/ with:
    GEMINI_API_KEY=your_key
    GEMINI_MODEL=gemini-...-flash
"""

import argparse
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from ingest.index import get_client, search   # Step 2.2

load_dotenv()
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
FALLBACK_MODEL = os.environ.get("GEMINI_FALLBACK_MODEL", "")   # optional, used if MODEL stays busy
MAX_RETRIES = 3
TOP_K = 5

_llm = None


def get_llm() -> genai.Client:
    """Create the Gemini client once and keep it alive (reads GEMINI_API_KEY).
    If it is created inside a function, Python closes it before streaming finishes."""
    global _llm
    if _llm is None:
        _llm = genai.Client()
    return _llm

SYSTEM_PROMPT = """You answer questions about PDF documents using ONLY the sources given.

Rules:
- Use only facts from the sources. Do not use outside knowledge.
- After every fact, cite its page like [p. 6]. Use the page numbers shown in the source headers.
- Tables are in Markdown; read rows and columns carefully, and say which year or column a number comes from.
- If the sources do not contain the answer, say: "I couldn't find this in the document." Do not guess.
- Be concise."""


def build_prompt(question: str, hits: list[dict]) -> str:
    """Number each retrieved chunk and show its pages, so the LLM can cite them."""
    blocks = []
    for i, h in enumerate(hits, 1):
        pages = ", ".join(str(p) for p in h["pages"])
        blocks.append(f"[Source {i} | {h['type']} | page {pages}]\n{h['text']}")
    sources = "\n\n---\n\n".join(blocks)
    return f"Sources:\n\n{sources}\n\n---\n\nQuestion: {question}"


def answer(question: str, doc_id: str | None = None, k: int = TOP_K):
    qdrant = get_client()
    hits = search(question, qdrant, k=k, doc_id=doc_id)   # 1. retrieve
    qdrant.close()

    prompt = build_prompt(question, hits)                  # 2. build prompt
    return hits, generate(prompt)                          # 3. generate (streaming)


def generate(prompt: str):
    """
    Stream the answer as text pieces.
    If Gemini is busy (503) or rate-limited (429), wait and retry,
    then try the fallback model. Retries only happen before the first
    word arrives, so the answer is never printed twice.
    """
    models_to_try = [MODEL] + ([FALLBACK_MODEL] if FALLBACK_MODEL else [])
    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.2)
    last_error = None

    for model in models_to_try:
        for attempt in range(MAX_RETRIES):
            started = False
            try:
                stream = get_llm().models.generate_content_stream(
                    model=model, contents=prompt, config=config)
                for part in stream:
                    started = True
                    yield part.text or ""
                return                                    # success
            except (errors.ServerError, errors.ClientError) as e:
                busy = getattr(e, "code", None) in (429, 500, 503)
                if started or not busy:                   # mid-answer failure or real error: stop
                    raise                                 # real error (bad key, bad model): stop
                last_error = e
                wait = 2 ** attempt * 2                   # 2s, 4s, 8s
                print(f"\n[{model} busy ({e.code}), retrying in {wait}s...]", flush=True)
                time.sleep(wait)
        if FALLBACK_MODEL and model != FALLBACK_MODEL:
            print(f"\n[switching to fallback model {FALLBACK_MODEL}]", flush=True)

    raise RuntimeError(f"Gemini is unavailable right now, try again later. Last error: {last_error}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--doc", default=None)
    ap.add_argument("--k", type=int, default=TOP_K)
    ap.add_argument("--show-sources", action="store_true", help="print the retrieved chunks")
    args = ap.parse_args()

    hits, stream = answer(args.question, args.doc, args.k)

    print(f"\nQ: {args.question}\n\nA: ", end="", flush=True)
    try:
        for text in stream:                   # print the answer word by word as it arrives
            print(text, end="", flush=True)
    except RuntimeError as e:
        print(f"\n[error] {e}")

    print("\n\nRetrieved sources:")
    for i, h in enumerate(hits, 1):
        print(f"  {i}. {h['type']:<6} pages {h['pages']}  score {h['score']}  — {h['heading'][:60]}")
        if args.show_sources:
            print("     " + h["text"][:300].replace("\n", " "))