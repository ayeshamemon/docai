"""
Step 2.3 — Answer questions with Gemini, using retrieved chunks, with page citations.

Usage (run from backend/):
    python -m retrieval.ask "What is the complexity of self-attention?"
    python -m retrieval.ask "How many employees does Apple have?" --doc f9a3e5b32cb7cbf2
    python -m retrieval.ask "391,035" --doc f9a3e5b32cb7cbf2 --mode dense   # compare modes
    python -m retrieval.ask "Which commodity peaks highest?" --doc 56935fc42267b24a
      (--images N = send up to N page images to Gemini, default 3; --images 0 = text only)

Page images are chosen by combining two signals (eval: each catches what the other misses):
  1. the top page from visual search (ColSmol)               e.g. T11: text broken by OCR
  2. pages of the top text chunks that are in the visual index  e.g. T8: rerank found the chart page
     (only when the reranker is confident, score > 0; a weak text hit would push out
      the visual pages: T11 lost page 4 to pages of a -3.6 hit)

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

from retrieval.search import TEXT_MODES, search   # dense / bm25 / hybrid / rerank / visual

load_dotenv()
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
FALLBACK_MODEL = os.environ.get("GEMINI_FALLBACK_MODEL", "")   # optional, used if MODEL stays busy
MAX_RETRIES = 3
TOP_K = 5
DEFAULT_IMAGES = 3   # max page images per answer (cost/latency cap)
TEXT_PAGE_MIN_SCORE = 0.0   # rerank score a text hit needs before its page images are used

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
- Some sources are PAGE IMAGES. Read charts, figures and scanned text from them directly.
  For charts, give approximate values and say they are read from the chart.
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


def build_contents(question: str, hits: list[dict], pages: list[dict]) -> list:
    """Text sources first, then each page image with a label Gemini can cite."""
    contents = [build_prompt(question, hits)]
    for p in pages:
        contents.append(f"[Page image | page {p['page']}]")
        contents.append(types.Part.from_bytes(data=open(p["image"], "rb").read(),
                                              mime_type="image/png"))
    return contents


def pick_pages(question: str, hits: list[dict], doc_id: str | None, n: int) -> list[dict]:
    """Up to n page images: top visual hit, then pages of the top text chunks, then more visual."""
    if n <= 0:
        return []
    from ingest.index import get_client
    from ingest.visual_index import PAGE_COLLECTION, get_page_client, page_point_id

    client = get_page_client(get_client())
    chosen, seen = [], set()

    def add(p):
        key = (p["doc_id"], p["page"])
        if key not in seen and len(chosen) < n:
            seen.add(key)
            chosen.append(p)

    visual = search(question, client, k=n, doc_id=doc_id, mode="visual")
    if visual:
        add(visual[0])                                            # 1. best visual page
    for h in hits:                                                # 2. text-guided pages
        if h["score"] < TEXT_PAGE_MIN_SCORE:                      #    only confident text hits
            continue
        for pg in h["pages"]:
            pts = client.retrieve(PAGE_COLLECTION, ids=[page_point_id(h["doc_id"], pg)])
            if pts:                                               # only pages with a figure / scan
                add({**pts[0].payload, "score": h["score"], "type": "page", "text": "",
                     "pages": [pg], "heading": "page image (from text hit)"})
    for v in visual[1:]:                                          # 3. fill up with visual
        add(v)
    client.close()
    return chosen


def answer(question: str, doc_id: str | None = None, k: int = TOP_K,
           mode: str = "rerank", images: int = DEFAULT_IMAGES):
    hits = search(question, k=k, doc_id=doc_id, mode=mode)            # 1a. text chunks
    pages = pick_pages(question, hits, doc_id, images)                # 1b. page images

    contents = build_contents(question, hits, pages)                   # 2. build prompt
    return hits + pages, generate(contents)                            # 3. generate (streaming)


def generate(contents):
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
                    model=model, contents=contents, config=config)
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
    ap.add_argument("--mode", choices=TEXT_MODES, default="rerank", help="text retrieval mode")
    ap.add_argument("--images", type=int, default=DEFAULT_IMAGES,
                    help="max page images sent to Gemini (0 = text only)")
    ap.add_argument("--show-sources", action="store_true", help="print the retrieved chunks")
    args = ap.parse_args()

    hits, stream = answer(args.question, args.doc, args.k, args.mode, args.images)

    print(f"\nQ: {args.question}\n\nA: ", end="", flush=True)
    try:
        for text in stream:                   # print the answer word by word as it arrives
            print(text, end="", flush=True)
    except RuntimeError as e:
        print(f"\n[error] {e}")

    print(f"\n\nRetrieved sources ({args.mode}):")
    for i, h in enumerate(hits, 1):
        print(f"  {i}. {h['type']:<6} pages {h['pages']}  score {h['score']}  — {h['heading'][:60]}")
        if args.show_sources:
            print("     " + h["text"][:300].replace("\n", " "))