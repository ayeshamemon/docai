"""
Step 5.2: entity + relationship extraction (the first half of GraphRAG).

For every chunk, Gemini reads the text and returns:
  entities:       {name, type, description}
  relationships:  {source, target, description, strength 1-10, evidence}
`evidence` must be a short quote copied from the chunk. We check it really is in the chunk,
so a relationship the model invented (not written in the text) is thrown away.

Usage (run from backend/, server stopped is not needed: this does not touch Qdrant):
    python -m graph.extract data/processed/<doc_id> --limit 5     # try 5 chunks first
    python -m graph.extract data/processed/<doc_id>               # whole document (resumes)
    python -m graph.extract data/processed/<doc_id> --show c12    # print one chunk's result
    python -m graph.extract data/processed/<doc_id> --redo        # start again from scratch

Writes data/processed/<doc_id>/graph/extractions.jsonl  (one line per chunk)
It is resumable: chunks already in the file are skipped, so a crash or rate limit loses nothing.
"""

import argparse
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import httpx
from google.genai import errors, types

from retrieval.ask import FALLBACK_MODEL, MODEL, get_llm

EXTRACT_MODEL = os.environ.get("EXTRACT_MODEL", MODEL)
MAX_RETRIES = 6
_exhausted: set[str] = set()   # models that hit their daily limit in this run
MIN_WORDS = 8                 # chunks shorter than this (page headers, stray labels) are skipped
BIBLIOGRAPHY = ("references", "bibliography", "works cited")   # list of cited papers: only names, no facts


def is_bibliography(chunk: dict) -> bool:
    h = re.sub(r"^[\d.\s]+", "", (chunk.get("heading") or "").lower()).strip()
    return h.startswith(BIBLIOGRAPHY)

# Fixed, generic types: works for any PDF. Anything else becomes Concept.
ENTITY_TYPES = ["Organization", "Person", "Product", "Place", "Date", "Metric", "Concept", "Event"]

PROMPT = """Extract a knowledge graph from the text below, taken from a PDF document.

ENTITIES: the important named things in the text.
- name: as written in the text, short (e.g. "iPhone", "Greater China", "BERT-large", "P100 GPU")
- type: one of {types}
- description: one sentence, using only facts from this text
Metric = a named measurement with its value, e.g. "iPhone net sales 2026" or "BLEU score of Transformer (big)".
Put the number in the description, e.g. "iPhone net sales in 2026 were $254,098 million, up 21%".

RELATIONSHIPS: how two of those entities are connected, as stated in the text.
- source, target: entity names exactly as in your entities list
- description: one sentence saying how they are related
- strength: 1-10, how central this link is to the text
- evidence: a SHORT quote (5-25 words) copied word for word from the text that states the link

Rules:
- Only what the text states. No outside knowledge, no guesses.
- Every relationship needs an exact evidence quote. If you cannot quote it, leave it out.
- Skip generic words ("the Company", "the table", "this section") and labels of document parts
  ("Table 3", "Section 3.2", "Figure 1"): they are not entities.
- If the text has nothing worth extracting, return empty lists.

Section heading: {heading}
Pages: {pages}

TEXT:
\"\"\"
{text}
\"\"\"

Return JSON only:
{{"entities": [{{"name": "", "type": "", "description": ""}}],
  "relationships": [{{"source": "", "target": "", "description": "", "strength": 5, "evidence": ""}}]}}"""


# ---------- checking the model's output ----------

def norm(s: str) -> str:
    """Lowercase, drop markdown table pipes and odd quotes, collapse spaces.
    So a quote still matches when only spacing / table formatting differs."""
    s = s.lower().replace("|", " ").replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"[*_`]", "", s)
    s = re.sub(r"\s+", " ", s)
    return re.sub(r"\$ ", "$", s).replace(" %", "%").strip()   # "$ 254,098" = "$254,098"


def quote_in_text(quote: str, text: str) -> bool:
    q = norm(quote).strip(' ."\'')
    return len(q) >= 8 and q in norm(text)


def clean(raw: dict, chunk: dict, stats: Counter) -> dict:
    """Keep only valid entities and relationships whose evidence is really in the chunk."""
    entities, seen = [], {}
    for e in raw.get("entities") or []:
        name = (e.get("name") or "").strip()
        if not name or len(name) > 80:
            stats["entity_bad_name"] += 1
            continue
        etype = e.get("type") if e.get("type") in ENTITY_TYPES else "Concept"
        key = name.lower()
        if key in seen:                           # same entity twice in one chunk
            continue
        seen[key] = name
        entities.append({"name": name, "type": etype,
                         "description": (e.get("description") or "").strip()})

    rels = []
    for r in raw.get("relationships") or []:
        s, t = (r.get("source") or "").strip(), (r.get("target") or "").strip()
        if s.lower() not in seen or t.lower() not in seen or s.lower() == t.lower():
            stats["rel_unknown_entity"] += 1      # points at something not in the entity list
            continue
        ev = (r.get("evidence") or "").strip()
        if not quote_in_text(ev, chunk["text"]):
            stats["rel_bad_evidence"] += 1        # quote not in the text: likely invented
            continue
        try:
            strength = max(1, min(10, int(r.get("strength", 5))))
        except (TypeError, ValueError):
            strength = 5
        rels.append({"source": seen[s.lower()], "target": seen[t.lower()],
                     "description": (r.get("description") or "").strip(),
                     "strength": strength, "evidence": ev})
    return {"entities": entities, "relationships": rels}


# ---------- calling Gemini ----------

def call_gemini(prompt: str) -> dict:
    """One JSON call with retries on busy / rate-limit errors, then the fallback model."""
    config = types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json")
    models = [EXTRACT_MODEL] + ([FALLBACK_MODEL] if FALLBACK_MODEL and FALLBACK_MODEL != EXTRACT_MODEL else [])
    models = [m for m in models if m not in _exhausted] or models
    last = None
    for model in models:
        limited = 0
        for attempt in range(MAX_RETRIES):
            try:
                resp = get_llm().models.generate_content(model=model, contents=prompt, config=config)
                return json.loads(resp.text or "{}")
            except json.JSONDecodeError as e:     # broken JSON: try once more
                last = e
            except (errors.ServerError, errors.ClientError, httpx.TimeoutException) as e:
                code = "timeout" if isinstance(e, httpx.TimeoutException) else getattr(e, "code", None)
                if code == 402:                   # credits used up: retrying will not help
                    raise SystemExit("[stopped] Gemini credits are used up (402). Everything done so far "
                                     "is saved; add credits or switch API key, then run the same command.")
                if code not in (429, 500, 503, 504, "timeout"):
                    raise
                last = e
                # 429 = per-minute limit (free tier): wait for the minute to reset
                limited = limited + 1 if code == 429 else 0
                if limited >= 3:                  # still limited after 3 minutes: daily limit reached
                    _exhausted.add(model)
                    print(f"    [{model}: daily limit reached, using the next model for the rest of this run]", flush=True)
                    break
                wait = 65 if code == 429 else min(2 ** attempt * 5, 60)
                print(f"    [{model} busy ({code}), retrying in {wait}s]", flush=True)
                time.sleep(wait)
    raise RuntimeError(f"extraction failed: {last}")


def extract_chunk(chunk: dict, stats: Counter) -> dict:
    prompt = PROMPT.format(types=", ".join(ENTITY_TYPES), heading=chunk.get("heading") or "-",
                           pages=", ".join(map(str, chunk["pages"])), text=chunk["text"])
    raw = call_gemini(prompt)
    stats["entities_raw"] += len(raw.get("entities") or [])
    stats["rels_raw"] += len(raw.get("relationships") or [])
    out = clean(raw, chunk, stats)
    return {"chunk_id": chunk["chunk_id"], "pages": chunk["pages"], "type": chunk["type"],
            "heading": chunk.get("heading"), **out}


# ---------- running over a document ----------

def out_file(doc_dir: Path) -> Path:
    return doc_dir / "graph" / "extractions.jsonl"


def load_done(path: Path) -> dict:
    if not path.exists():
        return {}
    return {r["chunk_id"]: r for r in map(json.loads, path.read_text().splitlines()) if r}


def extract_document(doc_dir: Path, limit: int | None = None, redo: bool = False, on_chunk=None):
    chunks = json.loads((doc_dir / "chunks.json").read_text())
    path = out_file(doc_dir)
    path.parent.mkdir(exist_ok=True)
    if redo and path.exists():
        path.unlink()
    done = load_done(path)
    todo = [c for c in chunks if c["chunk_id"] not in done and len(c["text"].split()) >= MIN_WORDS
            and not is_bibliography(c)]
    if limit:
        todo = todo[:limit]
    print(f"[extract] {doc_dir.name}: {len(chunks)} chunks, {len(done)} done, {len(todo)} to do "
          f"· model {EXTRACT_MODEL}")

    stats = Counter()
    t0 = time.time()
    with open(path, "a") as f:
        for i, c in enumerate(todo, 1):
            r = extract_chunk(c, stats)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()                              # saved after every chunk: safe to stop anytime
            stats["entities"] += len(r["entities"])
            stats["rels"] += len(r["relationships"])
            print(f"  [{i}/{len(todo)}] {c['chunk_id']:<24} p.{c['pages'][0]:<4} "
                  f"{len(r['entities']):>2} entities {len(r['relationships']):>2} relationships", flush=True)
            if on_chunk:
                on_chunk(i, len(todo))

    if todo:
        kept = stats["rels"] / max(1, stats["rels_raw"])
        print(f"\n[done] {len(todo)} chunks in {time.time() - t0:.0f}s")
        print(f"  entities: {stats['entities']} kept of {stats['entities_raw']}")
        print(f"  relationships: {stats['rels']} kept of {stats['rels_raw']} ({kept:.0%})")
        print(f"    dropped, evidence not in text: {stats['rel_bad_evidence']}")
        print(f"    dropped, unknown entity:      {stats['rel_unknown_entity']}")
    print(f"[saved] {path}")


def show(doc_dir: Path, which: str):
    done = load_done(out_file(doc_dir))
    hits = [r for cid, r in done.items() if cid == which or cid.endswith("_" + which)]
    if not hits:
        print(f"no extraction for {which}")
        return
    r = hits[0]
    print(f"{r['chunk_id']}  p.{r['pages']}  [{r['heading']}]\n\nENTITIES")
    for e in r["entities"]:
        print(f"  {e['type']:<12} {e['name']:<30} {e['description']}")
    print("\nRELATIONSHIPS")
    for x in r["relationships"]:
        print(f"  {x['source']} --[{x['strength']}]--> {x['target']}\n"
              f"      {x['description']}\n      evidence: \"{x['evidence']}\"")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("doc_dir", type=Path)
    ap.add_argument("--limit", type=int, help="only the next N chunks (for testing)")
    ap.add_argument("--redo", action="store_true", help="delete old results and start again")
    ap.add_argument("--show", help="print the result for one chunk, e.g. c12")
    args = ap.parse_args()
    if args.show:
        show(args.doc_dir, args.show)
    else:
        extract_document(args.doc_dir, args.limit, args.redo)