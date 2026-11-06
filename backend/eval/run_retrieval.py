"""
Retrieval evaluation: run every test question in every mode and score the ranks.

Usage (run from backend/):
    python -m eval.run_retrieval                       # all modes incl. visual
    python -m eval.run_retrieval --modes dense bm25 hybrid rerank
    python -m eval.run_retrieval --check               # only validate the questions file

How a question is judged
  - text modes (dense/bm25/hybrid/rerank): rank of the first chunk that contains `keyword`
    (multi-hop questions have `keywords`: the rank at which EVERY part has been found,
     so hit@5 means the model received all the evidence it needs)
    (if a question has no keyword, e.g. charts, the rank of the first chunk on `page`:
     that means "found the page", NOT "found the answer")
  - visual: rank of `page` among page images. Only for chart / scan questions, because
    only figure and scanned pages are in the visual index (digital text pages are not)
  ✅ = rank <= 5 (what the LLM receives)

Writes eval/results_retrieval.csv
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from ingest.index import get_client
from retrieval.search import CANDIDATES, SEARCHERS, TEXT_MODES, find_rank

HERE = Path(__file__).parent
QUESTIONS = HERE / "questions.jsonl"
PROCESSED = Path("data/processed")
TOP = 5
DEPTH = 30
VISUAL_CATEGORIES = {"chart", "scan-faded", "scan-clean"}


def load_questions():
    return [json.loads(l) for l in QUESTIONS.read_text().splitlines() if l.strip()]


def check(questions):
    """Make sure each keyword really exists in that document's chunks (and is not everywhere)."""
    ok = True
    chunks_cache = {}
    for q in questions:
        f = PROCESSED / q["doc"] / "chunks.json"
        if not f.exists():
            print(f"  {q['id']:<5} ⚠️  doc {q['doc']} has no chunks.json (chunk + index it first)")
            ok = False
            continue
        chunks = chunks_cache.setdefault(q["doc"], json.loads(f.read_text()))
        for kw in keywords_of(q):
            n = sum(kw.lower() in c["text"].lower() for c in chunks)
            if n == 0:
                print(f"  {q['id']:<5} ❌ keyword {kw!r} not in any chunk (OCR/parse differs?)")
                ok = False
            elif n > 3:
                print(f"  {q['id']:<5} ⚠️  keyword {kw!r} is in {n} chunks (too common)")
        # a multi-hop question must really need two places: no single chunk may hold every part
        kws = q.get("keywords") or []
        if len(kws) > 1 and any(all(k.lower() in c["text"].lower() for k in kws) for c in chunks):
            print(f"  {q['id']:<5} ⚠️  one chunk holds every part {kws}: not really multi-hop")
    print("[check] all keywords found" if ok else "[check] fix the lines above")
    return ok


def keywords_of(q: dict) -> list[str]:
    """One keyword (single-fact questions) or several (multi-hop: every part must be found)."""
    return q.get("keywords") or ([q["keyword"]] if q.get("keyword") else [])


def evidence_rank(hits, q):
    """Rank at which ALL parts of the evidence have been retrieved (0 = some part missing).
    Single-keyword questions: the usual rank. No keyword (charts): rank of the page."""
    kws = keywords_of(q)
    if not kws:
        return find_rank(hits, page=q["page"]) or 0
    ranks = [find_rank(hits, keyword=k) for k in kws]
    return 0 if any(r is None for r in ranks) else max(ranks)


def run(questions, modes):
    client = get_client()
    rows = []
    for q in questions:
        row = {"id": q["id"], "category": q["category"], "question": q["question"]}
        for mode in modes:
            if mode == "visual":
                if not q["page"] or q["category"] not in VISUAL_CATEGORIES:
                    row[mode] = ""                      # page not in the visual index
                    continue
                hits = SEARCHERS[mode](q["question"], client, DEPTH, q["doc"])
                r = find_rank(hits, page=q["page"])
            else:
                depth = CANDIDATES if mode == "rerank" else DEPTH
                hits = SEARCHERS[mode](q["question"], client, depth, q["doc"])
                r = evidence_rank(hits, q)
            row[mode] = r or 0                          # 0 = not found within depth
        rows.append(row)
        cells = "  ".join(f"{m}:{fmt(row[m])}" for m in modes)
        print(f"  {q['id']:<5} {q['category']:<13} {cells}", flush=True)
    client.close()
    return rows


def fmt(r):
    if r == "":
        return "  -  "
    return f"{'✅' if 0 < r <= TOP else '❌'}{r if r else '--':>3}"


def summarize(rows, modes):
    """Hit@5 (share in the top 5) and MRR (average of 1/rank) per mode, overall and per category."""
    def score(rs):
        rs = [r for r in rs if r != ""]
        if not rs:
            return "  n/a          "
        hit = sum(0 < r <= TOP for r in rs) / len(rs)
        mrr = sum(1 / r for r in rs if r) / len(rs)
        return f"{hit:>5.0%} {mrr:>5.2f} ({len(rs):>2})"

    cats = sorted({r["category"] for r in rows})
    print(f"\n{'':<15}" + "".join(f"{m:>18}" for m in modes))
    print(f"{'':<15}" + "".join(f"{'hit@5  MRR  (n)':>18}" for _ in modes))
    for c in cats + ["ALL"]:
        sub = rows if c == "ALL" else [r for r in rows if r["category"] == c]
        print(f"{c:<15}" + "".join(f"{score([r[m] for r in sub]):>18}" for m in modes))
    print("\nchart questions in text modes = page found, not answer found")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=TEXT_MODES + ["visual"],
                    choices=list(SEARCHERS))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--only", nargs="+", help="run only these question ids")
    args = ap.parse_args()

    questions = load_questions()
    if args.only:
        questions = [q for q in questions if q["id"] in args.only]
    print(f"[eval] {len(questions)} questions")
    check(questions)
    if not args.check:
        rows = run(questions, args.modes)
        summarize(rows, args.modes)
        out = HERE / "results_retrieval.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "category", "question"] + args.modes)
            w.writeheader()
            w.writerows(rows)
        print(f"[saved] {out}")