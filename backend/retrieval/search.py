"""
Step 2.4 — Retrieval modes.

  dense  : meaning only (bge-small)                         ← Step 2.2 baseline
  bm25   : exact words only                                 ← for diagnosis
  hybrid : dense + bm25, merged with RRF (rank fusion)
  rerank : hybrid top 20 → cross-encoder re-reads each (question, chunk) → top k

Usage (run from backend/):
    python -m retrieval.search "391,035" --doc f9a3e5b32cb7cbf2 --mode hybrid
    python -m retrieval.search "What is the complexity of self-attention?" --mode rerank

Test mode: where does the answer chunk rank in EVERY mode?
    python -m retrieval.search "391,035" --doc f9a3e5b32cb7cbf2 --find 391,035
"""

import argparse

from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient, models

from ingest.index import COLLECTION, doc_filter, get_client, get_dense, get_sparse

RERANK_MODEL = "BAAI/bge-reranker-base"
CANDIDATES = 20          # how many chunks each first-stage search hands on
MODES = ["dense", "bm25", "hybrid", "rerank"]

_reranker = None


def get_reranker() -> TextCrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = TextCrossEncoder(RERANK_MODEL)
    return _reranker


# ---------- query vectors ----------

def dense_query(query: str) -> list[float]:
    return next(iter(get_dense().query_embed([query]))).tolist()


def sparse_query(query: str) -> models.SparseVector:
    s = next(iter(get_sparse().query_embed(query)))
    return models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())


def to_hits(points) -> list[dict]:
    return [{"score": round(p.score, 3), **p.payload} for p in points]


# ---------- the four modes ----------

def search_dense(query, client, k, doc_id=None):
    pts = client.query_points(COLLECTION, query=dense_query(query), using="dense", limit=k,
                              query_filter=doc_filter(doc_id) if doc_id else None).points
    return to_hits(pts)


def search_bm25(query, client, k, doc_id=None):
    pts = client.query_points(COLLECTION, query=sparse_query(query), using="bm25", limit=k,
                              query_filter=doc_filter(doc_id) if doc_id else None).points
    return to_hits(pts)


def search_hybrid(query, client, k, doc_id=None):
    """Run dense and BM25 inside Qdrant, merge the two rankings with RRF."""
    flt = doc_filter(doc_id) if doc_id else None
    pts = client.query_points(
        COLLECTION,
        prefetch=[
            models.Prefetch(query=dense_query(query), using="dense", limit=CANDIDATES, filter=flt),
            models.Prefetch(query=sparse_query(query), using="bm25", limit=CANDIDATES, filter=flt),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=max(k, CANDIDATES),
    ).points
    return to_hits(pts)[:k]


def search_rerank(query, client, k, doc_id=None):
    """Hybrid gives ~20 candidates; the cross-encoder reads each with the question."""
    candidates = search_hybrid(query, client, CANDIDATES, doc_id)
    if not candidates:
        return []
    scores = list(get_reranker().rerank(query, [c["text"] for c in candidates]))
    for c, s in zip(candidates, scores):
        c["score"] = round(float(s), 3)
    return sorted(candidates, key=lambda c: c["score"], reverse=True)[:k]


SEARCHERS = {"dense": search_dense, "bm25": search_bm25,
             "hybrid": search_hybrid, "rerank": search_rerank}


def search(query: str, client: QdrantClient | None = None, k: int = 5,
           doc_id: str | None = None, mode: str = "hybrid") -> list[dict]:
    own = client is None
    client = client or get_client()
    try:
        return SEARCHERS[mode](query, client, k, doc_id)
    finally:
        if own:
            client.close()


# ---------- command line ----------

def find_rank(hits, keyword):
    kw = keyword.lower()
    return next((i for i, h in enumerate(hits, 1) if kw in h["text"].lower()), None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--doc", default=None)
    ap.add_argument("--mode", choices=MODES, default="hybrid")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--find", default=None,
                    help="keyword from the correct answer: prints its rank in every mode")
    args = ap.parse_args()

    client = get_client()
    if args.find:
        print(f'\nQ: {args.query}\nanswer keyword: "{args.find}"\n')
        for mode in MODES:
            # reranker only re-orders the hybrid top 20, so its depth is 20
            depth = CANDIDATES if mode == "rerank" else 30
            hits = SEARCHERS[mode](args.query, client, depth, args.doc)
            r = find_rank(hits, args.find)
            mark = "✅" if r and r <= args.k else "❌"
            where = f"rank {r}" if r else f"not in top {depth}"
            print(f"  {mode:<7} {mark}  {where}")
    else:
        for i, h in enumerate(SEARCHERS[args.mode](args.query, client, args.k, args.doc), 1):
            preview = h["text"].replace("\n", " ")[:160]
            print(f"\n#{i}  score {h['score']}  {h['type']}  doc {h['doc_id']}  pages {h['pages']}")
            print(f"    {preview}")
    client.close()