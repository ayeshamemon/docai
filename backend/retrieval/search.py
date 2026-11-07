"""
Step 2.4 — Retrieval modes.

  dense  : meaning only (bge-small)                         ← Step 2.2 baseline
  bm25   : exact words only                                 ← for diagnosis
  hybrid : dense + bm25, merged with RRF (rank fusion)
  rerank : hybrid top 20 → cross-encoder re-reads each (question, chunk) → top k
  visual : ColQwen2.5 searches PAGE IMAGES (charts, scans)   ← Step 2.5

Usage (run from backend/):
    python -m retrieval.search "391,035" --doc f9a3e5b32cb7cbf2 --mode hybrid
    python -m retrieval.search "What is the complexity of self-attention?" --mode rerank

Test mode: where does the answer chunk rank in EVERY mode?
    python -m retrieval.search "391,035" --doc f9a3e5b32cb7cbf2 --find 391,035
    python -m retrieval.search "Which commodity peaks highest?" --doc 56935fc42267b24a --page 9
  (--page = rank of the answer PAGE; works for text modes and the visual mode)
"""

import argparse

from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient, models

from ingest.index import COLLECTION, doc_filter, get_client, get_dense, get_sparse

RERANK_MODEL = "BAAI/bge-reranker-base"
CANDIDATES = 20          # how many chunks each first-stage search hands on
TEXT_MODES = ["dense", "bm25", "hybrid", "rerank"]
MODES = TEXT_MODES + ["visual", "graph", "graph-walk", "graph+rerank"]

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


def search_visual(query, client, k, doc_id=None):
    """ColPali-style: each question token finds its best patch on each page (MaxSim)."""
    from ingest.visual_index import PAGE_COLLECTION, embed_query, get_page_client
    get_page_client(client)                                # make sure the collection exists
    pts = client.query_points(PAGE_COLLECTION, query=embed_query(query), using="colpali",
                              limit=k, query_filter=doc_filter(doc_id) if doc_id else None).points
    return [{"score": round(p.score, 3), "type": "page", "text": "", "pages": [p.payload["page"]],
             "heading": f"page image ({p.payload['reason']})", **p.payload} for p in pts]


def search_graph(query, client, k, doc_id=None):
    from graph.search import search_graph as fn          # imported here: graph.search imports this file
    return fn(query, client, k, doc_id)


def search_graph_walk(query, client, k, doc_id=None):
    from graph.search import search_graph_walk as fn
    return fn(query, client, k, doc_id)


def search_graph_rerank(query, client, k, doc_id=None):
    from graph.search import search_graph_rerank as fn
    return fn(query, client, k, doc_id)


GRAPH_MODES = ["graph", "graph-walk", "graph+rerank"]
SEARCHERS = {"dense": search_dense, "bm25": search_bm25,
             "hybrid": search_hybrid, "rerank": search_rerank, "visual": search_visual,
             "graph": search_graph, "graph-walk": search_graph_walk,
             "graph+rerank": search_graph_rerank}


def search(query: str, client: QdrantClient | None = None, k: int = 5,
           doc_id: str | None = None, mode: str = "rerank") -> list[dict]:
    own = client is None
    client = client or get_client()
    try:
        return SEARCHERS[mode](query, client, k, doc_id)
    finally:
        if own:
            client.close()


# ---------- command line ----------

def find_rank(hits, keyword=None, page=None):
    """Rank of the first hit that contains the keyword, or that is on the given page."""
    for i, h in enumerate(hits, 1):
        if page is not None and page in h["pages"]:
            return i
        if keyword and keyword.lower() in h["text"].lower():
            return i
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--doc", default=None)
    ap.add_argument("--mode", choices=MODES, default="rerank")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--find", default=None,
                    help="keyword from the correct answer: prints its rank in every text mode")
    ap.add_argument("--page", type=int, default=None,
                    help="page of the correct answer: prints its rank in every mode, incl. visual")
    args = ap.parse_args()

    client = get_client()
    if args.find or args.page:
        target = f'page {args.page}' if args.page else f'keyword "{args.find}"'
        print(f'\nQ: {args.query}\nanswer: {target}\n')
        # visual can only be judged by page (page images have no text)
        modes = MODES if args.page else TEXT_MODES
        for mode in modes:
            # reranker only re-orders the hybrid top 20, so its depth is 20
            depth = CANDIDATES if mode == "rerank" else 30
            hits = SEARCHERS[mode](args.query, client, depth, args.doc)
            r = find_rank(hits, args.find, args.page)
            mark = "✅" if r and r <= args.k else "❌"
            where = f"rank {r}" if r else f"not in top {depth}"
            print(f"  {mode:<7} {mark}  {where}")
    else:
        for i, h in enumerate(SEARCHERS[args.mode](args.query, client, args.k, args.doc), 1):
            preview = (h["text"] or h.get("image", "")).replace("\n", " ")[:160]
            print(f"\n#{i}  score {h['score']}  {h['type']}  doc {h['doc_id']}  pages {h['pages']}")
            print(f"    {preview}")
    client.close()