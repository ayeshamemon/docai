"""
Step 5.4: graph search. Two modes, both end with the same reranker as the normal search.

  graph         question -> matching entities -> walk 1 hop -> their chunks -> rerank -> top k
  graph+rerank  normal rerank search keeps its best chunks (fact 1), the graph adds chunks
                connected to them (fact 2), with a fixed number of slots:
                    top k = (k - GRAPH_SLOTS) from rerank  +  GRAPH_SLOTS from the graph
                The graph chunks are picked by walk score (connection), reranker only breaks ties:
                the reranker is what missed fact 2 in the first place.

How the walk scores a chunk (higher = more likely useful):
  seed entity's own chunk ........................ seed score
  chunk of an entity linked by a STATED edge ...... seed score x 0.8 x strength/10
  next chunk in the same section (STRUCTURAL) ..... seed score x 0.5
Only 1 hop: the document is one connected map, so long walks reach everything.
Page edges are not walked (a page holds unrelated chunks); they stay in the graph for the UI.

Usage (run from backend/, API server stopped):
    python -m graph.search "Which GPUs were used, and what TFLOPS was assumed for them?" --doc bdfaa68d8984f0dc
    python -m graph.search "..." --doc <id> --mode graph --explain
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from graph.build import load_graph
from graph.index_entities import search_entities
from ingest.index import get_client

PROCESSED = Path("data/processed")
SEED_ENTITIES = 5            # entities found by meaning search
SEED_CHUNKS = 3              # graph+rerank: entities of the top 3 rerank chunks are seeds too
MAX_CHUNKS_PER_ENTITY = 6    # hub entities ("Apple", "Transformer") are in dozens of chunks
GRAPH_POOL = 20              # chunks handed to the reranker
GRAPH_SLOTS = 2              # graph+rerank: slots reserved for graph chunks in the top k
STATED_W, NEXT_W = 0.8, 0.5

_cache = {}


def load(doc_id: str):
    """Graph + chunk texts for one document, cached; reloaded if graph.json changes."""
    gfile = PROCESSED / doc_id / "graph" / "graph.json"
    if not gfile.exists():
        return None
    mtime = gfile.stat().st_mtime
    if doc_id not in _cache or _cache[doc_id][0] != mtime:
        chunks = {c["chunk_id"]: c for c in json.loads((PROCESSED / doc_id / "chunks.json").read_text())}
        _cache[doc_id] = (mtime, load_graph(PROCESSED / doc_id), chunks)
    return _cache[doc_id][1:]


def walk(g, seeds: dict[str, float]) -> tuple[dict, dict]:
    """seeds: entity id -> score. Returns chunk id -> score, and chunk id -> why (for --explain)."""
    score, why = defaultdict(float), {}

    def give(cid, s, reason):
        if cid in g and g.nodes[cid].get("node") == "chunk" and s > score[cid]:
            score[cid], why[cid] = s, reason

    def entity_chunks(eid):
        return g.nodes[eid].get("chunk_ids", [])[:MAX_CHUNKS_PER_ENTITY]

    for eid, s in seeds.items():
        if eid not in g:
            continue
        name = g.nodes[eid]["name"]
        for cid in entity_chunks(eid):                                   # 0 hops
            give(cid, s, f"mentions {name}")
            for nxt in list(g.successors(cid)) + list(g.predecessors(cid)):   # next chunk, same section
                if g.edges.get((cid, nxt), g.edges.get((nxt, cid), {})).get("rel") == "next":
                    give(nxt, s * NEXT_W, f"next to a chunk mentioning {name}")
        for other in set(g.successors(eid)) | set(g.predecessors(eid)):  # 1 hop, stated edges
            e = g.edges.get((eid, other)) or g.edges.get((other, eid))
            if not e or e.get("kind") != "stated" or g.nodes[other].get("node") != "entity":
                continue
            w = s * STATED_W * e.get("strength", 5) / 10
            for cid in entity_chunks(other):
                give(cid, w, f"{name} -> {g.nodes[other]['name']}: {e['descriptions'][0]}")
    return score, why


def rerank_chunks(query: str, chunks: list[dict]) -> list[dict]:
    from retrieval.search import get_reranker
    if not chunks:
        return []
    scores = list(get_reranker().rerank(query, [c["text"] for c in chunks]))
    out = [{**c, "score": round(float(s), 3)} for c, s in zip(chunks, scores)]
    return sorted(out, key=lambda c: c["score"], reverse=True)


def seeds_from_question(query, client, doc_id):
    return {h["entity_id"]: h["score"] for h in search_entities(query, client, doc_id, SEED_ENTITIES)}


def graph_pool(query, client, doc_id, seed_chunk_ids=(), explain=False):
    loaded = load(doc_id)
    if not loaded:
        return [], {}
    g, chunks = loaded
    seeds = seeds_from_question(query, client, doc_id)
    for cid in seed_chunk_ids:                    # entities of the chunks search already found
        if cid in g:
            for eid in g.predecessors(cid):
                if g.nodes[eid].get("node") == "entity":
                    seeds[eid] = max(seeds.get(eid, 0), 0.7)
    score, why = walk(g, seeds)
    best = sorted(score, key=score.get, reverse=True)[:GRAPH_POOL]
    pool = [{**chunks[c], "graph_score": round(score[c], 3), "graph_why": why[c]}
            for c in best if c in chunks]
    return pool, seeds


def search_graph(query, client, k, doc_id=None):
    if not doc_id:
        return []                                 # graphs are per document
    pool, _ = graph_pool(query, client, doc_id)
    return rerank_chunks(query, pool)[:k]


def search_graph_walk(query, client, k, doc_id=None):
    """Graph only, ordered by walk score (connection); the reranker only breaks ties."""
    if not doc_id:
        return []
    pool, _ = graph_pool(query, client, doc_id)
    ranked = rerank_chunks(query, pool)
    return sorted(ranked, key=lambda c: (c["graph_score"], c["score"]), reverse=True)[:k]


def search_graph_rerank(query, client, k, doc_id=None):
    from retrieval.search import CANDIDATES, search_rerank
    base = search_rerank(query, client, CANDIDATES, doc_id)
    if not doc_id:
        return base[:k]
    keep = max(1, min(k, 5) - GRAPH_SLOTS)
    top = base[:keep]
    pool, _ = graph_pool(query, client, doc_id, [h["chunk_id"] for h in base[:SEED_CHUNKS]])
    chosen = {h["chunk_id"] for h in top}
    # pick the graph chunks by how strongly they are CONNECTED (walk score); the reranker only
    # breaks ties. Picking by reranker repeats the mistake that made search miss fact 2.
    ranked = rerank_chunks(query, [c for c in pool if c["chunk_id"] not in chosen])
    extra = sorted(ranked, key=lambda c: (c["graph_score"], c["score"]), reverse=True)[:GRAPH_SLOTS]
    for c in extra:
        c["from_graph"] = True
    chosen |= {c["chunk_id"] for c in extra}
    rest = [h for h in base[keep:] if h["chunk_id"] not in chosen]
    return (top + extra + rest)[:k]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--doc", required=True)
    ap.add_argument("--mode", choices=["graph", "graph-walk", "graph+rerank"], default="graph+rerank")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--explain", action="store_true", help="show seed entities and why each chunk came")
    args = ap.parse_args()

    client = get_client()
    if args.explain:
        pool, seeds = graph_pool(args.query, client, args.doc)
        g, _ = load(args.doc)
        print("SEED ENTITIES")
        for eid, s in sorted(seeds.items(), key=lambda x: -x[1]):
            print(f"  {s:.3f}  {g.nodes[eid]['name']}")
        print("\nGRAPH POOL (before reranking)")
        for c in pool:
            print(f"  {c['graph_score']:.3f}  {c['chunk_id']:<24} p.{c['pages']}  {c['graph_why'][:90]}")
    fn = {"graph": search_graph, "graph-walk": search_graph_walk}.get(args.mode, search_graph_rerank)
    print(f"\nTOP {args.k} ({args.mode})")
    for i, h in enumerate(fn(args.query, client, args.k, args.doc), 1):
        tag = "  [graph]" if h.get("from_graph") else ""
        print(f"\n#{i}  score {h['score']}  {h['type']}  pages {h['pages']}{tag}")
        print("    " + h["text"].replace("\n", " ")[:160])
    client.close()