"""
Step 9.5b control: query decomposition (a cheaper alternative to the agent).

One Gemini call splits the question into 1-3 short search queries, ALL written at the start;
each query runs through normal search (hybrid + reranker); the results are merged in turn
(best of query 1, best of query 2, ..., then the second best of each) up to MAX_CHUNKS,
the same budget the agent used on average. Then the normal answer step.

What it has that the agent also has: several queries, about 7 chunks.
What it lacks: no loop, no tool choice, and the 2nd query cannot use what the 1st one found
(for "which product grew fastest, and why?" it cannot write "why did iPhone grow").

Usage (run from backend/, API server stopped):
    python -m agent.decompose "Which product grew fastest in 2026, and why?" --doc f9a3e5b32cb7cbf2
"""

import argparse
import time

MAX_QUERIES = 3
MAX_CHUNKS = 7               # ~ the agent's average (6.9) for a fair comparison
PER_QUERY = 5

PROMPT = """Split this question about a document into 1 to {n} SHORT search queries,
one per piece of information needed. Use only words from the question or obvious synonyms;
you do not know the document's content.

Question: {question}

Return JSON only: {{"queries": ["...", "..."]}}"""


def decompose(question: str) -> list[str]:
    from graph.extract import call_gemini
    out = call_gemini(PROMPT.format(n=MAX_QUERIES, question=question))
    queries = [q.strip() for q in out.get("queries", []) if isinstance(q, str) and q.strip()]
    return queries[:MAX_QUERIES] or [question]


def retrieve_decompose(question: str, doc_id: str, client=None, k: int = MAX_CHUNKS):
    """Same shape as ask.retrieve plus info: (hits, pages, info)."""
    from retrieval.search import search
    t0 = time.time()
    queries = decompose(question)
    lists = [search(q, client, k=PER_QUERY, doc_id=doc_id, mode="rerank") for q in queries]
    hits, seen = [], set()
    for rank in range(PER_QUERY):                      # round robin: best of each query first
        for lst in lists:
            if rank < len(lst) and lst[rank]["chunk_id"] not in seen and len(hits) < k:
                seen.add(lst[rank]["chunk_id"])
                hits.append(lst[rank])
    from retrieval.ask import pick_pages, DEFAULT_IMAGES
    pages = pick_pages(question, hits, doc_id, DEFAULT_IMAGES, client=client)
    return hits, pages, {"queries": queries, "seconds": round(time.time() - t0, 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--doc", required=True)
    args = ap.parse_args()
    from ingest.index import get_client
    from agent.tools import short_id
    client = get_client()
    hits, pages, info = retrieve_decompose(args.question, args.doc, client)
    print(f"queries: {info['queries']}")
    print(f"chunks:  {[short_id(h['chunk_id']) for h in hits]}  pages: {[p['page'] for p in pages]}")
    client.close()