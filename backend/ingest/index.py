"""
Step 2.2 — Embeddings + Qdrant (dense retrieval).

Index a chunked document:
    python ingest/index.py add data/processed/<doc_id>

Search (test that retrieval finds the right chunks — no LLM yet):
    python ingest/index.py search "What is the complexity of self-attention?"
    python ingest/index.py search "total net sales 2026" --doc f9a3e5b32cb7cbf2 --k 5
"""

import argparse
import json
import uuid
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # 384 numbers per text, 512-token limit
VECTOR_SIZE = 384
COLLECTION = "chunks"
QDRANT_PATH = "data/qdrant"              # local Qdrant: saved in a folder, no Docker yet

_model = None


def get_model() -> TextEmbedding:
    """Load the embedding model once (first time downloads ~130 MB)."""
    global _model
    if _model is None:
        _model = TextEmbedding(EMBED_MODEL)
    return _model


def get_client() -> QdrantClient:
    client = QdrantClient(path=QDRANT_PATH)
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(size=VECTOR_SIZE,
                                               distance=models.Distance.COSINE),
        )
    return client


def point_id(chunk_id: str) -> str:
    """Qdrant needs UUIDs; make a stable one from our chunk_id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def doc_filter(doc_id: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(
        key="doc_id", match=models.MatchValue(value=doc_id))])


def add_document(doc_dir: Path, client: QdrantClient) -> int:
    chunks = json.loads((doc_dir / "chunks.json").read_text())
    doc_id = chunks[0]["doc_id"]

    # re-indexing the same doc? remove its old points first
    client.delete(COLLECTION, points_selector=models.FilterSelector(filter=doc_filter(doc_id)))

    # 1. text -> numbers (passage_embed = embedding mode for documents)
    vectors = list(get_model().passage_embed([c["text"] for c in chunks], batch_size=32))

    # 2. store each vector with its chunk info (payload) for citations later
    client.upsert(COLLECTION, points=[
        models.PointStruct(id=point_id(c["chunk_id"]), vector=v.tolist(), payload=c)
        for c, v in zip(chunks, vectors)
    ])
    return len(chunks)


def search(query: str, client: QdrantClient, k: int = 5, doc_id: str | None = None) -> list[dict]:
    # query_embed = embedding mode for questions (bge adds its search instruction)
    qvec = next(iter(get_model().query_embed([query])))
    hits = client.query_points(
        COLLECTION,
        query=qvec.tolist(),
        limit=k,
        query_filter=doc_filter(doc_id) if doc_id else None,
    ).points
    return [{"score": round(h.score, 3), **h.payload} for h in hits]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add");    a.add_argument("doc_dir", type=Path)
    s = sub.add_parser("search"); s.add_argument("query")
    s.add_argument("--doc", default=None); s.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    client = get_client()
    if args.cmd == "add":
        n = add_document(args.doc_dir, client)
        total = client.count(COLLECTION).count
        print(f"[done] indexed {n} chunks from {args.doc_dir.name} (collection now has {total})")
    else:
        for i, h in enumerate(search(args.query, client, args.k, args.doc), 1):
            preview = h["text"].replace("\n", " ")[:160]
            print(f"\n#{i}  score {h['score']}  {h['type']}  doc {h['doc_id']}  pages {h['pages']}")
            print(f"    {preview}")
    client.close()  # close cleanly (avoids a harmless error message at exit)