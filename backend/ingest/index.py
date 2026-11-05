"""
Step 2.2 + 2.4 — Indexing: store every chunk with TWO vectors in Qdrant.

  dense : bge-small (384 numbers, meaning)
  bm25  : sparse vector (exact words; Qdrant adds the IDF part)

Usage:
    python ingest/index.py add data/processed/<doc_id>
    python ingest/index.py list                     # which documents are indexed

Search lives in retrieval/search.py.
"""

import argparse
import json
import uuid
from pathlib import Path

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

DENSE_MODEL = "BAAI/bge-small-en-v1.5"   # 384 numbers per text, 512-token limit
SPARSE_MODEL = "Qdrant/bm25"             # word counts -> BM25 (with Qdrant's IDF modifier)
VECTOR_SIZE = 384
COLLECTION = "docai_chunks"              # new name: the old "chunks" had dense only
QDRANT_PATH = "data/qdrant"              # local Qdrant: saved in a folder, no Docker yet

_dense = None
_sparse = None


def get_dense() -> TextEmbedding:
    global _dense
    if _dense is None:
        _dense = TextEmbedding(DENSE_MODEL)
    return _dense


def get_sparse() -> SparseTextEmbedding:
    global _sparse
    if _sparse is None:
        _sparse = SparseTextEmbedding(SPARSE_MODEL)
    return _sparse


def get_client() -> QdrantClient:
    client = QdrantClient(path=QDRANT_PATH)
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            COLLECTION,
            vectors_config={
                "dense": models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
            },
            sparse_vectors_config={
                # IDF modifier = Qdrant computes "how rare is this word" across all chunks
                "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF),
            },
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
    texts = [c["text"] for c in chunks]

    # re-indexing the same doc? remove its old points first
    client.delete(COLLECTION, points_selector=models.FilterSelector(filter=doc_filter(doc_id)))

    dense_vecs = list(get_dense().passage_embed(texts, batch_size=32))
    sparse_vecs = list(get_sparse().embed(texts, batch_size=32))

    client.upsert(COLLECTION, points=[
        models.PointStruct(
            id=point_id(c["chunk_id"]),
            vector={
                "dense": d.tolist(),
                "bm25": models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist()),
            },
            payload=c,
        )
        for c, d, s in zip(chunks, dense_vecs, sparse_vecs)
    ])
    return len(chunks)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add"); a.add_argument("doc_dir", type=Path)
    sub.add_parser("list")
    args = ap.parse_args()

    client = get_client()
    if args.cmd == "add":
        n = add_document(args.doc_dir, client)
        print(f"[done] indexed {n} chunks from {args.doc_dir.name} "
              f"(collection now has {client.count(COLLECTION).count})")
    else:
        points, _ = client.scroll(COLLECTION, limit=100000, with_payload=["doc_id"])
        counts = {}
        for p in points:
            counts[p.payload["doc_id"]] = counts.get(p.payload["doc_id"], 0) + 1
        for d, n in counts.items():
            print(f"  {d}: {n} chunks")
    client.close()