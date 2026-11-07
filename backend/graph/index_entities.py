"""
Step 5.3c: put the graph's entities into Qdrant, so a question can find where to start the walk.

Each entity becomes one point in the collection "docai_entities":
  vector  = bge-small embedding of "name (type): descriptions"
  payload = doc_id, entity id, name, type, aliases, chunk_ids, pages

Usage (run from backend/, with the API server STOPPED: local Qdrant allows one process):
    python -m graph.index_entities add data/processed/<doc_id>
    python -m graph.index_entities search "which GPU was used" --doc <doc_id>
"""

import argparse
import json
import uuid
from pathlib import Path

from qdrant_client import models

from ingest.index import VECTOR_SIZE, doc_filter, get_client, get_dense

ENTITY_COLLECTION = "docai_entities"
MAX_DESC_WORDS = 120          # keep the embedded text short (bge-small reads 512 tokens)


def ensure_collection(client):
    if not client.collection_exists(ENTITY_COLLECTION):
        client.create_collection(
            ENTITY_COLLECTION,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
        )


def entity_text(e: dict) -> str:
    words = " ".join(e["descriptions"]).split()[:MAX_DESC_WORDS]
    names = ", ".join([e["name"]] + e["aliases"])
    return f"{names} ({e['type']}): {' '.join(words)}"


def entity_point_id(doc_id: str, eid: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}/{eid}"))


def add_document(doc_dir: Path, client, lock=None):
    data = json.loads((doc_dir / "graph" / "graph.json").read_text())
    doc_id, entities = data["doc_id"], data["entities"]
    vectors = list(get_dense().embed([entity_text(e) for e in entities]))   # outside the lock
    points = [models.PointStruct(
        id=entity_point_id(doc_id, e["id"]), vector=v.tolist(),
        payload={"doc_id": doc_id, "entity_id": e["id"], "name": e["name"], "type": e["type"],
                 "aliases": e["aliases"], "chunk_ids": e["chunk_ids"], "pages": e["pages"]})
        for e, v in zip(entities, vectors)]

    def write():
        ensure_collection(client)
        client.delete(ENTITY_COLLECTION, points_selector=models.FilterSelector(filter=doc_filter(doc_id)))
        client.upsert(ENTITY_COLLECTION, points=points)          # replace: rebuilding is safe
    if lock:
        with lock:
            write()
    else:
        write()
    print(f"[entities] {doc_id}: {len(points)} entities indexed in {ENTITY_COLLECTION}")


def search_entities(query: str, client, doc_id: str | None = None, k: int = 10) -> list[dict]:
    if not client.collection_exists(ENTITY_COLLECTION):
        return []
    vec = next(iter(get_dense().query_embed([query]))).tolist()
    res = client.query_points(ENTITY_COLLECTION, query=vec, limit=k,
                              query_filter=doc_filter(doc_id) if doc_id else None)
    return [{"score": p.score, **p.payload} for p in res.points]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("doc_dir", type=Path)
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("--doc")
    s.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    client = get_client()
    if args.cmd == "add":
        add_document(args.doc_dir, client)
    else:
        for h in search_entities(args.query, client, args.doc, args.k):
            print(f"  {h['score']:.3f}  {h['type']:<12} {h['name']:<35} p.{h['pages']}")
    client.close()