"""
Step 2.5 — Visual index: search PAGE IMAGES with ColQwen2.5 (ColPali family).

Only pages where text fails are indexed:
  - scanned pages (from meta.json)
  - pages with figures/charts (from elements.json)
Digital text pages are already handled well by hybrid search.

Usage (run from backend/):
    python -m ingest.visual_index add data/processed/<doc_id>
    python -m ingest.visual_index add data/processed/<doc_id> --all-pages
    python -m ingest.visual_index list

Model: set VISUAL_MODEL in .env to switch (default ColQwen2.5; ColSmol is much faster on a laptop):
    VISUAL_MODEL=vidore/colqwen2.5-v0.2
    VISUAL_MODEL=vidore/colSmol-256M
"""

import argparse
import contextlib
import json
import threading
import os
import time
import uuid
from pathlib import Path

import torch
from dotenv import load_dotenv
from PIL import Image
from qdrant_client import QdrantClient, models

from ingest.index import QDRANT_PATH, doc_filter

load_dotenv()
VISUAL_MODEL = os.environ.get("VISUAL_MODEL", "vidore/colqwen2.5-v0.2")
PAGE_COLLECTION = "docai_pages"
PATCH_DIM = 128          # every patch / query token becomes 128 numbers

_model = None
_processor = None
_model_lock = threading.Lock()   # the model is used by one thread at a time (upload worker vs /chat)


# ---------- model ----------

def device_and_dtype():
    """
    CUDA (Kaggle/Colab) -> bfloat16. Mac GPU (MPS) or CPU -> float32,
    because bfloat16 adapter loading crashes (segfault) on MPS.
    Force a device with VISUAL_DEVICE=cpu / mps / cuda in .env.
    """
    forced = os.environ.get("VISUAL_DEVICE", "").lower()
    if forced == "cuda" or (not forced and torch.cuda.is_available()):
        return "cuda", torch.bfloat16
    if forced == "mps" or (not forced and torch.backends.mps.is_available()):
        return "mps", torch.float32
    return "cpu", torch.float32


def get_visual_model():
    """Load the ColPali-family model once (first time downloads several GB)."""
    global _model, _processor
    if _model is None:
        device, dtype = device_and_dtype()
        if "colsmol" in VISUAL_MODEL.lower():
            from colpali_engine.models import ColIdefics3 as Model
            from colpali_engine.models import ColIdefics3Processor as Processor
        else:
            from colpali_engine.models import ColQwen2_5 as Model
            from colpali_engine.models import ColQwen2_5_Processor as Processor
        print(f"[visual] loading {VISUAL_MODEL} on {device} ({dtype}) ...", flush=True)
        # load on CPU first, THEN move to the GPU (loading straight onto MPS crashes)
        _model = Model.from_pretrained(VISUAL_MODEL, torch_dtype=dtype).eval()
        _model = _model.to(device)
        _processor = Processor.from_pretrained(VISUAL_MODEL)
        print(f"[visual] model ready on {device}", flush=True)
    return _model, _processor


def embed_image(path: str) -> list[list[float]]:
    """One page image -> many patch vectors (multi-vector)."""
    model, proc = get_visual_model()
    batch = proc.process_images([Image.open(path).convert("RGB")]).to(model.device)
    with _model_lock, torch.no_grad():
        emb = model(**batch)[0]                          # (num_patches, 128)
    mask = batch["attention_mask"][0].bool()             # drop padding positions
    return emb[mask].float().cpu().tolist()


def embed_query(query: str) -> list[list[float]]:
    """One question -> one vector per token."""
    model, proc = get_visual_model()
    batch = proc.process_queries([query]).to(model.device)
    with _model_lock, torch.no_grad():
        emb = model(**batch)[0]
    mask = batch["attention_mask"][0].bool()
    return emb[mask].float().cpu().tolist()


# ---------- Qdrant ----------

def get_page_client(client: QdrantClient | None = None) -> QdrantClient:
    client = client or QdrantClient(path=QDRANT_PATH)
    if not client.collection_exists(PAGE_COLLECTION):
        client.create_collection(
            PAGE_COLLECTION,
            vectors_config={
                "colpali": models.VectorParams(
                    size=PATCH_DIM,
                    distance=models.Distance.COSINE,
                    # MAX_SIM = for each query token, take its best patch, then add up
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM),
                ),
            },
        )
    return client


def page_point_id(doc_id: str, page: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}_p{page}"))


# ---------- which pages? ----------

def select_pages(doc_dir: Path, all_pages: bool = False) -> list[dict]:
    meta = json.loads((doc_dir / "meta.json").read_text())
    elements = json.loads((doc_dir / "elements.json").read_text())
    figure_pages = {e["page"] for e in elements if e["type"] == "figure"}

    chosen = []
    for p in meta["pages"]:
        if all_pages and p["type"] != "blank":
            reason = "all"
        elif p["type"] == "scanned":
            reason = "scanned"
        elif p["page"] in figure_pages:
            reason = "figure"
        else:
            continue
        chosen.append({"page": p["page"], "image": str(doc_dir / p["image"]), "reason": reason})
    return chosen


def add_document(doc_dir: Path, client: QdrantClient, all_pages: bool = False,
                 lock=None, on_page=None) -> int:
    """lock: optional threading.Lock held only while writing to Qdrant.
    on_page(done, total): optional progress callback (the API uses it for the status)."""
    lock = lock or contextlib.nullcontext()
    meta = json.loads((doc_dir / "meta.json").read_text())
    doc_id = meta["doc_id"]
    pages = select_pages(doc_dir, all_pages)

    with lock:
        client.delete(PAGE_COLLECTION, points_selector=models.FilterSelector(filter=doc_filter(doc_id)))
    if on_page:
        on_page(0, len(pages))
    t0 = time.time()
    for i, p in enumerate(pages, 1):
        vecs = embed_image(p["image"])                   # slow part: no Qdrant lock
        with lock:
            client.upsert(PAGE_COLLECTION, points=[models.PointStruct(
                id=page_point_id(doc_id, p["page"]),
                vector={"colpali": vecs},
                payload={"doc_id": doc_id, "page": p["page"], "image": p["image"],
                         "reason": p["reason"], "filename": meta["filename"]},
            )])
        print(f"  page {p['page']:>3} ({p['reason']:<7}) {len(vecs)} patches  "
              f"[{i}/{len(pages)}, {time.time() - t0:.0f}s]", flush=True)
        if on_page:
            on_page(i, len(pages))
    return len(pages)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("doc_dir", type=Path)
    a.add_argument("--all-pages", action="store_true", help="index every non-blank page")
    sub.add_parser("list")
    args = ap.parse_args()

    client = get_page_client()
    if args.cmd == "add":
        n = add_document(args.doc_dir, client, args.all_pages)
        print(f"[done] indexed {n} page images from {args.doc_dir.name} "
              f"(collection now has {client.count(PAGE_COLLECTION).count})")
    else:
        pts, _ = client.scroll(PAGE_COLLECTION, limit=100000, with_payload=["doc_id", "reason"])
        counts = {}
        for p in pts:
            key = (p.payload["doc_id"], p.payload["reason"])
            counts[key] = counts.get(key, 0) + 1
        for (d, r), n in sorted(counts.items()):
            print(f"  {d}  {r:<8} {n} pages")
    client.close()