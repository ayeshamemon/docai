"""
Step 3.1 + 3.2 — FastAPI server.

Endpoints
    GET  /documents                 list documents and their status
    POST /documents                 upload a PDF (returns at once; processed in the background)
    GET  /documents/{doc_id}/status queued → parsing → chunking → indexing → visual_indexing → ready
    POST /chat                      streamed answer (SSE): sources → tokens → done

Loads every model ONCE at startup and keeps one shared Qdrant connection,
so questions are answered without the 5–10 s model loading of the CLI.

Run (from backend/):
    uvicorn api.main:app --port 8000

Try it:
    open http://localhost:8000/docs              (automatic test page)
    curl -N -X POST localhost:8000/chat -H "Content-Type: application/json" \
         -d '{"question": "How many employees does Apple have?", "doc_id": "f9a3e5b32cb7cbf2"}'

Note: local Qdrant allows ONE process. While the server runs, the CLI scripts
(search, ask, eval) cannot open Qdrant; stop the server first (Ctrl+C).
"""

import json
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import shutil
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field

from ingest.index import COLLECTION, doc_filter, get_client, get_dense, get_sparse
from ingest.parse import file_hash
from ingest.visual_index import PAGE_COLLECTION, get_page_client, get_visual_model
from retrieval.ask import DEFAULT_IMAGES, TOP_K, build_contents, generate, get_llm, retrieve
from retrieval.search import TEXT_MODES, get_reranker

from api import worker

PROCESSED = Path("data/processed")
state = {}                       # shared things the endpoints use (Qdrant client)
qdrant_lock = threading.Lock()   # one request at a time talks to local Qdrant


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load all models and open Qdrant once. Shutdown: close Qdrant."""
    t0 = time.time()
    print("[api] loading models ...", flush=True)
    get_dense(); get_sparse(); get_reranker(); get_visual_model(); get_llm()
    state["qdrant"] = get_page_client(get_client())
    worker.start_worker(state["qdrant"], qdrant_lock)     # Step 3.2: background uploads
    print(f"[api] ready in {time.time() - t0:.0f}s", flush=True)
    yield
    state["qdrant"].close()


app = FastAPI(title="DocAI", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],   # the Next.js app (Step 3.4)
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- documents ----------

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")       # opening the bare address shows the test page


@app.get("/health")
def health():
    return {"status": "ok"}


def doc_stage(doc_id: str, chunks_indexed: int) -> str:
    """Worker status if there is one; documents indexed with the CLI count as ready."""
    st = worker.get_status(doc_id)
    if st:
        return st["stage"]
    return "ready" if chunks_indexed else "not indexed"


@app.get("/documents")
def list_documents():
    """Every document: processed ones (with indexed counts) and ones still in the queue."""
    client = state["qdrant"]
    docs, seen = [], set()
    for meta_file in sorted(PROCESSED.glob("*/meta.json")):
        meta = json.loads(meta_file.read_text())
        flt = doc_filter(meta["doc_id"])
        with qdrant_lock:
            chunks = client.count(COLLECTION, count_filter=flt).count
            pages = client.count(PAGE_COLLECTION, count_filter=flt).count
        seen.add(meta["doc_id"])
        docs.append({
            "doc_id": meta["doc_id"],
            "filename": meta["filename"],
            "num_pages": meta["num_pages"],
            "page_types": meta.get("page_types"),
            "chunks_indexed": chunks,
            "pages_indexed": pages,
            "status": doc_stage(meta["doc_id"], chunks),
        })
    for st in list(worker._status.values()):        # uploaded but not parsed yet
        if st["doc_id"] not in seen:
            docs.append({"doc_id": st["doc_id"], "filename": st.get("filename"),
                         "status": st["stage"]})
    return docs


@app.post("/documents")
def upload_document(file: UploadFile = File(...)):
    """Save the PDF and queue it. Returns at once; poll /documents/{doc_id}/status."""
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "please upload a .pdf file")

    worker.UPLOADS.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=worker.UPLOADS, suffix=".pdf", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    doc_id = file_hash(tmp_path)                       # same id as parse.py → same doc = same id

    st = worker.get_status(doc_id)
    already_indexed = (PROCESSED / doc_id / "chunks.json").exists()
    if (st and st["stage"] not in ("failed",)) or (not st and already_indexed):
        tmp_path.unlink()                              # duplicate upload: reuse the existing one
        return st or {"doc_id": doc_id, "stage": "ready", "duplicate": True}

    dest_dir = worker.UPLOADS / doc_id                 # keep the original file name
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(file.filename).name
    shutil.move(str(tmp_path), dest)
    return worker.enqueue(doc_id, dest, Path(file.filename).name)


@app.get("/documents/{doc_id}/status")
def document_status(doc_id: str):
    st = worker.get_status(doc_id)
    if st:
        return st
    if (PROCESSED / doc_id / "chunks.json").exists():
        return {"doc_id": doc_id, "stage": "ready"}
    raise HTTPException(404, f"unknown doc_id {doc_id}")


# ---------- chat ----------

class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    doc_id: str | None = None                  # None = search all documents
    mode: Literal[tuple(TEXT_MODES)] = "rerank"
    k: int = Field(TOP_K, ge=1, le=20)
    images: int = Field(DEFAULT_IMAGES, ge=0, le=5)


def source_view(h: dict) -> dict:
    """What the UI needs to show a citation (no local file paths)."""
    if h["type"] == "page":
        return {"kind": "page", "doc_id": h["doc_id"], "page": h["page"],
                "reason": h.get("reason"), "score": h["score"]}
    return {"kind": "chunk", "doc_id": h["doc_id"], "chunk_id": h["chunk_id"],
            "type": h["type"], "heading": h["heading"], "pages": h["pages"],
            "boxes": h["boxes"],                       # for highlighting in the PDF
            "text": h["text"][:600], "score": h["score"]}


def sse(event: str, data) -> str:
    """One Server-Sent Event. JSON keeps newlines inside tokens safe."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/chat")
def chat(req: ChatRequest):
    if req.doc_id:
        if not (PROCESSED / req.doc_id).exists():
            raise HTTPException(404, f"unknown doc_id {req.doc_id}")
        stage = doc_stage(req.doc_id, 1)
        if stage != "ready":
            raise HTTPException(409, f"document is not ready yet (stage: {stage})")

    def stream():
        t0 = time.time()
        with qdrant_lock:                              # 1. retrieve (shared Qdrant)
            hits, pages = retrieve(req.question, req.doc_id, req.k, req.mode, req.images,
                                   client=state["qdrant"])
        t_retrieve = time.time() - t0
        yield sse("sources", [source_view(h) for h in hits + pages])

        t_first = None
        try:                                           # 2. generate, word by word
            for text in generate(build_contents(req.question, hits, pages)):
                if text:
                    if t_first is None:
                        t_first = time.time() - t0
                    yield sse("token", text)
        except Exception as e:                         # e.g. Gemini unavailable
            yield sse("error", str(e))
        total = time.time() - t0
        timing = {"retrieve_s": round(t_retrieve, 1),
                  "first_token_s": round(t_first, 1) if t_first else None,
                  "total_s": round(total, 1)}
        print(f"[chat] {timing}", flush=True)         # also visible in the server terminal
        yield sse("done", timing)

    return StreamingResponse(stream(), media_type="text/event-stream")