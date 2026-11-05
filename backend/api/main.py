"""
Step 3.1 — FastAPI server.

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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field

from ingest.index import COLLECTION, doc_filter, get_client, get_dense, get_sparse
from ingest.visual_index import PAGE_COLLECTION, get_page_client, get_visual_model
from retrieval.ask import DEFAULT_IMAGES, TOP_K, build_contents, generate, get_llm, retrieve
from retrieval.search import TEXT_MODES, get_reranker

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


@app.get("/documents")
def list_documents():
    """Every processed document, with how many chunks / page images are indexed."""
    client = state["qdrant"]
    docs = []
    for meta_file in sorted(PROCESSED.glob("*/meta.json")):
        meta = json.loads(meta_file.read_text())
        flt = doc_filter(meta["doc_id"])
        with qdrant_lock:
            chunks = client.count(COLLECTION, count_filter=flt).count
            pages = client.count(PAGE_COLLECTION, count_filter=flt).count
        docs.append({
            "doc_id": meta["doc_id"],
            "filename": meta["filename"],
            "num_pages": meta["num_pages"],
            "page_types": meta.get("page_types"),
            "chunks_indexed": chunks,
            "pages_indexed": pages,
            "status": "ready" if chunks else "not indexed",
        })
    return docs


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
    if req.doc_id and not (PROCESSED / req.doc_id).exists():
        raise HTTPException(404, f"unknown doc_id {req.doc_id}")

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