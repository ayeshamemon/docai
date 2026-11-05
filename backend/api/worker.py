"""
Step 3.2 — Background processing of uploaded PDFs.

One worker thread takes uploads from a queue and runs them ONE AT A TIME
(Docling and ColSmol are heavy; two at once would crawl on a laptop):

    queued → parsing → chunking → indexing → visual_indexing (i/N pages) → ready
                                                                          ↘ failed

The status is kept in memory and saved to data/processed/<doc_id>/status.json,
so a server restart does not forget which documents are ready.
"""

import json
import queue
import threading
import time
import traceback
from pathlib import Path

from ingest.chunk import chunk_document
from ingest.index import add_document as index_text
from ingest.parse import ingest as parse_pdf
from ingest.visual_index import add_document as index_pages

PROCESSED = Path("data/processed")
UPLOADS = Path("data/uploads")

_jobs: "queue.Queue[tuple[str, Path]]" = queue.Queue()
_status: dict[str, dict] = {}
_status_lock = threading.Lock()


# ---------- status ----------

def set_status(doc_id: str, stage: str, **extra):
    st = {"doc_id": doc_id, "stage": stage, "updated": time.time(), **extra}
    with _status_lock:
        old = _status.get(doc_id, {})
        st = {**{k: v for k, v in old.items() if k in ("filename", "queued_at")}, **st}
        _status[doc_id] = st
    folder = PROCESSED / doc_id
    if folder.exists():                      # the folder only exists after parsing starts
        (folder / "status.json").write_text(json.dumps(st, indent=2))
    return st


def get_status(doc_id: str) -> dict | None:
    with _status_lock:
        if doc_id in _status:
            return _status[doc_id]
    f = PROCESSED / doc_id / "status.json"
    if f.exists():
        return json.loads(f.read_text())
    return None


# ---------- jobs ----------

def enqueue(doc_id: str, pdf_path: Path, filename: str) -> dict:
    st = set_status(doc_id, "queued", filename=filename, queued_at=time.time(),
                    position=_jobs.qsize() + 1)
    _jobs.put((doc_id, pdf_path))
    return st


def process(doc_id: str, pdf_path: Path, client, lock):
    """The same pipeline as the CLI scripts, step by step, with status updates."""
    t0 = time.time()
    set_status(doc_id, "parsing")
    doc_dir = parse_pdf(pdf_path, PROCESSED)                      # Step 1

    set_status(doc_id, "chunking")
    chunks = chunk_document(doc_dir)                              # Step 2.1
    (doc_dir / "chunks.json").write_text(json.dumps(chunks, indent=2, ensure_ascii=False))

    set_status(doc_id, "indexing", chunks=len(chunks))
    index_text(doc_dir, client, lock=lock)                        # Step 2.2 / 2.4

    def on_page(done, total):
        set_status(doc_id, "visual_indexing", chunks=len(chunks), pages_done=done, pages_total=total)
    pages = index_pages(doc_dir, client, lock=lock, on_page=on_page)   # Step 2.5

    meta = json.loads((doc_dir / "meta.json").read_text())
    set_status(doc_id, "ready", chunks=len(chunks), page_images=pages,
               num_pages=meta["num_pages"], seconds=round(time.time() - t0, 1))


def _loop(client, lock):
    while True:
        doc_id, pdf_path = _jobs.get()
        try:
            process(doc_id, pdf_path, client, lock)
            print(f"[worker] {doc_id} ready", flush=True)
        except Exception as e:
            traceback.print_exc()
            set_status(doc_id, "failed", error=f"{type(e).__name__}: {e}")
        finally:
            _jobs.task_done()


def start_worker(client, lock):
    """Called once at server startup."""
    t = threading.Thread(target=_loop, args=(client, lock), daemon=True, name="upload-worker")
    t.start()
    return t