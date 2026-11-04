"""
Step 1 — PDF ingestion pipeline.

Usage:
    python ingest.py path/to/file.pdf [--out data/processed]

Output (per document):
    data/processed/<doc_id>/elements.json   -> list of text/table/figure elements
    data/processed/<doc_id>/pages/page_N.png -> rendered page images (for ColPali + viewer)
    data/processed/<doc_id>/meta.json        -> page count, page types, timings
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import pymupdf as fitz
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import PictureItem, TableItem, TextItem

MIN_TEXT_CHARS = 50      # less text than this -> blank or scanned
IMAGE_COVERAGE = 0.80    # one image covering >=80% of the page -> scanned
RENDER_DPI = 144


def file_hash(path: Path) -> str:
    """Stable doc_id: same file -> same id (lets us cache re-uploads)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def largest_image_coverage(page) -> float:
    """Fraction of the page covered by its single largest image."""
    page_area = page.rect.width * page.rect.height
    best = 0.0
    for img in page.get_image_info():
        r = fitz.Rect(img["bbox"]) & page.rect
        best = max(best, r.width * r.height / page_area)
    return best


def classify_page(page) -> str:
    """
    digital : real text layer, use it directly
    scanned : page is basically a photo -> needs OCR
              (even if it has a hidden, possibly bad, text layer)
    blank   : almost no text and no big image -> nothing to OCR
    """
    text_len = len(page.get_text("text").strip())
    coverage = largest_image_coverage(page)
    if coverage >= IMAGE_COVERAGE:
        return "scanned"
    if text_len < MIN_TEXT_CHARS:
        return "scanned" if coverage > 0.3 else "blank"
    return "digital"


def classify_and_render(pdf_path: Path, pages_dir: Path) -> list[dict]:
    """Label each page digital/scanned/blank and render it to PNG."""
    pages_dir.mkdir(parents=True, exist_ok=True)
    info = []
    with fitz.open(pdf_path) as pdf:
        for i, page in enumerate(pdf, start=1):
            img_path = pages_dir / f"page_{i}.png"
            page.get_pixmap(dpi=RENDER_DPI).save(img_path)
            info.append({
                "page": i,
                "type": classify_page(page),
                "width": page.rect.width,
                "height": page.rect.height,
                "image": str(img_path.relative_to(pages_dir.parent)),
            })
    return info


def build_converter(use_ocr: bool, force_ocr: bool = False) -> DocumentConverter:
    opts = PdfPipelineOptions()
    opts.do_ocr = use_ocr
    # force_ocr: ignore hidden text layers on scanned pages and re-read the image
    opts.ocr_options.force_full_page_ocr = force_ocr
    opts.do_table_structure = True
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


def extract_elements(pdf_path: Path, use_ocr: bool, force_ocr: bool = False) -> list[dict]:
    """Parse with Docling into text / table / figure elements with page + bbox."""
    doc = build_converter(use_ocr, force_ocr).convert(pdf_path).document
    elements = []
    for item, _level in doc.iterate_items():
        if not getattr(item, "prov", None):
            continue
        prov = item.prov[0]
        page_h = doc.pages[prov.page_no].size.height
        b = prov.bbox.to_top_left_origin(page_height=page_h)
        bbox = [round(b.l, 1), round(b.t, 1), round(b.r, 1), round(b.b, 1)]

        if isinstance(item, TableItem):
            kind, content = "table", item.export_to_markdown(doc=doc)
        elif isinstance(item, PictureItem):
            caption = item.caption_text(doc=doc) if hasattr(item, "caption_text") else ""
            kind, content = "figure", caption or ""
        elif isinstance(item, TextItem):
            if not item.text.strip():
                continue
            kind, content = "text", item.text.strip()
        else:
            continue

        elements.append({
            "id": f"e{len(elements)}",
            "page": prov.page_no,
            "type": kind,
            "label": str(getattr(item, "label", kind)),
            "content": content,
            "bbox": bbox,  # top-left origin, PDF points -> used for citation highlights
        })
    return elements


def ingest(pdf_path: Path, out_root: Path) -> Path:
    t0 = time.time()
    doc_id = file_hash(pdf_path)
    out_dir = out_root / doc_id
    if (out_dir / "elements.json").exists():
        print(f"[cache] {pdf_path.name} already processed -> {out_dir}")
        return out_dir

    pages = classify_and_render(pdf_path, out_dir / "pages")
    scanned = [p["page"] for p in pages if p["type"] == "scanned"]
    needs_ocr = bool(scanned)
    # mostly-scanned document -> re-OCR everything instead of trusting hidden text
    force_ocr = len(scanned) >= 0.5 * len(pages)
    elements = extract_elements(pdf_path, use_ocr=needs_ocr, force_ocr=force_ocr)

    (out_dir / "elements.json").write_text(json.dumps(elements, indent=2, ensure_ascii=False))
    meta = {
        "doc_id": doc_id,
        "filename": pdf_path.name,
        "num_pages": len(pages),
        "pages": pages,
        "ocr_used": needs_ocr,
        "ocr_forced": force_ocr,
        "page_types": {t: sum(p["type"] == t for p in pages) for t in ("digital", "scanned", "blank")},
        "counts": {k: sum(e["type"] == k for e in elements) for k in ("text", "table", "figure")},
        "seconds": round(time.time() - t0, 1),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] {pdf_path.name}: {meta['num_pages']} pages {meta['page_types']}, "
          f"{meta['counts']}, OCR={needs_ocr} forced={force_ocr}, {meta['seconds']}s -> {out_dir}")
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    args = ap.parse_args()
    ingest(args.pdf, args.out)