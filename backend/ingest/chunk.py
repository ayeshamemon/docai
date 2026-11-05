"""
Step 2.1 — Chunking.

Turns elements.json (small pieces from parse.py) into chunks.json
(meaningful blocks we will search later).

Rule-based, structure-aware chunking (no model). Rules:
  1. A heading starts a new chunk
  2. Each chunk gets its heading on top
  3. Paragraphs are joined whole, up to 300 words (MAX_WORDS, heading included)
  4. Tables are their own chunk; big ones are split by rows with the header repeated
     (caption/header repeated only if short, <= 40 words; otherwise first part only)
  5. Figures with captions are their own chunk (figures without captions are skipped)
  6. The table intro sentence ("The following table shows...") is attached to its table
  7. Tiny text chunks under 30 words (MIN_WORDS) are merged into a neighbour
  8. A giant paragraph (longer than the limit by itself) is split at sentence ends (rare)
Every chunk keeps its pages and bounding boxes, for clickable citations.

Usage:
    python ingest/chunk.py data/processed/<doc_id>                  # default 300 words
    python ingest/chunk.py data/processed/<doc_id> --max-words 150  # other size
    python ingest/chunk.py data/processed/<doc_id> --compare        # experiment: 150/300/500
"""

import argparse
import json
import re
from pathlib import Path

MAX_WORDS = 300          # target size of a text chunk (~400 tokens)
EMBED_TOKEN_LIMIT = 512  # bge-small reads at most 512 tokens, the rest is cut off
TOKENS_PER_WORD = 1.33   # rough estimate for English text
HEADING_LABELS = {"section_header", "title"}
MIN_WORDS = 30           # text chunks smaller than this get merged into a neighbour
MAX_REPEAT_WORDS = 40    # split tables repeat caption/header only if they are this short
INTRO_MAX_WORDS = 60     # a table intro sentence is short
INTRO_PATTERN = re.compile(r"(following table|table below|tables below|as follows|summar(y|izes))",
                           re.IGNORECASE)


def words(text: str) -> int:
    return len(text.split())


def split_table(md: str, budget: int) -> list[str]:
    """
    Split a big Markdown table into smaller tables by rows.
    Parts repeat the caption and the header row so each part still knows what
    its columns mean (e.g. which column is 2026), but ONLY if they are short.
    A long caption or "header" (e.g. a scanned name list read as one table, where
    the header row is full of names) appears in the first part only; repeating it
    would create dozens of near-identical chunks.
    """
    lines = md.split("\n")
    table_idx = [i for i, l in enumerate(lines) if l.lstrip().startswith("|")]
    if len(table_idx) < 3:                       # header + separator + at least one row
        return [md]
    first = table_idx[0]
    caption = "\n".join(lines[:first]).strip()
    header = lines[first:first + 2]              # header row + |---| line
    rows = [lines[i] for i in table_idx[2:]]

    # what later parts repeat: only short captions / headers
    rep_caption = caption if words(caption) <= MAX_REPEAT_WORDS else ""
    rep_header = header if words(header[0]) <= MAX_REPEAT_WORDS else []
    first_fixed = words(caption) + words("\n".join(header))
    later_fixed = words(rep_caption) + words("\n".join(rep_header)) + 2

    parts, current = [], []
    for row in rows:
        fixed = first_fixed if not parts else later_fixed
        if current and fixed + words("\n".join(current + [row])) > budget:
            parts.append(current)
            current = []
        current.append(row)
    if current:
        parts.append(current)

    out = []
    for i, p in enumerate(parts):
        if i == 0:
            cap, head = caption, header
        else:
            cap, head = rep_caption, rep_header
            if not cap or not head:                # something was left out: say so
                cap = (cap + "\n\n" if cap else "") + "(table continued)"
        out.append(((cap + "\n\n") if cap else "") + "\n".join(head + p))
    return out


def split_paragraph(text: str, budget: int) -> list[str]:
    """Split a very long paragraph at sentence ends (never mid-sentence if possible)."""
    sentences = re.split(r"(?<=[.!?;])\s+", text)
    parts, current = [], ""
    for s in sentences:
        if current and words(current) + words(s) > budget:
            parts.append(current)
            current = ""
        # a single sentence longer than the budget: cut it by words
        while words(s) > budget:
            w = s.split()
            parts.append(" ".join(w[:budget]))
            s = " ".join(w[budget:])
        current = f"{current} {s}".strip()
    if current:
        parts.append(current)
    return parts


def make_chunk(doc_id, idx, ctype, heading, parts, text_override=None):
    """parts = list of elements that go into this chunk (used for pages + boxes).
    text_override = use this as the body instead of joining the parts' content."""
    body = text_override if text_override is not None else "\n\n".join(p["content"] for p in parts)
    text = f"{heading}\n\n{body}" if heading else body
    return {
        "chunk_id": f"{doc_id}_c{idx}",
        "doc_id": doc_id,
        "type": ctype,                      # text / table / figure
        "heading": heading,
        "text": text,                       # what we embed and send to the LLM
        "pages": sorted({p["page"] for p in parts}),
        "boxes": [{"page": p["page"], "bbox": p["bbox"]} for p in parts],  # for highlights
    }


def chunk_document(doc_dir: Path, max_words: int = MAX_WORDS) -> list[dict]:
    meta = json.loads((doc_dir / "meta.json").read_text())
    elements = json.loads((doc_dir / "elements.json").read_text())
    doc_id = meta["doc_id"]

    chunks, buffer = [], []
    heading = ""

    def flush():
        """Close the current text chunk, if any."""
        if buffer:
            chunks.append(make_chunk(doc_id, len(chunks), "text", heading, buffer))
            buffer.clear()

    for el in elements:
        content = el["content"].strip()

        # Rules 1+2: a heading closes the current chunk and becomes the heading of the next ones
        if el["type"] == "text" and el["label"] in HEADING_LABELS:
            flush()
            heading = content
            continue

        budget = max_words - words(heading)  # room left after the heading

        # Rule 4: tables are their own chunk; big tables are split by rows
        if el["type"] == "table":
            # Rule 6: pull the table's intro sentence ("The following table shows...")
            # out of the text buffer and put it on top of the table
            parts_for_box = []
            if buffer and is_table_intro(buffer[-1]["content"]):
                intro_el = buffer.pop()
                content = intro_el["content"].strip() + "\n\n" + content
                parts_for_box = [intro_el]
            flush()
            # the intro sits before the first "|" line, so split_table repeats it in every part
            pieces = split_table(content, budget) if words(content) > budget else [content]
            for piece in pieces:
                chunks.append(make_chunk(doc_id, len(chunks), "table", heading,
                                         parts_for_box + [{**el, "content": piece}],
                                         text_override=piece))
            continue

        # Rule 5: figures only if they have a caption (charts get handled by ColPali later)
        if el["type"] == "figure":
            if content:
                flush()
                chunks.append(make_chunk(doc_id, len(chunks), "figure", heading, [el]))
            continue

        # Rule 3: normal text — add whole paragraphs to the buffer, close it when it gets too big
        if not content:
            continue
        # Rule 8: a single giant paragraph is split at sentence ends
        if words(content) > budget:
            flush()
            for piece in split_paragraph(content, budget):
                chunks.append(make_chunk(doc_id, len(chunks), "text", heading,
                                         [{**el, "content": piece}]))
            continue
        if buffer and words(" ".join(b["content"] for b in buffer)) + words(content) > max_words:
            flush()
        buffer.append(el)

    flush()
    chunks = merge_tiny(chunks, max_words)
    for i, c in enumerate(chunks):                 # renumber after merging
        c["chunk_id"] = f"{doc_id}_c{i}"
    return chunks


def is_table_intro(text: str) -> bool:
    """A short sentence that introduces the table right after it."""
    t = text.strip()
    return words(t) <= INTRO_MAX_WORDS and (t.endswith(":") or bool(INTRO_PATTERN.search(t)))


def merge_two(a: dict, b: dict) -> dict:
    """Join two text chunks into one (a's text first). Keeps b's heading as the label."""
    return {
        **b,
        "text": a["text"] + "\n\n" + b["text"],
        "pages": sorted(set(a["pages"]) | set(b["pages"])),
        "boxes": a["boxes"] + b["boxes"],
    }


def merge_tiny(chunks: list[dict], max_words: int) -> list[dict]:
    """
    Rule 7: a text chunk under MIN_WORDS is too vague to search well.
    Attach it to the NEXT text chunk (if the result still fits), else to the PREVIOUS one,
    else keep it. Tables and figures are never merged.
    """
    out, pending = [], None
    for c in chunks:
        if pending is not None:
            if c["type"] == "text" and words(pending["text"]) + words(c["text"]) <= max_words:
                c, pending = merge_two(pending, c), None
            else:
                prev = out[-1] if out else None
                if prev and prev["type"] == "text" and \
                        words(prev["text"]) + words(pending["text"]) <= max_words:
                    out[-1] = merge_two(prev, pending)
                else:
                    out.append(pending)
                pending = None
        if c["type"] == "text" and words(c["text"]) < MIN_WORDS:
            pending = c
            continue
        out.append(c)

    if pending is not None:                         # tiny chunk at the very end
        prev = out[-1] if out else None
        if prev and prev["type"] == "text" and \
                words(prev["text"]) + words(pending["text"]) <= max_words:
            out[-1] = merge_two(prev, pending)
        else:
            out.append(pending)
    return out


def stats(chunks: list[dict]) -> dict:
    sizes = [words(c["text"]) for c in chunks] or [0]
    too_long = [c for c in chunks if words(c["text"]) * TOKENS_PER_WORD > EMBED_TOKEN_LIMIT]
    return {
        "chunks": len(chunks),
        "text": sum(c["type"] == "text" for c in chunks),
        "table": sum(c["type"] == "table" for c in chunks),
        "figure": sum(c["type"] == "figure" for c in chunks),
        "avg_words": sum(sizes) // len(sizes),
        "min_words": min(sizes),
        "max_words": max(sizes),
        "over_limit": len(too_long),                                  # would get cut off
        "over_limit_tables": sum(c["type"] == "table" for c in too_long),
        "tiny": sum(c["type"] == "text" and words(c["text"]) < MIN_WORDS for c in chunks),
    }


def compare(doc_dir: Path, sizes=(150, 300, 500)):
    """Experiment: same document, different chunk sizes. Prints a table, saves nothing."""
    rows = [(n, stats(chunk_document(doc_dir, n))) for n in sizes]
    cols = ["chunks", "text", "table", "tiny", "avg_words", "min_words", "max_words",
            "over_limit", "over_limit_tables"]
    print(f"\nChunk-size experiment: {doc_dir.name}  (token limit {EMBED_TOKEN_LIMIT})\n")
    print(f"{'size':>6} " + " ".join(f"{c:>17}" for c in cols))
    for n, s in rows:
        print(f"{n:>6} " + " ".join(f"{s[c]:>17}" for c in cols))
    print("\nover_limit = chunks longer than ~512 tokens (their end would be ignored by the embedder)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("doc_dir", type=Path)
    ap.add_argument("--max-words", type=int, default=MAX_WORDS)
    ap.add_argument("--compare", action="store_true", help="run the 150/300/500 experiment")
    args = ap.parse_args()

    if args.compare:
        compare(args.doc_dir)
    else:
        chunks = chunk_document(args.doc_dir, args.max_words)
        out = args.doc_dir / "chunks.json"
        out.write_text(json.dumps(chunks, indent=2, ensure_ascii=False))
        s = stats(chunks)
        print(f"[done] {s['chunks']} chunks (text {s['text']}, table {s['table']}, figure {s['figure']}, tiny {s['tiny']}), "
              f"avg {s['avg_words']} words, {s['over_limit']} over token limit -> {out}")