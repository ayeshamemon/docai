"""
Step 6.1: the agent's tools. Each tool does one thing and returns a ToolResult:
  - text:   what the agent reads (short, numbered by chunk id, with pages)
  - chunks: the chunks found (kept for the final answer and its citations)
  - pages:  page images found (sent to Gemini with the final answer)

Tools (start small; add more only if the logs show they are needed):
  text_search(query)     normal search: hybrid + reranker, top 5
  graph_search(query)    GraphRAG: rerank top 3 + 2 connected chunks from the graph
  visual_search(query)   page images (charts, scans), top 2
  read_next(chunk)       the chunk before and after (an explanation right under a table)
  find_exact(term)       every chunk with this exact word / number
  calculate(expression)  exact arithmetic (no guessing numbers)

Usage (test from backend/, API server stopped):
    python -m agent.tools <doc_id> text_search "net sales by product 2026"
    python -m agent.tools <doc_id> read_next c112
    python -m agent.tools <doc_id> find_exact "h = 8"
    python -m agent.tools <doc_id> calculate "(254098 - 209586) / 209586 * 100"
"""

import ast
import json
import operator
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROCESSED = Path("data/processed")
TEXT_K = 5
VISUAL_K = 2
EXACT_MAX = 5
SHOW_CHARS = 1200            # how much of each chunk the agent reads per result


@dataclass
class ToolResult:
    text: str
    chunks: list[dict] = field(default_factory=list)
    pages: list[dict] = field(default_factory=list)


def short_id(chunk_id: str) -> str:
    """'bdfaa68d8984f0dc_c23' -> 'c23' (what the agent sees and passes back)."""
    return chunk_id.rsplit("_", 1)[-1]


def show(chunks: list[dict]) -> str:
    """Chunks as the agent reads them: [c23] p. 7 (text) heading / text."""
    if not chunks:
        return "Nothing found."
    out = []
    for c in chunks:
        pages = ", ".join(map(str, c["pages"]))
        tag = " · via graph" if c.get("from_graph") else ""
        out.append(f"[{short_id(c['chunk_id'])}] p. {pages} ({c['type']}){tag}\n"
                   f"{c['text'][:SHOW_CHARS]}")
    return "\n\n".join(out)


# ---------- safe calculator (no eval) ----------

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"round": round, "abs": abs, "min": min, "max": max}


def safe_calc(expression: str) -> float:
    """Numbers, + - * / ** %, brackets, round/abs/min/max. Anything else is refused."""
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in _FUNCS and not node.keywords):
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ValueError("only numbers, + - * / ** %, brackets and round/abs/min/max are allowed")
    expr = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", expression)   # "254,098" -> 254098 (not round(x, 1))
    expr = expr.replace("$", "").strip()          # "%" stays: it means remainder, not percent
    if len(expr) > 200:
        raise ValueError("expression too long")
    return ev(ast.parse(expr, mode="eval"))


# ---------- the toolbox ----------

class Toolbox:
    """All tools for ONE document (the agent answers about the open document)."""

    DESCRIPTIONS = {
        "text_search": "Search the document by meaning and keywords (best general search). "
                       "Arg: query, a short search phrase. Rephrase if results miss.",
        "graph_search": "Search that also follows connections between things in the document "
                        "(use when you need a fact linked to something you already found). Arg: query.",
        "visual_search": "Find page images: charts, figures, scanned or handwritten pages. "
                         "The page images are given to you with the final answer. Arg: query.",
        "read_next": "Read the chunk right before and after a chunk you already have "
                     "(e.g. the explanation under a table). Arg: chunk, an id like c23.",
        "find_exact": "List chunks containing this exact word, name or number "
                      "(use for exact terms, codes, numbers). Arg: term.",
        "calculate": "Exact arithmetic on numbers you found, e.g. growth: (new - old) / old * 100. "
                     "Arg: expression.",
    }

    def __init__(self, doc_id: str, client=None):
        self.doc_id = doc_id
        self.client = client
        self._chunks = None

    # chunks.json in reading order (for read_next / find_exact)
    @property
    def chunks(self) -> list[dict]:
        if self._chunks is None:
            self._chunks = json.loads((PROCESSED / self.doc_id / "chunks.json").read_text())
        return self._chunks

    def full_id(self, chunk: str) -> str:
        chunk = chunk.strip().strip("[]")
        return chunk if chunk.startswith(self.doc_id) else f"{self.doc_id}_{chunk}"

    def has_graph(self) -> bool:
        return (PROCESSED / self.doc_id / "graph" / "graph.json").exists()

    def run(self, tool: str, arg: str) -> ToolResult:
        fn = getattr(self, tool, None)
        if tool not in self.DESCRIPTIONS or fn is None:
            return ToolResult(f"Unknown tool '{tool}'. Tools: {', '.join(self.DESCRIPTIONS)}.")
        if not str(arg).strip():
            return ToolResult(f"{tool} needs an argument.")
        try:
            return fn(str(arg))
        except Exception as e:                       # a tool error is information, not a crash
            return ToolResult(f"{tool} failed: {e}")

    # --- search tools (reuse the existing retrieval) ---

    def text_search(self, query: str) -> ToolResult:
        from retrieval.search import search
        hits = search(query, self.client, k=TEXT_K, doc_id=self.doc_id, mode="rerank")
        return ToolResult(show(hits), chunks=hits)

    def graph_search(self, query: str) -> ToolResult:
        if not self.has_graph():
            return ToolResult("No graph for this document; use text_search instead.")
        from retrieval.search import search
        hits = search(query, self.client, k=TEXT_K, doc_id=self.doc_id, mode="graph+rerank")
        return ToolResult(show(hits), chunks=hits)

    def visual_search(self, query: str) -> ToolResult:
        from retrieval.search import search
        try:
            pages = search(query, self.client, k=VISUAL_K, doc_id=self.doc_id, mode="visual")
        except Exception:
            pages = []
        if not pages:
            return ToolResult("No page images for this document (no charts or scanned pages).")
        lines = [f"page {p['pages'][0]} ({p.get('reason', 'image')}): image will be shown to you "
                 f"with the final answer" for p in pages]
        return ToolResult("\n".join(lines), pages=pages)

    # --- reading tools (from chunks.json, no model) ---

    def read_next(self, chunk: str) -> ToolResult:
        cid = self.full_id(chunk)
        ids = [c["chunk_id"] for c in self.chunks]
        if cid not in ids:
            return ToolResult(f"No chunk {chunk}. Use an id from a search result, like c23.")
        i = ids.index(cid)
        around = [self.chunks[j] for j in (i - 1, i + 1) if 0 <= j < len(self.chunks)]
        return ToolResult(show(around), chunks=around)

    def find_exact(self, term: str) -> ToolResult:
        t = term.strip().lower()
        if len(t) < 2:
            return ToolResult("Term too short.")
        hits = [c for c in self.chunks if t in c["text"].lower()]
        if not hits:
            return ToolResult(f'No chunk contains "{term}" exactly. Try a shorter or different form.')
        note = (f'"{term}" is in {len(hits)} chunks; showing the first {EXACT_MAX}. '
                f"Use a more specific term if needed.\n\n" if len(hits) > EXACT_MAX else "")
        return ToolResult(note + show(hits[:EXACT_MAX]), chunks=hits[:EXACT_MAX])

    # --- calculator ---

    def calculate(self, expression: str) -> ToolResult:
        value = safe_calc(expression)
        nice = f"{value:,.4f}".rstrip("0").rstrip(".") if isinstance(value, float) else f"{value:,}"
        return ToolResult(f"{expression} = {nice}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        raise SystemExit(1)
    doc_id, tool, arg = sys.argv[1], sys.argv[2], " ".join(sys.argv[3:])
    client = None
    if tool in ("text_search", "graph_search", "visual_search"):
        from ingest.index import get_client
        client = get_client()
    print(Toolbox(doc_id, client).run(tool, arg).text)
    if client:
        client.close()