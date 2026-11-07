"""
Step 9.2: the agent loop (agentic RAG).

Instead of searching once, the agent works like a researcher:
  round 1   split the question into parts (a checklist) and pick a first tool
  round 2+  read what was found, tick off parts, pick the next tool
  stop      when every part has a source (or after MAX_ROUNDS)
  answer    the SAME answer step as normal mode (prompt, [p. N] citations, page images)

Each round Gemini returns one small JSON note (ReAct: one line of reasoning, then one action):
  {"thought": "It's iPhone, now I need why",
   "checklist": [{"part": "which product grew fastest", "source": "c111"},
                 {"part": "why it grew", "source": null}],
   "tool": "read_next", "arg": "c111"}

Stopping rules:
  1. "answer" is accepted only when every checklist part has a source the agent really found
  2. at most MAX_ROUNDS rounds; then it answers with what it has (the answer says what is missing)
  3. the same tool + argument twice is refused (forces a new idea)
  4. tool errors come back as text, never a crash

Usage (run from backend/, API server stopped):
    python -m agent.agent "Which product grew fastest in 2026, and why?" --doc f9a3e5b32cb7cbf2
"""

import argparse
import json
import time

from agent.tools import Toolbox, short_id

MAX_ROUNDS = 5
MAX_CHUNKS = 10              # chunks passed to the final answer (in the order found)
MAX_PAGES = 3                # page images passed to the final answer

PLAN_PROMPT = """You are a research agent answering a question about ONE document.
You cannot see the document; you can only use the tools below, one per round.

TOOLS
{tools}
answer: stop searching and write the answer. Arg: "" (only when every checklist part has a source)

HOW TO WORK
- First split the question into its parts (a checklist), e.g. "which product grew fastest" and "why".
- Search for one part at a time with SHORT, specific queries. Use what you already found to write
  the next query (e.g. after finding that the product is iPhone, search "why did iPhone sales increase").
- An explanation often sits right after a table: try read_next on the table's chunk id.
- For an exact name, code or number use find_exact.
- For a part, give the chunk id (like c23) or "page N" (for a page image) that answers it, or null.
- Never invent sources. If something is not in the document after a few tries, leave it null and answer.

QUESTION
{question}

ROUND {round} of {max_rounds}
{notes}

Reply with JSON only:
{{"thought": "one short sentence",
  "checklist": [{{"part": "...", "source": "c23" or null}}],
  "tool": "one tool name", "arg": "query, chunk id, term or expression"}}"""


class Agent:
    def __init__(self, doc_id: str, client=None, on_step=None, verbose=False):
        self.tools = Toolbox(doc_id, client)
        self.on_step = on_step            # callback for live steps (the app, step 9.6)
        self.verbose = verbose

    # ---------- one run ----------

    def run(self, question: str) -> dict:
        from graph.extract import call_gemini        # JSON call with retries + fallback

        found: dict[str, dict] = {}       # chunk id (short) -> chunk, in the order found
        pages: dict[int, dict] = {}       # page number -> page image hit
        history, done_calls, steps = [], set(), []
        checklist: list[dict] = []
        t0 = time.time()

        for rnd in range(1, MAX_ROUNDS + 1):
            notes = self._notes(history, checklist)
            prompt = PLAN_PROMPT.format(
                tools="\n".join(f"{n}: {d}" for n, d in self.tools.DESCRIPTIONS.items()
                                if n != "graph_search" or self.tools.has_graph()),
                question=question, round=rnd, max_rounds=MAX_ROUNDS, notes=notes)
            try:
                move = call_gemini(prompt)
            except Exception as e:                    # planner failed: answer with what we have
                steps.append({"round": rnd, "error": str(e)})
                break
            checklist = move.get("checklist") or checklist
            tool, arg = (move.get("tool") or "").strip(), str(move.get("arg") or "").strip()
            step = {"round": rnd, "thought": move.get("thought", ""), "tool": tool, "arg": arg,
                    "checklist": checklist}

            if tool == "answer":
                missing = self._unsourced(checklist, found, pages)
                if missing and rnd < MAX_ROUNDS:      # rule 1: not allowed to stop yet
                    msg = ("Not every part has a source you found: " + "; ".join(missing)
                           + ". Search for them, or set their source to null if truly not in the document.")
                    history.append((tool, arg, msg))
                    step["result"] = msg
                    self._emit(step, steps)
                    continue
                self._emit(step, steps)
                break

            key = (tool, arg.lower())
            if key in done_calls:                     # rule 3: no repeats
                msg = f"You already ran {tool}({arg!r}). Try a different tool or query."
                history.append((tool, arg, msg))
                step["result"] = msg
                self._emit(step, steps)
                continue
            done_calls.add(key)

            res = self.tools.run(tool, arg)           # rule 4: errors come back as text
            for c in res.chunks:
                found.setdefault(short_id(c["chunk_id"]), c)
            for p in res.pages:
                pages.setdefault(p["pages"][0], p)
            history.append((tool, arg, res.text))
            step["result_chunks"] = [short_id(c["chunk_id"]) for c in res.chunks]
            step["result_pages"] = [p["pages"][0] for p in res.pages]
            self._emit(step, steps)

        hits = list(found.values())[:MAX_CHUNKS]
        return {"hits": hits, "pages": list(pages.values())[:MAX_PAGES], "steps": steps,
                "checklist": checklist, "seconds": round(time.time() - t0, 1)}

    # ---------- helpers ----------

    def _notes(self, history, checklist) -> str:
        if not history:
            return "Nothing searched yet."
        parts = []
        if checklist:
            parts.append("CHECKLIST SO FAR\n" + "\n".join(
                f"- {c.get('part')}: {c.get('source') or 'not found yet'}" for c in checklist))
        parts.append("WHAT YOU FOUND (oldest first)")
        for i, (tool, arg, text) in enumerate(history, 1):
            parts.append(f"--- step {i}: {tool}({arg!r})\n{text}")
        return "\n\n".join(parts)

    @staticmethod
    def _unsourced(checklist, found, pages) -> list[str]:
        """Parts whose source is missing or was never actually found (no invented sources)."""
        bad = []
        for c in checklist:
            src = str(c.get("source") or "").strip().lower()
            if src in ("", "none", "null"):
                continue                               # declared not in the document: allowed
            if src.startswith("page"):
                ok = any(str(p) in src for p in pages)
            else:
                ok = src.strip("[]") in found
            if not ok:
                bad.append(f"{c.get('part')} (source {c.get('source')} was not found by a tool)")
        if not checklist:
            bad.append("no checklist yet")
        return bad

    def _emit(self, step, steps):
        steps.append(step)
        if self.on_step:
            self.on_step(step)
        if self.verbose:
            got = step.get("result_chunks") or step.get("result_pages") or step.get("result", "")
            print(f"  [{step['round']}] {step['thought']}\n      -> {step['tool']}({step['arg']!r})  {got}",
                  flush=True)


def retrieve_agent(question: str, doc_id: str, client=None, verbose=False):
    """Same shape as ask.retrieve: (hits, pages), plus the steps for logging."""
    out = Agent(doc_id, client, verbose=verbose).run(question)
    return out["hits"], out["pages"], out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--doc", required=True)
    ap.add_argument("--no-answer", action="store_true", help="only show the steps")
    args = ap.parse_args()

    from ingest.index import get_client
    client = get_client()
    print(f"Q: {args.question}\n")
    hits, pages, out = retrieve_agent(args.question, args.doc, client, verbose=True)
    print(f"\nchecklist: {json.dumps(out['checklist'], ensure_ascii=False)}")
    print(f"collected: {[short_id(h['chunk_id']) for h in hits]}  pages: {[p['pages'][0] for p in pages]}"
          f"  ({out['seconds']} s)")
    if not args.no_answer:
        from retrieval.ask import build_contents, generate
        print("\nANSWER\n")
        for text in generate(build_contents(args.question, hits, pages)):
            print(text, end="", flush=True)
        print()
    client.close()