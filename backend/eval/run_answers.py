"""
Step 4.1 — Answer evaluation: is the FINAL answer right, not just the retrieval?

For every question in eval/questions.jsonl:
  1. retrieve exactly like the app (rerank, top 5 chunks, up to 3 page images)
  2. answer with ONE model, no fallback (fair comparison)
  3. judge the answer against the reference with a DIFFERENT model:
       correct / partial / wrong / abstained
  4. check citations without any model:
       cites_evidence: a cited page holds the answer (keyword in a retrieved chunk on that
                       page; for charts/scans without a keyword: the expected page)
       grounded:       every cited page is a page the model was actually given

Results are saved after every question (eval/answers.jsonl), so a crash or a 503 storm
loses nothing: just run it again and it continues.

Usage (from backend/, with the API server STOPPED — local Qdrant allows one process):
    python -m eval.run_answers                        # answer + judge everything
    python -m eval.run_answers --only T8 T11 Q26      # a few questions
    python -m eval.run_answers --redo T8              # answer T8 again
    python -m eval.run_answers --rejudge              # judge again, keep the answers
    python -m eval.run_answers --summary              # only print the summary
    python -m eval.run_answers --review 15            # you grade 15 random answers
"""

import argparse
import csv
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from google.genai import errors, types

import retrieval.ask as ask
from ingest.index import get_client
from retrieval.ask import build_contents, generate, get_llm, retrieve

HERE = Path(__file__).parent
QUESTIONS = HERE / "questions.jsonl"
ANSWERS = HERE / "answers.jsonl"   # rerank; other modes: answers_<mode>.jsonl
MODE = "rerank"
REVIEW = HERE / "human_review.jsonl"
RESULTS_CSV = HERE / "results_answers.csv"

ANSWER_MODEL = "gemini-3.6-flash"
JUDGE_MODEL = "gemini-3.8-flash"
VERDICTS = ["correct", "partial", "wrong", "abstained"]
PAUSE_S = 2.0                      # between questions, to stay under rate limits

JUDGE_PROMPT = """You grade answers from a document question-answering system.

Question: {question}
Reference answer (from the document): {reference}
System answer: {answer}

Choose exactly one verdict:
- "correct": the system answer states the key fact(s) of the reference. Wording, units and
  formatting may differ. If the reference gives an approximate value (~, about), a value close
  to it is fine. Extra details are fine unless they contradict the reference.
- "partial": some key fact is right but another is wrong or missing, or a value is clearly off
  (e.g. the right chart series but a peak value far from the reference).
- "wrong": the main answer differs from the reference, or it contradicts it.
- "abstained": the system says the information is not in the document / it could not find it,
  without giving a substantive answer.

Return JSON only: {{"verdict": "correct|partial|wrong|abstained", "reason": "<= 25 words"}}"""

# Citations: any [...] bracket; inside it, every number list that follows p./pp./page counts.
# Handles [p. 8]  [p. 3, 4]  [pp. 3-5]  [page 12]  [pages 3 and 5]  and labelled forms like
# [Source 4, p. 2, 3; Page image, p. 3]. Brackets without p./page (e.g. [Table 2]) are ignored.
BRACKET = re.compile(r"\[([^\]\n]{1,120})\]")
PAGE_LIST = re.compile(r"(?:\bpp?\.|\bpages?\b)\s*(\d+(?:\s*(?:[–-]|,|and|&)\s*\d+)*)", re.I)


# ---------- helpers ----------

def load_questions() -> list[dict]:
    return [json.loads(l) for l in QUESTIONS.read_text().splitlines() if l.strip()]


def load_results(path: Path | None = None) -> dict[str, dict]:
    path = path or ANSWERS
    if not path.exists():
        return {}
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return {r["id"]: r for r in rows}           # later lines win (redo / rejudge)


def save_result(row: dict, path: Path | None = None):
    with open(path or ANSWERS, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def cited_pages(text: str) -> list[int]:
    """Page numbers cited in an answer (same rules as the frontend)."""
    pages = []
    for inner in BRACKET.findall(text):
        for group in PAGE_LIST.findall(inner):
            for part in re.split(r",|\band\b|&", group):
                nums = [int(x) for x in re.findall(r"\d+", part)]
                if len(nums) == 2 and 0 <= nums[1] - nums[0] < 20:
                    pages += list(range(nums[0], nums[1] + 1))
                elif nums:
                    pages.append(nums[0])
    return sorted(set(pages))


def citation_checks(q: dict, answer: str, hits: list[dict], pages_sent: list[dict]) -> dict:
    cited = set(cited_pages(answer))
    given = {p for h in hits for p in h["pages"]} | {p["page"] for p in pages_sent}
    keywords = q.get("keywords") or ([q["keyword"]] if q.get("keyword") else [])

    def pages_with(kw):
        return {p for h in hits if kw.lower() in h["text"].lower() for p in h["pages"]}

    if keywords:
        # every part of the evidence must be cited (multi-hop: all parts; else the one keyword).
        # A part that was not retrieved at all falls back to its expected page.
        expected = q.get("pages") or [q["page"]]
        parts = [pages_with(k) or ({expected[i]} if i < len(expected) else set())
                 for i, k in enumerate(keywords)]
        parts = [p for p in parts if p]
        cites_evidence = all(cited & p for p in parts) if parts else None
    else:
        evidence_pages = {q["page"]} if q["page"] else set()
        cites_evidence = bool(cited & evidence_pages) if evidence_pages else None

    return {
        "cited_pages": sorted(cited),
        "cites_evidence": cites_evidence,
        "grounded": all(p in given for p in cited) if cited else None,
    }


def with_retries(fn, what: str, tries: int = 4):
    """Retry Gemini calls on busy / rate-limit / timeout errors."""
    for attempt in range(tries):
        try:
            return fn()
        except (errors.ServerError, errors.ClientError) as e:
            code = getattr(e, "code", None)
            if code not in (429, 500, 503, 504) or attempt == tries - 1:
                raise
            wait = 5 * 2 ** attempt
            print(f"    [{what}: busy ({code}), retrying in {wait}s]", flush=True)
            time.sleep(wait)


# ---------- answer + judge ----------

def answer_question(q: dict, client) -> dict:
    t0 = time.time()
    hits, pages = retrieve(q["question"], q["doc"], mode=MODE, client=client)   # same as the app
    t_retrieve = time.time() - t0
    text = "".join(generate(build_contents(q["question"], hits, pages)))
    return {
        "id": q["id"], "category": q["category"], "doc": q["doc"],
        "question": q["question"], "reference": q["answer"],
        "answer": text.strip(),
        "answer_model": ask.MODEL,
        "sources": [{"pages": h["pages"], "type": h["type"]} for h in hits],
        "page_images": [p["page"] for p in pages],
        **citation_checks(q, text, hits, pages),
        "retrieve_s": round(t_retrieve, 1),
        "total_s": round(time.time() - t0, 1),
    }


def judge(row: dict, judge_model: str) -> dict:
    prompt = JUDGE_PROMPT.format(question=row["question"], reference=row["reference"],
                                 answer=row["answer"] or "(empty)")
    resp = with_retries(lambda: get_llm().models.generate_content(
        model=judge_model, contents=prompt,
        config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json"),
    ), "judge")
    try:
        out = json.loads(resp.text)
        verdict = out.get("verdict", "").strip().lower()
        if verdict not in VERDICTS:
            raise ValueError(verdict)
        return {"verdict": verdict, "reason": out.get("reason", ""), "judge_model": judge_model}
    except Exception:
        return {"verdict": None, "reason": f"unparseable judge output: {resp.text[:120]}",
                "judge_model": judge_model}


# ---------- summary ----------

def pct(n, d):
    return f"{100 * n / d:.0f}%" if d else "  -"


def summarize(rows: list[dict]):
    rows = [r for r in rows if r.get("verdict")]
    if not rows:
        print("[summary] nothing judged yet")
        return
    # take the category from questions.jsonl, so relabelled questions group correctly
    current = {}
    qfile = Path(__file__).parent / "questions.jsonl"
    for l in qfile.read_text().splitlines():
        if l.strip():
            q = json.loads(l)
            current[q["id"]] = q["category"]
    for r in rows:
        r["category"] = current.get(r["id"], r["category"])
    groups = defaultdict(list)
    for r in rows:
        groups[r["category"]].append(r)
    groups["ALL"] = rows

    head = f"{'category':<18}{'n':>4}  {'correct':>8}{'partial':>8}{'wrong':>7}{'abstain':>8}   {'evidence':>9}{'grounded':>9}{'avg s':>7}"
    print("\n" + head + "\n" + "-" * len(head))
    for cat in sorted(k for k in groups if k != "ALL") + ["ALL"]:
        g = groups[cat]
        c = Counter(r["verdict"] for r in g)
        ev = [r["cites_evidence"] for r in g if r["cites_evidence"] is not None]
        gr = [r["grounded"] for r in g if r["grounded"] is not None]
        avg = sum(r["total_s"] for r in g) / len(g)
        if cat == "ALL":
            print("-" * len(head))
        print(f"{cat:<18}{len(g):>4}  {pct(c['correct'], len(g)):>8}{pct(c['partial'], len(g)):>8}"
              f"{pct(c['wrong'], len(g)):>7}{pct(c['abstained'], len(g)):>8}   "
              f"{pct(sum(ev), len(ev)):>9}{pct(sum(gr), len(gr)):>9}{avg:>7.1f}")
    print("\nevidence = a cited page holds the answer · grounded = every cited page was given to the model")

    with open(RESULTS_CSV, "w", newline="") as f:
        cols = ["id", "category", "question", "reference", "answer", "verdict", "reason",
                "cited_pages", "cites_evidence", "grounded", "page_images", "total_s"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["id"]))
    print(f"[saved] {RESULTS_CSV}")


# ---------- human spot-check ----------

def review(n: int, seed: int = 7):
    """You grade a random sample; then agreement with the judge is reported."""
    results = [r for r in load_results().values() if r.get("verdict")]
    done = load_results(REVIEW)
    todo = [r for r in results if r["id"] not in done]
    random.Random(seed).shuffle(todo)
    keys = {"c": "correct", "p": "partial", "w": "wrong", "a": "abstained"}

    for r in todo[: max(0, n - len(done))]:
        print("\n" + "=" * 80)
        print(f"{r['id']}  [{r['category']}]\nQ: {r['question']}\nREFERENCE: {r['reference']}\n")
        print(f"ANSWER:\n{r['answer']}\n")
        choice = ""
        while choice not in keys:
            choice = input("Your verdict? [c]orrect [p]artial [w]rong [a]bstained  (q = quit): ").strip().lower()
            if choice == "q":
                break
        if choice == "q":
            break
        save_result({"id": r["id"], "human": keys[choice], "judge": r["verdict"]}, REVIEW)

    reviews = list(load_results(REVIEW).values())
    if not reviews:
        return
    agree = sum(x["human"] == x["judge"] for x in reviews)
    # Cohen's kappa: agreement corrected for chance
    n_r = len(reviews)
    pe = sum((sum(x["human"] == v for x in reviews) / n_r) * (sum(x["judge"] == v for x in reviews) / n_r)
             for v in VERDICTS)
    po = agree / n_r
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    print(f"\n[review] you and the judge agree on {agree}/{n_r} ({100 * po:.0f}%), Cohen's kappa {kappa:.2f}")
    for x in reviews:
        if x["human"] != x["judge"]:
            print(f"  disagree {x['id']}: you={x['human']} judge={x['judge']}")


# ---------- main ----------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", help="only these question ids")
    ap.add_argument("--redo", nargs="+", default=[], help="answer these ids again")
    ap.add_argument("--rejudge", action="store_true", help="judge existing answers again")
    ap.add_argument("--summary", action="store_true", help="only print the summary")
    ap.add_argument("--review", type=int, metavar="N", help="grade N random answers yourself")
    ap.add_argument("--mode", default="rerank",
                    help="retrieval mode, e.g. graph+rerank (results go to answers_<mode>.jsonl)")
    ap.add_argument("--answer-model", default=ANSWER_MODEL)
    ap.add_argument("--judge-model", default=JUDGE_MODEL)
    args = ap.parse_args()

    if args.mode != "rerank":
        MODE = args.mode
        ANSWERS = HERE / f"answers_{args.mode.replace('+', '_').replace('-', '_')}.jsonl"
    if args.review:
        review(args.review)
        raise SystemExit
    if args.summary:
        summarize(list(load_results().values()))
        raise SystemExit

    # one answer model, no fallback: every answer comes from the same model
    ask.MODEL = args.answer_model
    ask.FALLBACK_MODEL = ""
    assert args.answer_model != args.judge_model, "use a different model for judging"

    questions = load_questions()
    if args.only:
        questions = [q for q in questions if q["id"] in args.only]
    results = load_results()
    print(f"[eval] {len(questions)} questions · answers: {args.answer_model} · judge: {args.judge_model}")

    client = get_client()
    try:
        for i, q in enumerate(questions, 1):
            row = results.get(q["id"])
            need_answer = row is None or q["id"] in args.redo or not row.get("answer")
            need_judge = need_answer or args.rejudge or not row.get("verdict")
            if not (need_answer or need_judge):
                continue
            try:
                if need_answer:
                    row = answer_question(q, client)
                    time.sleep(PAUSE_S)
                row = {**row, **judge(row, args.judge_model)}
            except Exception as e:
                print(f"  {q['id']:<5} ❌ {type(e).__name__}: {str(e)[:120]}  (run again to retry)")
                if row and row.get("answer"):
                    save_result({**row, "verdict": None})      # keep the answer, judge later
                continue
            save_result(row)
            results[q["id"]] = row
            ev = {True: "evidence ✓", False: "evidence ✗", None: ""}[row["cites_evidence"]]
            print(f"  [{i}/{len(questions)}] {q['id']:<5} {str(row['verdict']):<9} {ev:<11} "
                  f"{row['total_s']:>5.1f}s  {row['reason'][:70]}", flush=True)
    finally:
        client.close()

    summarize(list(load_results().values()))