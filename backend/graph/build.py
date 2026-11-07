"""
Step 5.3a: merge extracted entities into one graph per document.

Merging rules (safest first; a wrong merge is worse than a missed one):
  1. Same name after cleanup (case, punctuation, leading "the", plural "s") -> merge automatically
  2. Similar names (embedding of name + description, or one name inside the other) -> candidate only
  3. Gemini decides each candidate: same thing / not same (batched, cached)
  4. Type: Metric and Date only merge with the same type; the other types may merge with each
     other (Gemini labels "big models" Concept and "Transformer (big)" Product), Gemini decides
  5. Never across documents: one graph per document
  Guard: two groups are never joined if Gemini said "not same" for any pair across them,
  so "Transformer" ~ base and "Transformer" ~ big can never chain base and big together.

Usage (run from backend/, after graph.extract):
    python -m graph.build data/processed/<doc_id>            # merge + save graph
    python -m graph.build data/processed/<doc_id> --dry-run  # show candidates, no Gemini calls
    python -m graph.build data/processed/<doc_id> --redo     # forget cached Gemini decisions

Writes data/processed/<doc_id>/graph/
    graph.json            entities (nodes) + relationships (edges)
    merge_decisions.json  Gemini's same / not-same answers (cache, re-runs are free)
    merges.txt            every merge, readable, for the hand check
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

from graph.extract import call_gemini, is_bibliography, load_done, out_file
from ingest.index import get_dense

SIM_THRESHOLD = 0.88       # cosine similarity of "name: description" to become a candidate
MAX_NEIGHBOURS = 3         # candidates per entity (closest only)
BATCH = 80                 # candidate pairs per Gemini call (fewer calls on the free tier)
STRICT_TYPES = {"Metric", "Date"}

JUDGE_PROMPT = """You check whether two entity mentions from the SAME document refer to exactly the
same real thing.

"same" = the same thing under a different name, e.g. "NVIDIA P100 GPU" and "P100",
         "Apple" and "Apple Inc.", "big models" and "Transformer (big)".
"not same" = different things, even if related: a variant or a part ("Transformer (base model)" vs
         "Transformer (big)"), general vs specific ("Transformer" vs "Transformer (big)"),
         a thing vs its measurement ("iPhone" vs "iPhone net sales"), different years or regions.
If unsure, answer not same.

Pairs:
{pairs}

Return JSON only: {{"answers": [{{"pair": 0, "same": true}}, ...]}} with one answer per pair."""


# ---------- rule 1: cleanup ----------

def clean_name(name: str) -> str:
    s = name.lower().replace("’", "'")
    s = re.sub(r"^the\s+", "", s)
    s = re.sub(r"[^\w\s.%$-]", " ", s)          # drop brackets, quotes, commas
    s = re.sub(r"\s+", " ", s).strip(" .-")
    words = s.split()
    if words and len(words[-1]) > 3 and words[-1].endswith("s") and not words[-1].endswith("ss"):
        words[-1] = words[-1][:-1]                # "big models" = "big model"
    return " ".join(words)


def types_compatible(a: str, b: str) -> bool:
    return a == b or (a not in STRICT_TYPES and b not in STRICT_TYPES)


# ---------- collect mentions and apply rule 1 ----------

def exact_groups(extractions: list[dict]) -> tuple[list[dict], dict]:
    """One group per (type family, cleaned name). Returns groups and (chunk_id, name) -> group index."""
    groups, index, where = [], {}, {}
    for ex in extractions:
        for e in ex["entities"]:
            fam = e["type"] if e["type"] in STRICT_TYPES else "general"
            key = (fam, clean_name(e["name"]))
            if not key[1]:
                continue
            if key not in index:
                index[key] = len(groups)
                groups.append({"names": Counter(), "types": Counter(), "descriptions": [],
                               "chunk_ids": set(), "pages": set()})
            g = groups[index[key]]
            g["names"][e["name"]] += 1
            g["types"][e["type"]] += 1
            if e["description"] and e["description"] not in g["descriptions"]:
                g["descriptions"].append(e["description"])
            g["chunk_ids"].add(ex["chunk_id"])
            g["pages"].update(ex["pages"])
            where[(ex["chunk_id"], e["name"].lower())] = index[key]
    for g in groups:
        g["name"] = g["names"].most_common(1)[0][0]
        g["type"] = g["types"].most_common(1)[0][0]
    return groups, where


# ---------- rule 2: candidates ----------

def find_candidates(groups: list[dict]) -> list[tuple[int, int, float]]:
    texts = [f"{g['name']}: {g['descriptions'][0] if g['descriptions'] else ''}" for g in groups]
    vecs = np.array(list(get_dense().embed(texts)), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    sims = vecs @ vecs.T
    cleaned = [clean_name(g["name"]) for g in groups]

    def inside(i, j):           # one cleaned name is a whole-word part of the other ("p100" in "nvidia p100 gpu")
        x, y = cleaned[i], cleaned[j]
        return len(x) >= 3 and re.search(rf"\b{re.escape(x)}\b", y) is not None

    pairs = {}
    for i in range(len(groups)):
        # a) the closest neighbours by meaning, above the threshold
        taken = 0
        for j in map(int, np.argsort(-sims[i])):
            if sims[i, j] < SIM_THRESHOLD or taken >= MAX_NEIGHBOURS:
                break
            if j != i and types_compatible(groups[i]["type"], groups[j]["type"]):
                pairs[tuple(sorted((i, j)))] = float(sims[i, j])
                taken += 1
        # b) names inside each other, even if far apart in meaning
        for j in range(len(groups)):
            if j != i and types_compatible(groups[i]["type"], groups[j]["type"]) and inside(i, j):
                pairs.setdefault(tuple(sorted((i, j))), float(sims[i, j]))
    return sorted(((a, b, s) for (a, b), s in pairs.items()), key=lambda x: -x[2])


# ---------- rule 3: Gemini decides ----------

def pair_key(a: dict, b: dict) -> str:
    return " || ".join(sorted([a["name"], b["name"]]))


def judge(groups, candidates, cache: dict, cache_file: Path | None = None) -> dict:
    todo = [(a, b) for a, b, _ in candidates if pair_key(groups[a], groups[b]) not in cache]
    for start in range(0, len(todo), BATCH):
        batch = todo[start:start + BATCH]
        lines = []
        for k, (a, b) in enumerate(batch):
            da = groups[a]["descriptions"][0] if groups[a]["descriptions"] else ""
            db = groups[b]["descriptions"][0] if groups[b]["descriptions"] else ""
            lines.append(f'{k}. A: "{groups[a]["name"]}" ({groups[a]["type"]}): {da}\n'
                         f'   B: "{groups[b]["name"]}" ({groups[b]["type"]}): {db}')
        raw = call_gemini(JUDGE_PROMPT.format(pairs="\n".join(lines)))
        answers = {int(x["pair"]): bool(x.get("same")) for x in raw.get("answers", [])
                   if str(x.get("pair", "")).isdigit()}
        for k, (a, b) in enumerate(batch):
            cache[pair_key(groups[a], groups[b])] = answers.get(k, False)   # no answer = not same
        if cache_file:                                       # save after every batch: a crash loses nothing
            cache_file.write_text(json.dumps(cache, indent=1, ensure_ascii=False))
        print(f"  [judge] {min(start + BATCH, len(todo))}/{len(todo)} pairs", flush=True)
    return cache


# ---------- apply merges (union-find with the "not same" guard) ----------

def merge_groups(groups, candidates, decisions):
    parent = list(range(len(groups)))
    members = {i: {i} for i in range(len(groups))}
    said_no = {(a, b) for a, b, _ in candidates if not decisions.get(pair_key(groups[a], groups[b]))}
    said_no |= {(b, a) for a, b in said_no}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    merged, blocked = [], []
    for a, b, sim in candidates:                       # most similar first
        if not decisions.get(pair_key(groups[a], groups[b])):
            continue
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if any((x, y) in said_no for x in members[ra] for y in members[rb]):
            blocked.append((a, b))                     # would chain two "not same" things
            continue
        parent[rb] = ra
        members[ra] |= members.pop(rb)
        merged.append((a, b, sim))
    return find, merged, blocked


# ---------- build the graph ----------

def build(doc_dir: Path, dry_run=False, redo=False):
    extractions = [x for x in load_done(out_file(doc_dir)).values() if not is_bibliography(x)]
    if not extractions:
        raise SystemExit(f"no extractions for {doc_dir.name}: run graph.extract first")
    gdir = doc_dir / "graph"
    cache_file = gdir / "merge_decisions.json"
    cache = {} if redo or not cache_file.exists() else json.loads(cache_file.read_text())

    mentions = sum(len(x["entities"]) for x in extractions)
    groups, where = exact_groups(extractions)
    print(f"[build] {doc_dir.name}: {mentions} entity mentions -> {len(groups)} after rule 1 (same name)")

    candidates = find_candidates(groups)
    print(f"[build] {len(candidates)} candidate pairs (rule 2)")
    if dry_run:
        for a, b, s in candidates[:60]:
            print(f"  {s:.2f}  {groups[a]['name']!r:<40} ~ {groups[b]['name']!r}")
        return

    cache = judge(groups, candidates, cache, cache_file)
    cache_file.write_text(json.dumps(cache, indent=1, ensure_ascii=False))
    find, merged, blocked = merge_groups(groups, candidates, cache)
    yes = sum(cache.get(pair_key(groups[a], groups[b]), False) for a, b, _ in candidates)
    print(f"[build] Gemini said same for {yes} of {len(candidates)} pairs; "
          f"{len(merged)} merges applied, {len(blocked)} blocked by the not-same guard")

    # entities = one per root group
    roots = defaultdict(list)
    for i in range(len(groups)):
        roots[find(i)].append(i)
    entities, gid_to_eid = [], {}
    for k, (root, idx) in enumerate(sorted(roots.items())):
        names, types = Counter(), Counter()
        descs, chunks, pages = [], set(), set()
        for i in idx:
            names.update(groups[i]["names"])
            types.update(groups[i]["types"])
            descs += [d for d in groups[i]["descriptions"] if d not in descs]
            chunks |= groups[i]["chunk_ids"]
            pages |= groups[i]["pages"]
            gid_to_eid[i] = f"e{k}"
        name = names.most_common(1)[0][0]
        entities.append({"id": f"e{k}", "name": name, "type": types.most_common(1)[0][0],
                         "aliases": sorted(n for n in names if n != name),
                         "descriptions": descs, "chunk_ids": sorted(chunks),
                         "pages": sorted(pages)})

    # edges: one per (source, target); keep every description and evidence quote
    edges = {}
    for ex in extractions:
        for r in ex["relationships"]:
            ga = where.get((ex["chunk_id"], r["source"].lower()))
            gb = where.get((ex["chunk_id"], r["target"].lower()))
            if ga is None or gb is None:
                continue
            s, t = gid_to_eid[ga], gid_to_eid[gb]
            if s == t:
                continue                               # became a self-link after merging
            e = edges.setdefault((s, t), {"source": s, "target": t, "kind": "stated",
                                          "descriptions": [], "evidence": [], "strength": 0})
            if r["description"] not in e["descriptions"]:
                e["descriptions"].append(r["description"])
            e["evidence"].append({"quote": r["evidence"], "chunk_id": ex["chunk_id"],
                                  "pages": ex["pages"]})
            e["strength"] = max(e["strength"], r["strength"])

    chunks, structural = structural_edges(doc_dir, entities)
    out = {"doc_id": doc_dir.name, "entities": entities, "edges": list(edges.values()),
           "chunks": chunks, "structural_edges": structural}
    (gdir / "graph.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))

    with open(gdir / "merges.txt", "w") as f:
        f.write("MERGED (Gemini said same)\n")
        for a, b, s in merged:
            f.write(f"  {s:.2f}  {groups[a]['name']}  <=>  {groups[b]['name']}\n")
        f.write("\nNOT MERGED (Gemini said not same)\n")
        for a, b, s in candidates:
            if not cache.get(pair_key(groups[a], groups[b])):
                f.write(f"  {s:.2f}  {groups[a]['name']}  x  {groups[b]['name']}\n")
        f.write("\nBLOCKED (would chain two not-same things)\n")
        for a, b in blocked:
            f.write(f"  {groups[a]['name']}  <=>  {groups[b]['name']}\n")

    g = to_networkx(out)
    multi = sum(len(e["chunk_ids"]) > 1 for e in entities)
    only_stated = nx.Graph([(e["source"], e["target"]) for e in out["edges"]])
    only_stated.add_nodes_from(e["id"] for e in entities)
    kinds = Counter(e["rel"] for e in structural)
    print(f"[graph] {len(entities)} entities ({multi} appear in 2+ chunks), {len(edges)} stated edges")
    print(f"[graph] structural: {len(chunks)} chunks, " + ", ".join(f"{v} {k}" for k, v in kinds.items()))
    print(f"[graph] connected pieces: {nx.number_connected_components(only_stated)} with stated edges only "
          f"-> {nx.number_connected_components(g.to_undirected())} with structural edges")
    print(f"[saved] {gdir / 'graph.json'}  ·  merges for the hand check: {gdir / 'merges.txt'}")


def structural_edges(doc_dir: Path, entities: list[dict]):
    """Edges from the layout (no LLM): entity -> chunk it came from, chunk -> its section,
    chunk -> its page, chunk -> next chunk in the same section (reading order)."""
    all_chunks = json.loads((doc_dir / "chunks.json").read_text())
    used = {cid for e in entities for cid in e["chunk_ids"]}
    chunks, edges = [], []
    for c in all_chunks:
        if is_bibliography(c):
            continue
        cid = c["chunk_id"]
        chunks.append({"id": cid, "pages": c["pages"], "heading": c.get("heading"), "type": c["type"]})
        if c.get("heading"):
            edges.append({"source": cid, "target": "s:" + c["heading"], "kind": "structural", "rel": "in_section"})
        for p in c["pages"]:
            edges.append({"source": cid, "target": f"p:{p}", "kind": "structural", "rel": "on_page"})
    for a, b in zip(chunks, chunks[1:]):
        if a["heading"] and a["heading"] == b["heading"]:
            edges.append({"source": a["id"], "target": b["id"], "kind": "structural", "rel": "next"})
    for e in entities:
        for cid in e["chunk_ids"]:
            if cid in used:
                edges.append({"source": e["id"], "target": cid, "kind": "structural", "rel": "mentioned_in"})
    return chunks, edges


def to_networkx(data: dict) -> nx.DiGraph:
    """Nodes: entities (e12), chunks (<doc>_c5), sections (s:<heading>), pages (p:7)."""
    g = nx.DiGraph()
    for e in data["entities"]:
        g.add_node(e["id"], node="entity", **e)
    for c in data.get("chunks", []):
        g.add_node(c["id"], node="chunk", **c)
    for e in data["edges"]:
        g.add_edge(e["source"], e["target"], **e)
    for e in data.get("structural_edges", []):
        for n in (e["source"], e["target"]):
            if n not in g:
                g.add_node(n, node="section" if n.startswith("s:") else "page")
        g.add_edge(e["source"], e["target"], **e)
    return g


def load_graph(doc_dir: Path) -> nx.DiGraph:
    return to_networkx(json.loads((doc_dir / "graph" / "graph.json").read_text()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("doc_dir", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="only list candidate pairs")
    ap.add_argument("--redo", action="store_true", help="ignore cached Gemini decisions")
    args = ap.parse_args()
    build(args.doc_dir, args.dry_run, args.redo)