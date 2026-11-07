# DocAI: project notes

Ask a PDF anything and see the exact passage behind every answer. A multimodal RAG app for
real-world PDFs (tables, charts, scanned and handwritten pages) with three answer modes:
**normal** (hybrid search + reranker + page images), **Graph** (GraphRAG, for facts connected
across pages) and **Agent** (agentic RAG: plans, searches several times, calculates, checks).

---

## 1. Headline results

| What | Result |
|---|---|
| Single-fact answers (55 questions, 6 documents) | **96% correct, 0% wrong, 100% of citations grounded** |
| Retrieval, hybrid + reranker (55 questions) | **96% hit@5**, MRR 0.86 |
| Faded 1960 scan | 100% correct, incl. an answer that exists only in the page image |
| Multi-hop "bridge" answers (12): normal → Graph → **Agent** | 50% → 67% → **100%** |
| Multi-hop, all 26: normal → decomposition → **Agent** | 69% → 88% → **96%** |
| Hard chained questions (12): normal → Graph → decomposition → **Agent** | 50% → 67% → 75% → **83%** |
| Single-fact answers in Agent mode (safety check) | 95% correct, **0% wrong** |
| Wrong answers, every mode, every question set | **0%** |

---

## 2. Tech stack

| Part | Tool | Why |
|---|---|---|
| PDF parsing | **Docling** (layout, reading order, TableFormer for tables) | Structure + bounding boxes; tables as Markdown |
| OCR | **RapidOCR** (inside Docling) | Free, local; only for scanned pages |
| Page classification, page images | **PyMuPDF** | Labels pages digital / scanned / blank; saves PNGs |
| Text embeddings | **bge-small-en-v1.5** via **fastembed** (384 dimensions) | Small, CPU-friendly, free |
| Keyword search | **BM25** (fastembed `Qdrant/bm25`, sparse vectors) | Exact words and numbers |
| Reranker | **bge-reranker-base** (cross-encoder, fastembed) | Re-reads each (question, chunk) pair |
| Visual retrieval | **ColSmol-256M** (ColPali family, `colpali-engine`) | Searches page *images* (charts, scans) |
| Vector database | **Qdrant** (local mode) | Dense + sparse in one point, multi-vector MaxSim, RRF fusion |
| LLM | **Gemini Flash** (`google-genai`): 3.6-flash answers, 3.8-flash judges / fallback | Reads text + page images; streaming; JSON output |
| Graph | **NetworkX** | One graph per document |
| Agent | **Plain Python loop** (no framework), Gemini JSON, ReAct pattern | Every line explainable and loggable |
| Backend | **FastAPI**, Server-Sent Events, background worker, threads | Streamed answers and live agent steps |
| Frontend | **Next.js 16**, React, **react-pdf** (PDF.js), react-markdown, Tailwind CSS v4 | PDF viewer with highlights, streamed chat |

Runs on a MacBook Air (models on CPU / Apple GPU); only Gemini is a cloud API.

---

## 3. How it works, step by step

```
PDF → Parse → Chunk → Index (text + page images [+ graph]) → Retrieve (normal / Graph / Agent) → Answer with citations → UI
```

### Step 1: Parse (`backend/ingest/parse.py`)
1. Hash the file → `doc_id` (the same file is never processed twice).
2. **Classify every page** with PyMuPDF:
   - `digital`: real text layer → no OCR
   - `scanned`: one image covers ≥ 80% of the page → OCR, even if a (bad) hidden text layer exists
   - `blank`: < 50 characters and no large image → skipped
   - ≥ 50% scanned pages → forced full-page OCR for the whole document
3. Save every page as a PNG (144 DPI) for page-image retrieval and the viewer.
4. **Docling** reads layout, reading order (also two-column papers), headings, tables (→ Markdown), figures and captions, each with a **bounding box** (used to highlight the evidence).

### Step 2: Chunk (`backend/ingest/chunk.py`), rule-based, no model
1. A new heading starts a new chunk; the heading is added to every chunk under it.
2. Paragraphs are joined into chunks of up to **300 words**.
3. A paragraph is never cut, unless it alone is longer than the limit (then split at sentence ends).
4. **Tables are their own chunk**; big tables are split by rows, repeating the header (only if ≤ 40 words).
5. Figures with captions get their own chunk; figures without captions are left to visual search.
6. **Table intro sentences stay with their table** ("The following table shows net sales…").
7. **Tiny chunks** (< 30 words) are merged into a neighbour (never into a table).
8. Every chunk keeps its pages and bounding boxes (for clickable citations).

Why 300 words: largest size with no truncated text for the embedder's 512-token limit (tested 150 / 300 / 500).

### Step 3: Index
- **Text** (`index.py`): each chunk → one Qdrant point with a dense vector (bge-small) and a sparse vector (BM25 with IDF: rare words count more). The full chunk is the payload.
- **Page images** (`visual_index.py`): only scanned pages and pages with figures. ColSmol cuts each page into ~1,000 patches → one vector per patch; Qdrant scores them with **MaxSim** (each question word finds its best-matching patch; the page score is the sum).

### Step 4: Retrieve (`backend/retrieval/search.py`)
| Mode | What it does |
|---|---|
| `dense` | Meaning only |
| `bm25` | Exact words only |
| `hybrid` | Dense top 20 + BM25 top 20, fused with **Reciprocal Rank Fusion** (RRF) inside Qdrant |
| **`rerank`** (default) | Hybrid top 20 → cross-encoder reranker → top 5 |
| `visual` | Page images (ColSmol, MaxSim) |
| **`graph+rerank`** | Rerank's best 3 + 2 chunks found by walking the graph (Step 8) |
| **`agent`** | Several tool calls chosen by Gemini (Step 9) |

### Step 5: Answer (`backend/retrieval/ask.py`)
- Gemini gets the **top 5 chunks + up to 3 page images** (Agent: everything it collected, ~7 chunks).
- **Choosing page images (normal modes, documents with image pages):** slot 1 = best visual page; then pages of *confident* text hits (reranker score > 0) that have an image; then more visual pages; max 3. Skipped for documents without image pages. Example T11 (faded scan): all text hits scored below 0, so the visual pages 3, 4, 9 were used and Gemini read "17 days" on page 4.
- **Prompt rules:** answer only from the sources; cite every fact as `[p. N]`; read tables by row and column; say "I couldn't find this in the document" instead of guessing.
- **Reliability:** streaming; 90 s timeout; retry on 429 / 500 / 503 / 504 / timeout; fallback model; no retry once text has started.

### Step 6: The app
- **Backend (FastAPI):** models loaded once at startup; one shared Qdrant connection with a lock. Upload → background worker (parse → chunk → index → visual index) with live status; duplicates detected by file hash. `POST /chat` streams Server-Sent Events: `step` (Agent mode) → `sources` → `token`… → `done` (timing). The agent runs in a background thread that pushes each step into a queue, which the stream sends to the browser as it happens.
- **Frontend (Next.js):** document list with drag-and-drop upload and progress; PDF viewer (lazy page rendering, zoom); streamed chat with Markdown; **clickable citations** that scroll to the page and **highlight the source passage**; mode menu (Rerank / Hybrid / **Graph** / **Agent**); page-images toggle; dark mode; landing page.
- **Graph mode:** sources found by the graph get a **via graph** tag and the connection that brought them ("P100 GPU → big models: trained on"); disabled for documents without a graph.
- **Agent mode:** each step appears live ("1. Searched 'net sales by product 2026' · 5 passages", with the agent's one-line thought), then the answer with citations; the steps stay visible.

### Step 7: Evaluation (`backend/eval/`)
- **Test set: 103 questions over 6 documents** (Apple 10-K, Attention paper, BERT paper, IMF report with charts, 1960 Gazette faded scan, EPA letter clean scan):
  - 55 single-fact in 9 categories (table, formula table, number, exact term, synonym, prose, chart, clean scan, faded scan)
  - 26 multi-hop: **bridge** (12: you need fact 1 to know what fact 2 is) and **compare** (14: both facts named in the question)
  - 10 agent questions: calculation, three parts, comparison across sections, **not in the document**
  - 12 hard chained questions: parts in different places, the next part depends on the first, usually a calculation
- **Retrieval eval:** each fact has a **keyword** that must appear in the retrieved chunks (all facts in the top 5 for multi-hop). Metrics: **hit@5** and **MRR**. Keywords are only for scoring; the system never sees them. `--check` validates keywords (not found, too common, or one chunk holding every fact of a "multi-hop" question).
- **Answer eval:** same pipeline as the app. One model answers (gemini-3.6-flash, no fallback); a **different model judges** (gemini-3.8-flash) against a hand-written answer key: correct / partial / wrong / abstained. For "not in the document" questions, saying so counts as correct and any invented figure as wrong. Two checks need no model:
  - **evidence:** the answer cites a page that holds the answer (a page for *every* fact)
  - **grounded:** every cited page was actually given to the model (no invented citations)
- **Human check:** 15 random verdicts graded by hand without seeing the judge: 15/15 agreement.
- **Control experiments for the agent** (same chunk budget, ~7): rerank with 7 chunks, and **query decomposition** (Step 10).

### Step 8: GraphRAG (`backend/graph/`)

**Idea:** normal search finds chunks that *sound like* the question. Bridge questions need a second fact that does not ("Which GPUs were used, and what TFLOPS was assumed for that GPU?": the TFLOPS footnote never mentions training). A graph is a **map of the document**; from fact 1 it leads to fact 2.

**8a. Extraction (`extract.py`):** Gemini reads each chunk (temperature 0, JSON) and returns:
- **Entities:** important named things, name as written, one of 8 types (Organization, Person, Product, Place, Date, Metric, Concept, Event), a one-sentence description with numbers.
- **Relationships:** only links the text states, with a description, a **strength 1–10** and an **exact evidence quote** (5–25 words).
- Not entities: generic words ("the Company") and labels of document parts ("Table 3"). Skipped: chunks under 8 words and the bibliography.
- **Checks in code:** the quote must really be in the chunk (ignoring case, spacing, table pipes, "$ 254" vs "$254"), ≥ 8 characters; both ends must be entities of the chunk; no self-links; unknown type → Concept; strength clamped to 1–10 (missing → 5). **98–100% of relationships kept** on every document.

**8b. Merging (`build.py`): the same thing in different chunks → one node.** Safest first (a wrong merge is worse than a missed one):
1. Same name after cleanup (case, punctuation, "the", plural "s") → merge automatically.
2. Similar names (embedding similarity of name + description ≥ 0.88, or one name inside the other: "P100" in "NVIDIA P100 GPU") → only a "maybe".
3. Gemini decides each "maybe": same / not same; "if unsure, not same". 80 pairs per call; decisions cached per batch.
4. Metric and Date merge only with the same type; other types may merge ("big models" Concept = "Transformer (big)" Product).
5. Never across documents.
- **The "not same" guard:** two groups are never joined if Gemini said "not same" for any pair across them (stops Transformer = big + Transformer = base → base = big).

**8c. Three kinds of connection:** merging (same thing in different chunks), **stated edges** (a link in one sentence, quote checked), **structural edges** (entity → its chunks, chunk → section, chunk → page, chunk → next chunk; from the layout). Structural edges join each document into one map (IMF: 251 pieces → 1).

**8d. Entity index (`index_entities.py`):** every entity embedded into Qdrant so a question can find where to start.

**8e. Graph search, mode `graph+rerank` (`search.py`):**
1. Normal search → keep the best **3** chunks (usually fact 1).
2. Starting points: the 5 entities closest to the question + every entity inside those 3 chunks. **Date entities are hubs** ("2026" touches almost every chunk): never starting points, never walked through.
3. **Walk 1 step**, score chunks (the *walk score*): a chunk mentioning a starting entity → its score; via a stated edge → × 0.8 × strength/10; next chunk in the same section → × 0.5; highest score wins; an entity gives at most 6 chunks, **chosen by relevance** to the question; page edges not walked; keep the best 20.
4. Pick **2 chunks by walk score** (the reranker only breaks ties).
5. 3 + 2 = 5 chunks: the same budget as normal search.

### Step 9: Agentic RAG (`backend/agent/`)

**Idea:** normal and Graph modes search once. The agent works like a researcher: it splits the question into parts, searches, reads, notices what is missing, **writes the next search using what it just found**, calculates, and only then answers.

**9a. Tools (`tools.py`)**, 6 to start (more only if the logs show a need; too many similar tools make agents pick worse):
| Tool | Uses | For |
|---|---|---|
| `text_search(query)` | hybrid + reranker, top 5 | most questions |
| `graph_search(query)` | graph+rerank | a fact linked to something already found |
| `visual_search(query)` | page images, top 2 | charts, scans |
| `read_next(chunk)` | the chunks before and after | an explanation right under a table |
| `find_exact(term)` | every chunk with the exact word / number | exact terms, codes, numbers |
| `calculate(expression)` | safe arithmetic (parsed, never `eval`) | differences, growth, totals |

Each result shows short chunk ids (`[c111] p. 29`) so the agent can pass them on; tool errors come back as text, never a crash. The calculator accepts only numbers, operators, brackets and round / abs / min / max (`__import__` is refused).

**9b. The loop (`agent.py`):** the agent keeps a notebook (question, **checklist** of parts, everything found). Each round Gemini (JSON, temperature 0) returns **one thought, the updated checklist (each part with the chunk id that answers it), and one tool call**; the code runs the tool and adds the result.

**9c. Stopping rules, enforced in code:**
1. `answer` is accepted only when every part has a source **a tool really returned**; an invented chunk id is refused. A part may be null ("not in the document").
2. At most **5 rounds**, then it answers with what it has and says what is missing.
3. The same tool + argument twice is refused, forcing a new idea.
4. A tool error comes back as text; a planner failure answers with what was found.

**9d. Answer:** the **same answer step as normal mode** (prompt, `[p. N]` citations, highlighting) on every collected chunk (max 10) and page image (max 3).

**Why the "thought":** one line of reasoning before acting (**ReAct**: Reason + Act) makes tool choice more sensible, shows *why* a run failed, and is shown live in the app.

**Why an agent and not a fixed recipe:** each question needs different steps (search then read next; search again; find an exact name; calculate; look at a chart; or one search), and the second step depends on what the first *found*: the second M01 search must say "iPhone", known only after reading the table. Code cannot read a table and conclude "iPhone grew most"; an LLM can. Our code builds and runs the tools and enforces the rules; the LLM only chooses the next tool and the words. A fixed recipe (a *workflow*) is cheaper and predictable; an agent adapts but costs more calls.

### Step 10: Query decomposition, the control (`backend/agent/decompose.py`)
One Gemini call splits the question into 1–3 short queries **all written at the start**; each runs through normal search; results merged in turn (best of each query first) up to 7 chunks; normal answer step. It has the agent's extra queries and chunk budget but no loop: it cannot write "why did iPhone grow" because it never saw that the answer is iPhone. Comparing it with the agent isolates the value of "the next search uses what was found".

---

## 4. Results

### Retrieval, single-fact (55 questions), hit@5 / MRR
| Category (n) | Dense | BM25 | Hybrid | Rerank | Visual |
|---|---|---|---|---|---|
| chart (7) | 86% / 0.59 | 100% / 0.81 | 100% / 0.79 | 100% / 0.89 | 86% / 0.88 |
| exact-term (6) | 83% / 0.74 | 83% / 0.77 | 83% / 0.74 | 83% / 0.85 | |
| formula-table (3) | 67% / 0.69 | 100% / 1.00 | 100% / 0.78 | 100% / 0.75 | |
| number (11) | 91% / 0.74 | 100% / 0.69 | 100% / 0.79 | 100% / 0.95 | |
| prose (4) | 100% / 0.81 | 100% / 1.00 | 100% / 1.00 | 100% / 1.00 | |
| scan-clean (4) | 100% / 0.71 | 100% / 0.75 | 100% / 0.71 | 100% / 0.88 | 100% / 0.88 |
| scan-faded (8) | 88% / 0.82 | 88% / 0.89 | 100% / 0.94 | 88% / 0.80 | 100% / 0.86 |
| synonym (7) | 100% / 0.83 | 57% / 0.53 | 86% / 0.78 | 100% / 0.86 | |
| table (5) | 80% / 0.39 | 40% / 0.24 | 80% / 0.30 | 100% / 0.67 | |
| **All** | **89% / 0.71** | **85% / 0.73** | **95% / 0.77** | **96% / 0.86** | **94% / 0.87** (18) |

### Answers, single-fact (55 questions)
| | Correct | Partial | Wrong | Abstained | Evidence | Grounded | Time |
|---|---|---|---|---|---|---|---|
| rerank | **96%** | 2% (a chart value) | **0%** | 2% (Q12) | 100% | 100% | 7.0 s |
| graph+rerank | **96%** | 4% (charts) | **0%** | 0% (Q12 fixed) | 100% | 100% | ~13 s |
| agent | **95%** | 4% (charts T8, T9) | **0%** | 2% (T11, faded scan) | 98% | 100% | ~15 s |

### GraphRAG retrieval (81 questions, hit@5, all facts in the top 5)
| | n | rerank | graph-walk | **graph+rerank** |
|---|---|---|---|---|
| Single-fact | 55 | 96% | lower | **98%** |
| Compare | 14 | 79% | 57% | **86%** |
| Bridge | 12 | 42% | 33% | **58%** |
| All | 81 | 85% | – | **90%** |

Gained 6 (Q12, M02, M06, M20, M24, M26), lost 1 (M03). Graph sizes: Apple 10-K 889 entities / 906 stated edges; IMF 640 / 441; Gazette 599 / 370; BERT 176 / 164; Attention 138 / 118.

### All modes on the 26 multi-hop questions (answers)
| Mode | Chunks | Bridge (12) | Compare (14) | All (26) | Wrong | Time |
|---|---|---|---|---|---|---|
| rerank | 5 | 50% | 86% | 69% | 0% | ~10 s |
| rerank | 7 | 67% | 93% | 81% | 0% | 9 s |
| graph+rerank | 5 | 67% | 93% | 81% | 0% | ~14 s |
| query decomposition | ~7 | 75% | **100%** | 88% | 0% | 13.5 s |
| **agent** | ~7 | **100%** | 93% | **96%** | 0% | 16.4 s |

### Agent questions (10: calculation, three parts, cross-section compare, not in the document)
| | rerank | graph+rerank | decompose | agent |
|---|---|---|---|---|
| Correct | 9/10 | 8/10 | **10/10** | **10/10** |
| Wrong | 0 | 0 | 0 | 0 |

Mostly too easy to separate the modes (see challenge 21); only A05 (numbers in two different sections) needed two searches.

### Hard chained questions (12)
| | rerank | graph+rerank | decompose | **agent** |
|---|---|---|---|---|
| Correct | 6 (50%) | 8 (67%) | 9 (75%) | **10 (83%)** |
| Partial / couldn't find | 4 / 2 | 1 / 3 | 2 / 1 | 2 / **0** |
| Wrong | 0 | 0 | 0 | 0 |
| Time | 11 s | 15 s | 19 s | 28 s |

Only the agent solved H03 (Japan → both years' operating income → 14,650 − 13,955 = 695) and H12 (chosen encoding → Table 3 row → −0.1).

**Agent safety check (55 single-fact questions): 95% correct, 0% wrong.** Q12 correct; the two chart partials are the same as in Graph mode; the one miss is **T11** ("earned leave for 17 days", faded scan): the agent searched text 3+ times but never called `visual_search`, while normal mode always sends page images on scanned documents. Agent on all 103 questions: 94% correct, 0% wrong, 3.2 rounds and 6.4 chunks per question.

**Agent overall on 48 multi-step questions: 94% correct, 0% wrong, 0 "couldn't find", 3.8 rounds and ~7 chunks per question, 19 s.**

**Tools the agent chose:** `calculate` on every calculation question; 3–4 different searches before saying "not in the document"; all 6 tools used; two questions used all 5 rounds and were still correct.

---

## 5. Findings: what wins over what, and why (with examples)

### Retrieval: dense vs BM25 vs hybrid
- **BM25 beats dense on exact words and numbers.** T1 "complexity of self-attention": the answer is a formula table; dense ranks it 15th, BM25 1st. T10 (Gazette): dense 21st, BM25 1st ("Indian Administrative Service").
- **Dense beats BM25 on synonyms.** T2 "How many employees?": the text says "full-time equivalent"; BM25 6th, dense 1st.
- **Hybrid wins overall (95%)**: the two fail on different questions; RRF misses neither.

### Reranker vs hybrid
- **The reranker wins on tables and numbers** (tables MRR 0.30 → 0.67): it reads question and chunk together (Q03 Greater China row found only by rerank).
- **Hybrid wins on noisy OCR text** (faded scans 100% vs 88%): OCR errors lower the cross-encoder's confidence.
- **But the reranker is the wrong judge for fact 2 of a bridge question:** it asks "does this chunk answer the question?", and fact 2 alone never does (M24: the TFLOPS footnote scored −3.9).

### Page images vs text
- **Page images win when OCR destroyed the answer.** T11 "earned leave for 17 days": OCR produced `ear   s       ty`; only the page image has it.
- **Only confident text hits may pick page images** (score > 0); otherwise weak hits used up the slots (T11 fixed).

### graph+rerank vs rerank
- **The graph wins when fact 2 is connected to fact 1 but does not sound like the question.** M24: the footnote is reached through the merged "P100" entity.
- **It reaches with 5 chunks what rerank needs 7 for** (both 67% on bridge).
- **It loses when its 2 slots push out a useful chunk** (M03: rank 4 → 6).

### graph+rerank vs graph-only
- **Graph-only loses single-fact questions** (Q18, Q29 not found) and collapses on the 10-K (44%): hubs ("Apple", "net sales") connect to everything. The graph works as a helper, not a replacement.

### More chunks (k = 7) vs 5
- **Wins when the answer sits just below the top 5** (M22, M24 at rank 6). Does not help when the chunk is not retrieved at all (M01, M02).

### Query decomposition vs more chunks
- **Wins when the question has independent parts** (M01, M05, M09) and on compare questions (100%).

### Agent vs query decomposition
- **The agent wins when the next search needs the first answer** (+25 points on bridge):
  - **M02** "Which segment grew fastest, and how did currency affect it?": the agent finds *Greater China*, then searches its currency effect (renminbi); decomposition never knew it was Greater China.
  - **M22**: *Japan* first, then the yen. **M24**: *P100* first, then the TFLOPS footnote.
  - **H03, H12**: three-step chains with a calculation; only the agent.
- **It rewrites a failed query:** M01 round 1 found paragraphs without numbers; round 2 listed the product names and got the table and the reason.
- **It never gave up** on the hard questions (0 "couldn't find" vs 1–3 for the others).
- **Decomposition wins when parts are independent** (H07: two numbers on different pages; compare questions 100% vs 93%) and is cheaper.

### One line per method
| Method | Wins on |
|---|---|
| BM25 | exact words, numbers, codes |
| Dense | synonyms, paraphrases |
| Hybrid + reranker | single-fact questions (default) |
| Page images | charts, faded scans |
| More chunks | answers just below the top 5 |
| GraphRAG | a connected fact that does not sound like the question, with fewer chunks |
| Query decomposition | questions with independent parts |
| Agent | questions whose next search depends on what was just found |

---

## 6. Worked example: M01 in every mode (the clearest interview story)

**"Which product grew fastest in 2026, and why?"** Two pieces, next to each other on p. 29: the net sales table (**c111**: iPhone +21%) and the reason (**c112**: "iPhone net sales increased … due to higher net sales of Pro models").

**Experiment 1: the same three queries in both search tools** (does c111 / c112 reach the top 5?)
| Query | Normal search | Graph search |
|---|---|---|
| The full question | ❌ table ❌ reason | ❌ table ✅ reason |
| "net sales by product 2026" | ✅ table ❌ reason | ✅ table ❌ reason |
| "why did iPhone net sales increase in 2026" | ❌ table ✅ reason | ❌ table ✅ reason |

- **No single query finds both, with either tool; the query matters more than the tool.**
- Normal search misses the reason because the question never says "iPhone" (that is what is asked), and six paragraphs look alike ("Mac / Americas / Greater China / iPhone net sales increased … due to …").
- The full question misses the table because "grew" and "why" match paragraphs, not a table of numbers.
- Once "iPhone" is in the query, plain search finds the reason at once: the hard part is **knowing what to ask**, which needs part 1 first.

**Experiment 2: why did the graph find the reason?** (`--explain`) It started from dates ("2026") and found c112 only because it mentions "2026": **a right answer for the wrong reason**, which uncovered two bugs (challenge 17). After the fixes the graph reached both pieces but could not rank them (16 starting entities, all tied at 0.7; challenge 18).

**The agent:** round 1 `text_search("fastest growing product 2026 growth")` → paragraphs, no table; round 2 `text_search("Net Sales by Product Category iPhone Mac iPad Services")` → table **and** reason; round 3 answer: "iPhone … 21% [p. 29] … due to higher net sales of Pro models [p. 29]". **Solved for the first time**, by a route nobody planned.

**Lessons:** one query cannot serve a question whose parts need different words; a bridge question's second query depends on the first answer; inspect *why* a result was found, not only whether.

---

## 7. Challenges and how they were solved

1. **Scanned PDFs with a bad hidden text layer.** A 1960 Gazette scan passed as "digital" because archive.org had added an error-full text layer. → Classify by image coverage (≥ 80% = scanned); force OCR when most pages are scanned.
2. **Tables not retrieved.** The intro sentence was a separate chunk, leaving the table as pipes and numbers. → Chunking rule 6: "total net sales in 2026" went from "couldn't find" to "$478,834 million [p. 29]".
3. **Tiny chunks as noise.** 10-word chunks matched unrelated questions. → Rule 7, merge tiny chunks.
4. **One name list repeated 44 times.** A scanned name list became one giant table whose "header" was repeated in 41 parts. → Repeat headers only if ≤ 40 words.
5. **Dense misses exact terms, BM25 misses synonyms.** → Hybrid (RRF) + reranker.
6. **A small test set gave the wrong conclusion.** On 11 questions the reranker added nothing; on 55 it was clearly best (MRR 0.77 → 0.86). → Decide on bigger test sets.
7. **Answers that OCR destroyed.** → Page-image retrieval + Gemini reading the image; only confident text hits may pick page images. An answer no text-only RAG can give.
8. **Citations not clickable.** With page images in the prompt, Gemini wrote `[Source 4, p. 2, 3; Page image, p. 3]`. Found by the answer eval. → Prompt rule + tolerant parser: evidence citations 91% → 100%.
9. **The reranker undid the graph.** Both facts were in the graph pool, but the reranker scored them −3.9 and −5.1 and picked useless chunks. → Pick graph chunks by walk score; reranker only breaks ties: 57% → 86% on the first test.
10. **Same thing, different names.** "Transformer (big)" / "big models"; "NVIDIA P100 GPU" / "P100". → Merging with embedding candidates + Gemini decision + the "not same" guard (needed 4 times on Apple).
11. **Wrong ground truth.** On a dual-axis chart the model was right and the answer key wrong. → Corrected; evaluation catches your own mistakes too.
12. **"Multi-hop" questions that were not.** Two facts in one sentence of the abstract; "two questions joined with *and*". → `--check` warns; split into bridge and compare.
13. **API limits and cost.** Credits ran out mid-extraction; free tier per-minute and daily limits; a crash lost 320 merge decisions. → Everything resumable and saved per chunk / batch; 65 s wait on 429; a model at its daily limit is skipped; clear stop on 402.
14. **Frontend bugs.** Safari crashed in PDF.js (ReadableStream not async-iterable) → polyfill. The window scrolled instead of the viewer → `minmax(0,1fr)` grid row. The viewer jumped to page 1 (loading pages suspended the whole viewer) → a Suspense boundary per page + `startTransition`. Scroll jank → update once per animation frame.
15. **Model crashes on a Mac.** ColQwen2.5 / ColSmol segfaulted loading onto the Apple GPU in bfloat16 → load in float32 on CPU, then move.
16. **Slow, overloaded LLM.** 184 s for one answer → switched models, 90 s timeout, retries: ~7–11 s.
17. **The graph found a right answer for the wrong reason (M01).** Dates acted as hubs, and a hub's 6 chunks were picked by **text order of ids** (c1, c10, c11, c12, c2…, effectively random); `--explain` hid some starting points. → Dates excluded; hub chunks ordered by relevance; `--explain` shows every starting point. Bridge retrieval 50% → 58%.
18. **The graph reached the answer but could not rank it.** After the fixes, 16 starting entities all scored 0.7, so all 20 candidates tied and the tie-break chose "Foreign Exchange Rate Risk". → Not tuned (one question needs it; overfitting risk); this is where the agent helps.
19. **Is the agent better because it thinks or because it reads more?** It passed ~7 chunks, normal modes 5. → Two controls with the same budget: rerank with 7 chunks (67% on bridge) and query decomposition (75%); the agent's 100% is real, and its edge is on questions whose next search depends on the first answer.
20. **Agent safety.** Searching harder could make it talk itself into an answer. → Stopping rules in code (no invented sources, no repeats, 5 rounds), "not in the document" questions in the eval: 0% wrong everywhere, refusals only after 3–4 searches.
21. **Question design for agents.** The first 10 agent questions were too easy (numbers in one table, refusals): every mode scored 8–10/10. → 12 harder questions with parts in different places and dependent steps: 50% → 83%.
22. **The agent forgot to look at the page image (T11).** On the faded Gazette scan the answer exists only in the image; normal mode always sends page images for scanned documents, but the agent only does when it chooses `visual_search`, and it kept searching text. → Recorded as the agent's one single-fact miss; a fix would be to give the agent the page images automatically for scanned documents (not done, to avoid tuning to one question).

---

## 8. Limitations

- **Small, single-author test set:** 103 questions written by one person from the same documents; only 12 bridge and 12 hard questions (one question = 8 points). One run per question; chart answers and agent paths vary between runs.
- **Judge from the same model family** as the answerer (human check 15/15, all on "correct" verdicts).
- **Chart values are approximate** (peaks tend to be overestimated).
- **Page images in normal mode:** for documents with scans or charts, up to 3 images are sent on every question, even when the text already answers it (safe but slower; only the agent sends images on demand).
- **GraphRAG: where it fails, and why**
  1. **Common entities (hubs) connect to everything** ("Apple", "net sales"): graph-only search on the 10-K, 44%. (Dates excluded; others capped at 6 chunks chosen by relevance.)
  2. **No path when facts share no entity or sentence:** the graph only knows extracted links; table rows often have no relationships.
  3. **It cannot tell which connection matters:** "U.S. dollar" counts as much as "iPhone" (M01 tie at 0.7).
  4. **A wrong starting point sends the walk the wrong way** (the first M01 run started from "2026").
  5. **Slot displacement:** the 2 graph slots replace rerank's #4 and #5 (M03: rank 4 → 6).
  - Fixable inside the graph: 1, 3, 5 (weight specific entities, threshold for graph slots); built into the approach: 2, 4 (where the agent does better). Untuned weights (0.8, 0.5, 0.88); strength 1–10 is subjective; extraction costs one Gemini call per chunk (Apple: 418 calls, 67 min), so uploads do not build a graph automatically.
- **Agent:** misses answers that exist only in a page image unless it decides to call `visual_search` (T11); slowest mode (16–28 s, 3–4 Gemini planning calls + the answer); passes more chunks (~7); can wander (A07: two exact-term searches and a page image for a number in a text table); one run per question.
- **Local Qdrant** allows one process at a time (the eval and the server cannot run together); production would run it as a service.

---

## 9. Key decisions and why (quick answers)

- **Rule-based chunking:** Docling already gives structure; rules are free, instant, deterministic and testable.
- **Hybrid + reranker as default:** best on 55 questions; dense and BM25 cover each other's blind spots.
- **Page images only for scanned / figure pages:** digital text pages are handled better and cheaper by text search.
- **A different model as judge + model-free citation checks:** less self-grading bias; "grounded" catches invented citations without a model.
- **Evidence quotes checked in code:** every graph edge traces to an exact sentence in the PDF.
- **Graph as a helper (graph+rerank), not a replacement:** graph-only lost single-fact questions.
- **Same chunk budget across compared modes:** gains come from better chunks or better reasoning, not more text; the controls (k = 7, decomposition) test exactly that.
- **Own agent loop, no framework:** ~150 lines, every step explainable and logged; a framework (e.g. LangGraph) would fit bigger multi-agent systems.
- **Rules enforced in code, not only in the prompt:** the agent cannot stop with invented sources or repeat itself, whatever the model says.
- **The agent reuses the normal answer step:** same citations, highlighting and safety prompt in every mode.
- **Each method has its place:** a sensible production design is routing: decomposition (or normal search) by default, the agent for questions that need dependent steps.