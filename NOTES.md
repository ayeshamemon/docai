# DocAI: project notes

Ask a PDF anything and see the exact passage behind every answer. A multimodal RAG app for
real-world PDFs (tables, charts, scanned and handwritten pages), with a GraphRAG mode for
questions that join facts from different pages.

---

## 1. Headline results

| What | Result |
|---|---|
| Single-fact answers (55 questions, 6 documents) | **96% correct, 0% wrong, 100% of citations grounded** |
| Retrieval, hybrid + reranker (55 questions) | **96% hit@5**, MRR 0.86 |
| Multi-hop "bridge" answers (12 questions), rerank → GraphRAG | **50% → 67% correct** |
| Multi-hop "compare" answers (14 questions), rerank → GraphRAG | 86% → 93% correct |
| Single-fact answers with GraphRAG | 96% (unchanged), 0% wrong |
| Faded 1960 scan | 100% correct, incl. an answer that exists only in the page image |

---

## 2. Tech stack

| Part | Tool | Why |
|---|---|---|
| PDF parsing | **Docling** (layout, reading order, TableFormer for tables) | Gives structure + bounding boxes; tables as Markdown |
| OCR | **RapidOCR** (inside Docling) | Free, local; only turned on for scanned pages |
| Page classification, page images | **PyMuPDF** | Fast; labels pages digital / scanned / blank; saves PNGs |
| Text embeddings | **bge-small-en-v1.5** via **fastembed** (384 dimensions) | Small, CPU-friendly, free |
| Keyword search | **BM25** (fastembed `Qdrant/bm25`, sparse vectors) | Exact words and numbers |
| Reranker | **bge-reranker-base** (cross-encoder, fastembed) | Re-reads each (question, chunk) pair; better ranking |
| Visual retrieval | **ColSmol-256M** (ColPali family, `colpali-engine`) | Searches page *images* (charts, scans) |
| Vector database | **Qdrant** (local mode) | Dense + sparse vectors in one point, multi-vector MaxSim, RRF fusion |
| LLM | **Gemini Flash** (`google-genai`): 3.6-flash answers, 3.8-flash judges/fallback | Reads text + page images; streaming |
| Graph | **NetworkX** | One graph per document |
| Backend | **FastAPI**, Server-Sent Events, background worker | Streamed answers, uploads processed in the background |
| Frontend | **Next.js 16**, React, **react-pdf** (PDF.js), react-markdown, Tailwind CSS v4 | PDF viewer with highlights, streamed chat |

Runs on a MacBook Air (models on CPU / Apple GPU); only Gemini is a cloud API.

---

## 3. How it works, step by step

```
PDF → Parse → Chunk → Index (text + page images [+ graph]) → Retrieve → Answer with citations → UI
```

### Step 1: Parse (`backend/ingest/parse.py`)
1. Hash the file → `doc_id` (the same file is never processed twice).
2. **Classify every page** with PyMuPDF:
   - `digital`: real text layer → no OCR
   - `scanned`: one image covers ≥ 80% of the page → OCR, even if a (bad) hidden text layer exists
   - `blank`: < 50 characters and no large image → skipped
   - ≥ 50% scanned pages → forced full-page OCR for the whole document
3. Save every page as a PNG (144 DPI) for page-image retrieval and the viewer.
4. **Docling** reads layout, reading order (also two-column papers), headings, tables (→ Markdown), figures and captions, each with a **bounding box** (used later to highlight the evidence).

### Step 2: Chunk (`backend/ingest/chunk.py`), rule-based, no model
1. A new heading starts a new chunk; the heading is added to the start of every chunk under it.
2. Paragraphs are joined into chunks of up to **300 words**.
3. A paragraph is never cut, unless it alone is longer than the limit (then split at sentence ends).
4. **Tables are their own chunk**; big tables are split by rows, repeating the header row (only if the header is short, ≤ 40 words).
5. Figures with captions get their own chunk; figures without captions are left to visual search.
6. **Table intro sentences stay with their table** ("The following table shows net sales…").
7. **Tiny chunks** (< 30 words) are merged into a neighbour (never into a table).
8. Every chunk keeps its pages and bounding boxes (for clickable citations).

Why 300 words: largest size with no truncated text for the embedder's 512-token limit (tested 150 / 300 / 500).

### Step 3: Index
- **Text** (`index.py`): each chunk → one Qdrant point with two vectors: dense (bge-small) and sparse (BM25, with IDF so rare words count more). The full chunk is stored as payload.
- **Page images** (`visual_index.py`): only scanned pages and pages with figures (digital text pages are left to text search). ColSmol turns each page into ~1,000 patch vectors; Qdrant scores them with **MaxSim** (each question word finds its best-matching patch).

### Step 4: Retrieve (`backend/retrieval/search.py`)
| Mode | What it does |
|---|---|
| `dense` | Meaning only |
| `bm25` | Exact words only |
| `hybrid` | Dense top 20 + BM25 top 20, fused with **Reciprocal Rank Fusion** (RRF) inside Qdrant |
| **`rerank`** (default) | Hybrid top 20 → cross-encoder reranker → top 5 |
| `visual` | Page images (ColSmol, MaxSim) |
| **`graph+rerank`** | Rerank's best 3 + 2 chunks found by walking the graph (Step 8) |

### Step 5: Answer (`backend/retrieval/ask.py`)
- Gemini gets the **top 5 chunks + up to 3 page images**.
- **Choosing page images:** the top visual page, plus pages of *confident* text hits (reranker score > 0) that have an image, filled up with more visual pages; max 3. Skipped for documents without page images.
- **Prompt rules:** answer only from the sources; cite every fact as `[p. N]`; read tables by row and column; say "I couldn't find this in the document" instead of guessing.
- **Reliability:** streaming; 90 s timeout; retry on 429 / 500 / 503 / 504 / timeout; fallback model; no retry once text has started (no duplicated text).

### Step 6: The app
- **Backend (FastAPI):** all models loaded once at startup; one shared Qdrant connection with a lock. Upload → background worker (parse → chunk → index → visual index) with live status; duplicates detected by file hash. `POST /chat` streams Server-Sent Events: `sources` → `token`… → `done` (timing).
- **Frontend (Next.js):** document list with drag-and-drop upload and progress; PDF viewer (lazy page rendering, zoom); streamed chat with Markdown; **clickable citations** that scroll to the page and **highlight the source passage** (bounding boxes scaled to the rendered page); mode menu (Rerank / Hybrid / Graph) and a page-images toggle; dark mode; landing page.

### Step 7: Evaluation (`backend/eval/`)
- **Test set:** 81 questions over 6 documents (Apple 10-K, Attention paper, BERT paper, IMF report with charts, 1960 Gazette faded scan, EPA letter clean scan): 55 single-fact questions in 9 categories (table, formula table, number, exact term, synonym, prose, chart, clean scan, faded scan) + 26 multi-hop.
- **Retrieval eval:** each question has a **keyword** that must appear in the retrieved chunk (multi-hop: one keyword per fact, *all* must be in the top 5). Charts are judged by page. Metrics: **hit@5** and **MRR**. Keywords are only for scoring; the RAG system never sees them. `--check` validates every keyword against the chunks first.
- **Answer eval:** same pipeline as the app. One model answers (gemini-3.6-flash, no fallback); a **different model judges** (gemini-3.8-flash) against a hand-written answer key: correct / partial / wrong / abstained. Two checks need no model:
  - **evidence:** the answer cites a page that holds the answer (multi-hop: a page for *every* fact)
  - **grounded:** every cited page was actually given to the model (no invented citations)
- **Human check:** 15 random verdicts graded by hand without seeing the judge's verdict: 15/15 agreement.

### Step 8: GraphRAG (`backend/graph/`)

**Idea:** normal search finds chunks that *sound like* the question. Bridge questions need a second fact that does not sound like the question ("Which GPUs were used, and what TFLOPS was assumed for that GPU?": the TFLOPS footnote never mentions training). A graph is a **map of the document**: the things in it and how they are connected. From fact 1, the map leads to fact 2.

**8a. Extraction (`extract.py`):** Gemini reads each chunk (temperature 0, JSON output) and returns:
- **Entities:** important named things, name as written, one of 8 fixed types (Organization, Person, Product, Place, Date, Metric, Concept, Event), one-sentence description using only this chunk, with numbers ("iPhone net sales 2026 → $254,098M, up 21%").
- **Relationships:** only links the text states, each with a description, a **strength 1–10** (how central the link is) and an **exact evidence quote** (5–25 words).
- Not entities: generic words ("the Company") and labels of document parts ("Table 3", "Section 3.2").
- Skipped: chunks under 8 words and the bibliography (list of cited papers) at the end of papers.

**Checks in code (the model cannot get past these):**
- The evidence quote must really be in the chunk (ignoring case, spacing, table pipes, "$ 254" vs "$254"), at least 8 characters → otherwise dropped as invented.
- Both ends must be entities of that chunk; no self-links.
- Unknown type → Concept. Strength clamped to 1–10; missing → 5.
- Result: **98–100% of relationships kept** on every document, including OCR'd scans.

**8b. Merging (`build.py`): the same thing in different chunks → one node.** Safest first, because a wrong merge is worse than a missed one:
1. **Same name after cleanup** (case, punctuation, "the", plural "s") → merge automatically.
2. **Similar names** (embedding similarity of name + description ≥ 0.88, or one name inside the other: "P100" in "NVIDIA P100 GPU") → only a "maybe".
3. **Gemini decides each "maybe"**: same / not same; "if unsure, not same". 80 pairs per call; decisions cached and saved after every batch.
4. **Type rule:** Metric and Date only merge with the same type ("iPhone" never merges with "iPhone net sales"); other types may merge, because Gemini labels the same thing differently ("big models" Concept, "Transformer (big)" Product).
5. **Never across documents.**
- **The "not same" guard:** two groups are never joined if Gemini said "not same" for any pair across them. Stops chains like Transformer = big and Transformer = base → base = big.
- A merged node keeps all descriptions, all its chunks and pages, and all edges; duplicate edges keep every evidence quote and the highest strength.

**8c. Three kinds of connection:**
| Kind | Connects | Made by |
|---|---|---|
| Merging | the same thing mentioned in different chunks | name + Gemini |
| **Stated edge** | two things the text links in one sentence | Gemini, quote checked |
| **Structural edge** | entity → its chunks; chunk → its section; chunk → its page; chunk → next chunk in the same section | code, from the layout |

Structural edges join every document into one connected map (IMF: 251 pieces → 1).

**8d. Entity index (`index_entities.py`):** every entity is embedded ("name, aliases (type): descriptions") into a Qdrant collection, so a question can find where to start.

**8e. Graph search (`search.py`), mode `graph+rerank`:**
1. Normal search (dense + BM25 → RRF → reranker) → keep the best **3** chunks (usually fact 1).
2. **Starting points:** the 5 entities closest to the question + every entity inside those 3 chunks.
3. **Walk 1 step** and score chunks (the *walk score*):
   - a chunk that mentions a starting entity → its similarity score
   - a chunk of an entity linked by a stated edge → score × 0.8 × strength/10
   - the next chunk in the same section → score × 0.5
   - a chunk reached several ways keeps its highest score; an entity gives at most 6 chunks; page edges are not walked; keep the best 20.
4. Pick **2 chunks by walk score** (reranker only breaks ties), skipping ones already chosen.
5. 3 + 2 = **5 chunks** to Gemini: the same budget as normal search, so comparisons are fair.

Why 1 step only: the map is fully connected, so longer walks reach everything. Why pick by walk score: see challenge 9.

Two more rules (added after checking M01 with `--explain`, challenge 17):
- **Date entities are hubs** ("2026" touches almost every chunk of a 10-K): never starting points, never walked through.
- **A hub entity's 6 chunks are chosen by relevance** to the question (hybrid search rank), not by chunk order.

**8f. In the app:** "Graph" in the mode menu (disabled for documents without a graph); sources found by the graph get a **via graph** tag and the connection that brought them ("P100 GPU → big models: trained on").

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

- Dense and BM25 fail on **different** questions (BM25: synonyms, tables; dense: exact terms, formula tables); hybrid misses neither.
- The reranker is best overall, mostly by ranking tables and numbers higher (tables MRR 0.30 → 0.67), but slightly weaker on noisy OCR text.

### Answers, single-fact (55 questions, rerank)
| Correct | Partial | Wrong | Abstained | Evidence cited | Grounded | Avg time |
|---|---|---|---|---|---|---|
| **96%** | 2% (a chart value) | **0%** | 2% (the one retrieval miss) | 100% | 100% | 7.0 s |

### GraphRAG (all 81 questions)
**Retrieval, hit@5 (all facts in the top 5):**
| | n | rerank | graph-walk | **graph+rerank** |
|---|---|---|---|---|
| Single-fact | 55 | 96% | lower (drops several) | **98%** |
| Compare | 14 | 79% | 57% | **86%** |
| Bridge | 12 | 42% | 33% | **58%** |
| All | 81 | 85% | – | **90%** |

**Answers (same answer and judge models):**
| | rerank | **graph+rerank** |
|---|---|---|
| Bridge (12) | 50% correct | **67%** |
| Compare (14) | 86% | **93%** |
| Single-fact (55) | 96% | **96%** |
| Wrong | 0% | **0%** |

- Newly correct answers: M02 (Greater China sales + currency effect), M06, M20, M24 (P100 + TFLOPS footnote), Q12 (failed in every mode since the start).
- Retrieval after the graph fixes (challenge 17): gained 6 questions, lost 1 (M03, see limitations: slot displacement).
- Still not fully answered: M01 (why iPhone grew), M05, M09, M22.
- `graph-walk` (graph only, ordered by walk score) and pure `graph` (graph only, reranked) are comparisons: both are weaker than normal search alone; the graph works as a **helper**, not a replacement.

**Graph sizes:** Apple 10-K 889 entities / 906 stated edges; IMF 640 / 441; Gazette 599 / 370; BERT 176 / 164; Attention 138 / 118. Gemini judged 5–20% of merge candidates "same"; hand check of merges: all clearly correct except two doubtful author-citation merges in BERT.

---

## 5. Challenges and how they were solved

1. **Scanned PDFs with a bad hidden text layer.** A 1960 Gazette scan passed as "digital" because archive.org had added an error-full text layer (`Prcsident`). → Classify by image coverage (≥ 80% = scanned) and force OCR when most pages are scanned.
2. **Tables not retrieved.** The intro sentence ("The following table shows net sales…") was a separate chunk, leaving the table as pipes and numbers. → Chunking rule 6. "Total net sales in 2026" went from "couldn't find" to "$478,834 million [p. 29]".
3. **Tiny chunks as noise.** 10-word chunks ("To the Shareholders and the Board…") matched unrelated questions. → Rule 7, merge tiny chunks.
4. **One name list repeated 44 times.** A scanned two-column name list became one giant table; repeating its "header" (full of names) in all 41 parts put one name in 44 chunks. → Repeat headers only if ≤ 40 words.
5. **Dense search misses exact terms, BM25 misses synonyms.** → Hybrid (RRF), then a reranker.
6. **A small test set gave the wrong conclusion.** On 11 questions the reranker added nothing; on 55 it was clearly best (MRR 0.77 → 0.86). → Decide on bigger test sets.
7. **Answers that OCR destroyed.** "earned leave for 17 days" became `ear   s       ty`. → Page-image retrieval + Gemini reading the image. Then low-confidence text hits were taking the image slots → only use text-hit pages with reranker score > 0. Result: an answer no text-only RAG can give.
8. **Citations not clickable.** With page images in the prompt, Gemini wrote `[Source 4, p. 2, 3; Page image, p. 3]`; the parser found nothing. Found by the answer eval. → Prompt rule + tolerant parser (same rules in app and eval): evidence citations 91% → 100%.
9. **The reranker undid the graph.** For the P100 question both facts were in the graph pool, but the reranker gave the footnote −3.9 and the training chunk −5.1 and picked useless chunks: it judges "does this chunk answer the whole question", and fact 2 alone never does. → Pick graph chunks by walk score (connection); reranker only breaks ties. 7-question test: 57% → 86%.
10. **Same thing, different names.** "Transformer (big)" / "big models" / "big"; "NVIDIA P100 GPU" / "P100". → Merging with embedding candidates + Gemini decision + the "not same" guard (needed 4 times on Apple).
11. **Wrong ground truth.** On a dual-axis chart the model picked the series on the right-hand scale, which was correct; the answer key had misread it. → Corrected; evaluation catches your own mistakes too.
12. **Multi-hop questions that were not really multi-hop.** One question's two facts sat in one sentence of the abstract; a batch of "two questions joined with *and*" was too easy. → `--check` warns when one chunk holds every fact; split multi-hop into **bridge** and **compare**.
13. **API limits and cost.** Prepaid credits ran out mid-extraction; the free tier hit per-minute and daily limits; a crash lost 320 merge decisions. → Everything resumable and saved per chunk / per batch; on a 429 wait 65 s; a model stuck at its daily limit is skipped for the rest of the run; clear stop message on 402.
14. **Frontend bugs.**
    - Safari crashed in PDF.js (its ReadableStream is not async-iterable) → polyfill + fallback.
    - The whole window scrolled instead of the viewer (a grid row grew to fit the PDF) → `minmax(0,1fr)` row.
    - The viewer jumped back to page 1 while scrolling: newly visible pages suspended the nearest Suspense boundary (the whole viewer), which remounted it → a Suspense boundary per page + `startTransition`.
    - Scroll jank → update state once per animation frame.
15. **Model crashes on a Mac.** ColQwen2.5 and ColSmol segfaulted loading their adapter onto the Apple GPU in bfloat16 → load in float32 on CPU, then move to the GPU.
16. **Slow, overloaded LLM.** 184 s for one answer with an overloaded model → switched main/fallback models, 90 s timeout, retries: ~7–11 s.
17. **The graph found a right answer for the wrong reason.** On M01 ("Which product grew fastest, and why?") `--explain` showed the starting entities were dates ("2026", "Fourth Quarter 2026"), and the reason chunk entered only because it mentions "2026", like a dozen unrelated chunks (stock buybacks, taxes). Two bugs behind it: dates acted as hubs, and an entity in many chunks kept its first 6 chunks by **text order of the ids** (c1, c10, c11, c12, c2… effectively random). `--explain` also hid the starting points that come from the top rerank chunks. → Dates excluded as starting points and hops; a hub's chunks ordered by relevance; `--explain` shows every starting point and its source. Graph eval re-run after the fix.

---

## 6. Limitations

- **Small, single-author test set:** 81 questions, written by one person from the same documents, mostly lookups; only 12 bridge questions (one question = 8 points). One run per question; chart answers vary between runs.
- **Judge from the same model family** as the answerer (human check 15/15, all on "correct" verdicts).
- **Chart values are approximate** (peaks tend to be overestimated).
- **GraphRAG:**
  - **Slot displacement:** the fixed 3 + 2 split pushes out rerank's #4–#5; before the graph fixes this lost 2 questions (M03, M26) while gaining 5; after them, 1 lost (M03) and 6 gained. A 4 + 1 split or a walk-score threshold could help, but tuning on the same 81 questions would overfit.
  - **Untuned weights:** 0.8 (stated edge), 0.5 (next chunk) and the 0.88 merge threshold are reasonable guesses.
  - **Strength is subjective:** Gemini's 1–10 is uncalibrated (mostly 7–9), so it only adjusts scores slightly.
  - **Hub entities** ("Apple", "net sales") connect to dozens of chunks; capped at 6 chunks per entity.
  - **Cost and time:** extraction is one Gemini call per chunk (Apple 10-K: 418 calls, 67 min), so new uploads do not build a graph automatically.
  - **Slower answers:** ~13 s vs ~7 s in the eval run.
  - **Noise:** table rows produce entities without relationships; author-year citations in the text become entities.
- **Local Qdrant** allows one process at a time; production would run it as a service.

---

## 7. Key decisions and why (quick answers)

- **Rule-based chunking, not semantic/LLM chunking:** Docling already gives structure; rules are free, instant, deterministic and testable.
- **Hybrid + reranker as default:** best on 55 questions; dense and BM25 cover each other's blind spots.
- **Page images only for scanned / figure pages:** digital text pages are handled better (and cheaper) by text search.
- **A different model as judge + model-free citation checks:** reduces self-grading bias; "grounded" catches invented citations without any model.
- **Evidence quotes checked in code:** every graph edge can be traced to an exact sentence in the PDF.
- **Graph as a helper (graph+rerank), not a replacement:** graph-only modes lost single-fact questions; the hybrid design keeps normal search and only adds connected chunks.
- **Fixed chunk budget (5) for every mode:** a fair comparison; gains come from better chunks, not more chunks.

---

## Step 9: Agentic RAG (in progress)

**Idea:** today's modes search once and answer. An agent works like a researcher: it plans, searches, reads, notices what is missing, writes a better search itself, checks, and only then answers. Example (M01, unsolved by every mode): search "net sales by product" → iPhone grew most → *new* search "why did iPhone net sales increase" → "higher net sales of Pro models". Normal search cannot ask the second question because it does not know the answer is iPhone until it reads the table.

**Plan:** 9.1 tools → 9.2 the loop (Gemini picks one tool per round, JSON) → 9.3 stopping rules → 9.4 answer + self-check → 9.5 eval on the same 81 questions (correct / wrong / steps / time) → 9.6 "Agent" mode in the app with live steps. Plain Python loop, no agent framework (every line explainable, easy to log).

**Stopping rules (planned):** split the question into parts first and answer only when every part has a source; at most 5 rounds, then answer with what was found and say what is missing; the same search twice is blocked; nothing found → "not in the document".

### Worked example: M01 in every mode (why an agent is needed)

**Question (M01, bridge):** "Which product grew fastest in 2026, and why?" Two parts:
- part 1 (which product): the net sales table, chunk **c111**, p. 29: iPhone +21%
- part 2 (why): chunk **c112**, p. 29: "iPhone net sales increased … due to higher net sales of Pro models"

The two chunks sit **next to each other** in the PDF. Every mode failed this question in the eval.

**Experiment 1: the same three queries in both tools** (does c111 / c112 reach the top 5?)

| Query | text_search (normal) | graph_search (GraphRAG) |
|---|---|---|
| "Which product grew fastest in 2026, and why?" (the full question) | ❌ table ❌ reason | ❌ table ✅ reason |
| "net sales by product 2026" | ✅ table ❌ reason | ✅ table ❌ reason |
| "why did iPhone net sales increase in 2026" | ❌ table ✅ reason | ❌ table ✅ reason |

**What it shows:**
- **No single query finds both parts, with either tool.**
- **The query matters more than the tool:** on the last two queries both tools return the same two chunks.
- **Why normal search misses the reason:** the question never says "iPhone" (that is what is asked). The document has six near-identical paragraphs ("Mac / Americas / Greater China / iPhone net sales increased … due to …"); without knowing it is the iPhone, the right one cannot be told apart.
- **Why the full question misses the table:** "grew" and "why" match explanation paragraphs; a table of pipes and numbers scores low against such words.
- **Once "iPhone" is in the query, plain search finds the reason at once:** the hard part is not searching, it is *knowing what to ask*, and that needs part 1 first.

**Experiment 2: why did GraphRAG find the reason?** (`graph.search --explain`)
- *First look:* the starting entities found by meaning were dates ("2026", "Fourth Quarter 2026") and c112 entered the pool only because it mentions "2026", like a dozen unrelated chunks (buybacks, taxes, inventories). **A right answer for the wrong reason.**
- *Bugs found:* (1) dates act as hubs (a 10-K mentions 2026 everywhere); (2) a hub's 6 chunks were picked by **text order of chunk ids** (c1, c10, c11, c12, c2, …: effectively random); (3) `--explain` hid the starting points that come from the top rerank chunks.
- *Fixes:* dates are never starting points or hops; a hub's chunks are ordered by hybrid-search relevance; `--explain` shows every starting point and its source.
- *After the fixes:* iPhone becomes a starting point (from a regional paragraph that mentions it); **both** the table and the reason are now in the graph pool. But the top 3 rerank chunks give **16 starting entities, all with score 0.7** (laptops, Mac, Services, U.S. dollar, renminbi, …), so **all 20 pool chunks tie at 0.700** and the reranker breaks the tie: it picked "Foreign Exchange Rate Risk" (useless) and c112, and the table lost (tables score low with the reranker).
- *Open problem:* the graph now reaches both halves but cannot tell which are important: "U.S. dollar" from a regional paragraph counts as much as "iPhone". Candidate fix: starting scores by the source chunk's rank and by how specific an entity is (an entity in 3 chunks says more than one in 40). Decided to measure the three fixes on all 81 questions first, instead of tuning on one question (overfitting).

**Experiment 3: the three fixes on all 81 questions (retrieval, hit@5):**

| | rerank | graph+rerank before fixes | **graph+rerank after fixes** |
|---|---|---|---|
| Single-fact (55) | 96% | 98% | **98%** |
| Compare (14) | 79% | 86% | **86%** |
| Bridge (12) | 42% | 50% | **58%** |
| All (81) | 85% | 89% | **90%** |

- **M26 recovered** (rank 6 → 5): BERT-base sized like OpenAI GPT → GPT's GLUE score. No question got worse.
- Now graph+rerank vs rerank: **gained 6** (Q12, M02, M06, M20, M24, M26), **lost 1** (M03, slot displacement).
- M01 still missed (rank 10): the tie problem from experiment 2 is real, but fixing it was postponed (only one question clearly needs it; risk of overfitting).

**What an agent does instead (the plan):** two focused searches in order:
1. "net sales by product 2026" → table → "it's the iPhone, +21%"
2. "why did iPhone net sales increase" (written *after* step 1) → c112 → "Pro models"

Or simply `read_next(c111)` → c112, because the reason is the next chunk. This is what no single query, and no single retrieval mode, can do.

**Lessons:**
- One query cannot serve a question whose parts need different words (a table for "which", a paragraph for "why").
- A bridge question's second query depends on the first answer; only something that reads and then searches again can write it.
- Inspect *why* a result was found, not only *whether*: the first GraphRAG "success" on M01 was luck, and checking it found two real bugs.

### 9.1 Tools (`backend/agent/tools.py`)
Started with 6 tools; more (read a whole page, read a full split table, document outline, graph entity lookup, look at one page image) only if the logs show they are needed: too many similar tools make agents pick worse.

| Tool | Uses | For |
|---|---|---|
| `text_search(query)` | hybrid + reranker, top 5 | most questions |
| `graph_search(query)` | graph+rerank | a fact linked to something already found |
| `visual_search(query)` | page images, top 2 | charts, scans (images go to Gemini with the final answer) |
| `read_next(chunk)` | chunk before and after, from chunks.json | an explanation right under a table (M22, M05) |
| `find_exact(term)` | every chunk with the exact word / number | exact terms and numbers (like "h = 8") |
| `calculate(expression)` | safe arithmetic (parsed, no eval) | growth, differences, totals |

- Each tool returns what the agent reads (chunks with short ids like `c23` and pages) plus the chunks themselves, kept for the final answer and its citations.
- A tool error is returned to the agent as text, never a crash.
- **Calculator safety:** only numbers, + − × ÷ ** %, brackets and round / abs / min / max; anything else (e.g. `__import__`) is refused. Thousands separators are removed only between digits ("254,098" → 254098) so `round(x, 1)` still works (a bug found in testing).