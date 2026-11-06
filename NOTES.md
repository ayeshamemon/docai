# Project notes (for the README)

Running log of design decisions, experiments and findings. Turn these into README sections at the end.

---

## Step 1: PDF ingestion

**Pipeline (`ingest/parse.py`)**
1. Hash the file → `doc_id` (re-uploads are skipped: cache)
2. PyMuPDF: label every page digital / scanned / blank, and save a PNG of each page
3. Decide OCR: any scanned page → OCR on; ≥50% scanned → forced full-page OCR
4. Docling: layout model (paragraphs, headings, tables, figures, reading order), TableFormer (table structure → Markdown), RapidOCR (scanned pages), bounding boxes
5. Save `elements.json` (each piece: page, type, label, content, bbox), `meta.json` (page types, OCR used, counts, time) and `pages/`

**Tools**
- PyMuPDF: page classification and page images (PNG, 144 DPI)
- Docling: layout, reading order, tables (TableFormer → Markdown), figures, captions, bounding boxes
- OCR: RapidOCR (Docling's built-in engine), turned on only when needed

**Page classification (3 types)**
- `digital`: real text layer → no OCR
- `scanned`: one image covers ≥80% of the page → OCR, even if a hidden text layer exists
- `blank`: under 50 characters and no large image → skipped
- If ≥50% of pages are scanned → forced full-page OCR (ignores bad hidden text layers)

**Why the scanned rule matters (finding)**
- The first version only checked "has text?". An archive.org scan (1960 Gazette of India) passed as 15/16 "digital" because it had a hidden OCR layer from archive.org with many errors (`Prcsident`, `Endia`).
- The first 10-K run: one nearly empty page (p. 113) turned OCR on for all 123 pages. The `blank` type fixed this (OCR=False).

**Test results**

| PDF | Pages | Result |
|---|---|---|
| Attention Is All You Need | 15 | 150 text, 4 tables, 7 figures; Table 1 rows/columns intact |
| BERT (two-column) | 16 | correct reading order (left column, then right); 8 tables, 5 figures |
| Apple 10-K | 123 | 59 tables; income statement intact, totals add up (355,533 + 123,301 = 478,834) |
| IMF report (charts) | 46 | 26 figures detected; chart values not extracted (left for ColPali) |
| EPA letter (clean scan, 3 pp) | 3 | near-perfect OCR |
| Gazette of India 1960 (faded scan) | 16 | ~76% of sampled words correct (e→c errors from faded print) |

**Speed (MacBook Air, CPU):** digital ~16 s for 16 pages; 10-K ~90 s for 123 pages; the first run is slower (model download).

**Known limitations**
- Superscripts flattened (`O(n2 · d)` instead of `O(n² · d)`)
- Letterhead/page headers can appear out of order in scanned letters
- Very large tables can drop a few cells (2 of 447 in one IMF table)
- Docling detects charts but does not read their values → handled by ColPali + VLM (Step 2)
- Faded old scans → OCR errors → handled by ColPali + VLM reading page images

---

## Step 2.1: Chunking

**Approach:** rule-based, structure-aware chunking (no model). Docling already provides the structure, so rules are free, instant and deterministic. Semantic or LLM chunking was not needed.

**Rules**
1. A new heading starts a new chunk; the heading is prepended to every chunk under it
2. Paragraphs are joined whole into chunks of up to ~300 words (heading included)
3. Paragraphs are never cut, except a single paragraph longer than the limit → split at sentence ends
4. Tables are always their own chunk; large tables are split by rows, repeating caption and header row in every part
5. Figures with captions → own chunk; figures without captions → skipped (ColPali covers them)
6. Table intro sentences stay with their table: short text (≤60 words) right before a table that ends with ":" or says "following table / table below / as follows / summarizes" is moved into the table chunk (repeated in every part of a split table)
7. Tiny text chunks (<30 words) are merged into the next text chunk, else the previous one, if the result stays ≤300 words; tables and figures are never merged
8. Every chunk keeps its pages and bounding boxes (for clickable citations)

**Why rules 6 and 7 were added (from search tests)**
- Rule 6: "total net sales in 2026" missed the revenue tables because the intro sentence ("The following table shows net sales by category for 2026…") was a separate chunk, leaving the table with only pipes and numbers
- Rule 7: 10-word chunks like "To the Shareholders and the Board of Directors of Apple Inc." matched unrelated questions (ranked #2/#3 for the employee question)

**Experiment: chunk size (150 / 300 / 500 words)**
Embedder limit: 512 tokens (bge-small). `over_limit` = chunks whose end would be truncated.

Before table/paragraph splitting:

| Doc | Size | Chunks | Over limit |
|---|---|---|---|
| Attention | 150 | 66 | 1 (table) |
| Attention | **300** | **49** | **1 (table)** |
| Attention | 500 | 44 | 4 |
| Apple 10-K | 150 | 602 | 6 (4 tables, 2 paragraphs) |
| Apple 10-K | **300** | **462** | **6** |
| Apple 10-K | 500 | 393 | 69 |

After adding table row-splitting and sentence splitting (Apple 10-K):

| Size | Chunks | Table chunks | Max words | Over limit |
|---|---|---|---|---|
| 150 | 779 | 105 | 183 | 0 |
| **300** | **482** | **70** | **316** | **0** |
| 500 | 395 | 61 | 518 | 70 |

**Findings**
- 300 words is the largest size with no truncated text on both documents → chosen as default
- Chunk size cannot fix big tables (they were over the limit at every size); row-splitting with repeated headers fixed them
- Average chunk size is well below the limit (~130–150 words), because headings close chunks early in structured documents

---

## Step 2.2: Embeddings + dense retrieval

**Setup**
- Embedding model: `BAAI/bge-small-en-v1.5` via `fastembed` (384 dims, 512-token limit, CPU-friendly, free, local)
- Separate query and passage embedding modes (bge query instruction)
- Vector DB: Qdrant (local mode, saved to a folder; Docker later), cosine distance
- Each point stores the full chunk (text, pages, boxes, type) as payload, for citations

**First search test**
- Query: "What is the complexity of self-attention?" (Attention + 10-K indexed together)
- Top 5 all from the correct document; #1–#3 from Section 4 "Why Self-Attention" (pp. 6–7)
- **Miss:** Table 1 (the exact answer, `O(n2 · d)`) was not in the top 5
- Reason: tables are mostly symbols and formulas, so their embeddings match natural-language questions poorly
- → Motivation for hybrid search (BM25) and a reranker. Use this as a dense vs. hybrid example in the evaluation.

**Second search test**
- Query: "What were Apple's total net sales?"
- #1 Note 2 – Revenue table (p. 42), #2 net sales by category table (p. 29): both contain the total net sales row → answerable
- The income statement (p. 35) was not in the top 5 (the same total is on p. 42)
- **Finding: heading prefixes help tables get retrieved.** Both table hits start with natural-language headings added by our chunker ("Note 2 - Revenue", "Products and Services Performance"), which match the question. Attention Table 1 sat under "4 Why Self-Attention" with a formula-heavy body and was missed.

---

**Third search test**
- Query: "How many employees does Apple have?" (filtered to the 10-K)
- #1 "Human Capital" (p. 8), the correct section, score 0.732; the rest ≈0.67 (clear gap)
- #2/#3 were tiny generic chunks ("To the Shareholders and the Board of Directors of Apple Inc.", ~10 words) → **finding: tiny chunks add noise**, since they loosely match many questions

**Dense retrieval summary:** right document every time; right section at #1 for prose questions; tables found when their heading matches the question, missed when the body is formula-heavy.

## Step 2.3: Answer generation (Gemini)

**Setup**
- LLM: Gemini Flash (`gemini-3.8-flash`, set in `.env`), streaming, temperature 0.2
- Top-5 chunks sent as numbered sources labelled with type and page
- System prompt: sources only, cite `[p. X]` after every fact, read tables by row/column, say "I couldn't find this in the document" instead of guessing
- Reliability: automatic retry on 503/429/500 (2s/4s/8s backoff) + optional fallback model; no retry after the answer has started (no duplicate text)

**First end-to-end test (T1)**
- Q: "What is the complexity of self-attention?"
- Gemini hit two 503s; retries recovered automatically
- Answer was correct and cited (constant sequential ops, faster than RNNs when n < d) [pp. 6–7]
- **Finding: grounding works.** Table 1 was not retrieved, and Gemini said the exact formula was not in the sources instead of answering O(n²·d) from its own knowledge
- **Finding: the bottleneck is retrieval, not the LLM.** Fixing T1 needs better retrieval (Step 2.4)

**Second end-to-end test (T13)**
- Q: "What were Apple's total net sales in 2026?" (expected: $478,834M, p. 35 / p. 42)
- 3.8-flash returned 503 three times → fallback to 3.6-flash worked
- Retrieval returned only narrative text chunks (Mac, Americas, iPhone, Europe, pp. 28–29); no revenue table
- Gemini answered "I couldn't find this in the document" (grounded, no hallucinated number)
- **Finding: dense retrieval is sensitive to wording.** Without "in 2026" the same question retrieved the revenue tables (#1, #2). Adding the year made prose chunks ("net sales increased during 2026…") rank higher than tables, where "2026" only appears as a column header.
- **Finding: table intro sentences are separated from their tables.** "The following table shows net sales by category…" (p. 29) became its own text chunk.

**Third end-to-end test (T2), before chunking fixes**
- Q: "How many employees does Apple have?" (10-K)
- Answer: "approximately 167,000 full-time equivalent employees [p. 8]" ✅ correct and cited
- Retrieval: #1 Human Capital (0.732); #2/#3 tiny auditor-letter chunks (0.676), #4 stock plan, #5 subsidiaries table
- The answer was right because #1 was right; the noise at #2–#5 did not mislead the LLM

### After chunking fixes (rules 6 + 7), dense retrieval re-run

| Test | Before | After |
|---|---|---|
| T13 net sales 2026 | no table in top 5 (prose only) | ✅ revenue table at #2 (p. 29, 0.78), starting with its intro sentence |
| T2 employees | tiny "To the Shareholders…" chunks at #2/#3 | ✅ gone; auditor sign-off merged into a neighbour (pp. 58–59) |
| T1 self-attention | Table 1 missing | ❌ unchanged (formula table; needs hybrid/reranker) |

**T13 end-to-end after the fix:** "Apple's total net sales in 2026 were $478,834 million [p. 29]" ✅ (before: "I couldn't find this in the document")

**Finding:** the two chunking rules fixed the failures they targeted, with no change to the embedding model; T1 confirms that chunking alone cannot fix formula-heavy tables.

---

## Step 2.4: Hybrid search (dense + BM25) and reranking

**Setup**
- Each chunk stored with two vectors in one Qdrant point: `dense` (bge-small, 384-d, cosine) + `bm25` (sparse, `Qdrant/bm25` via fastembed; Qdrant `IDF` modifier adds word rarity)
- New collection `docai_chunks` (old `chunks` was dense-only); 5 docs, 792 chunks
- Modes (`retrieval/search.py`): `dense`, `bm25`, `hybrid` (dense top 20 + BM25 top 20, fused with RRF inside Qdrant), `rerank` (hybrid top 20 → `BAAI/bge-reranker-base` cross-encoder → top 5)
- `--find "<keyword>"` prints the rank of the first chunk containing the answer keyword in every mode (✅ = top 5)

**Results on the dense failures**

| Test | Dense | BM25 | Hybrid | Rerank |
|---|---|---|---|---|
| T1 "complexity of self-attention" (keyword: `Self-Attention (restricted)`) | ❌ 15 | ✅ 1 | ✅ 3 | ✅ 4 |
| T1 variant "complexity **per layer** of self-attention" | ✅ 4 | ✅ 1 | ✅ 2 | ✅ 3 |
| T6 "Which year had total net sales of 391,035?" | ✅ 1 | ✅ 1 | ✅ 1 | ✅ 1 |
| T6b "391,035" | ✅ 3 | ✅ 1 | ✅ 2 | ✅ 1 |
| T10 Gazette IAS appointment (keyword: `Dilsukh`) | ❌ 21 | ✅ 1 | ✅ 2 | ✅ 4 |

**Findings**
- Hybrid fixed every text-answerable dense failure: T1 (15 → 3) and T10 (21 → 2) moved into the top 5
- BM25 alone ranked the answer #1 in every case above: exact words ("Indian Administrative", "391,035", "Self-Attention") are what these questions hinge on
- Dense is wording-sensitive, BM25 is not: using Table 1's exact column name ("per layer") moved dense from 15 to 4; BM25 stayed #1
- **The reranker ranked tables and noisy OCR lower than BM25** (T1: 1 → 4, T10: 1 → 4), though always within the top 5. Cross-encoders are trained mostly on prose; Markdown pipes, formulas and OCR errors (`plcascd`, `Scrvice`) lower their confidence. Rerank is not automatically better.

**Regression check (tests dense already passed)**

| Test | Dense | BM25 | Hybrid | Rerank |
|---|---|---|---|---|
| T2 employees (`full-time equivalent`) | ✅ 1 | ❌ 6 | ✅ 1 | ✅ 1 |
| T3 Note 2 (`performance obligations`) | ✅ 1 | ✅ 1 | ✅ 1 | ✅ 1 |
| T4 Item 1A (`could have a material adverse effect`) | ✅ 1 | ✅ 1 | ✅ 1 | ✅ 1 |
| T5 BLEU (`28.4`) | ✅ 1 | ✅ 1 | ✅ 1 | ✅ 1 |
| T7 Services 2025 (`109,158`) | ✅ 2 | ❌ 6 | ✅ 2 | ✅ 2 |
| T12 EPA clean scan (`public website`) | ✅ 1 | ✅ 1 | ✅ 1 | ✅ 2 |
| T13 net sales 2026 (`478,834`) | ✅ 2 | ✅ 2 | ✅ 4 | ✅ 1 |

**Summary over all 11 text-answerable tests (T1–T7, T6b, T10, T12, T13)**

| Mode | In top 5 | Mean rank | Misses |
|---|---|---|---|
| Dense | 9/11 | | T1, T10 (exact words) |
| BM25 | 9/11 | | T2, T7 (meaning: "employees" vs "full-time equivalent") |
| Hybrid | **11/11** | 1.73 | none |
| Rerank | **11/11** | 1.73 | none |

**Key finding: dense and BM25 fail on different questions; hybrid fails on none.**

**(Superseded by the 55-question evaluation below.)** **Finding: the reranker gave no measurable gain over hybrid** on this set (same 11/11, same mean rank 1.73): it helped T13 (4 → 1) but lowered tables/OCR (T1 3 → 4, T10 2 → 4), and costs a ~1 GB model plus extra latency. Only 11 questions, so not conclusive → default set to hybrid; rerank kept as an option and re-tested in the Step 4 evaluation.

**Lesson: `--find` keywords must be unique to the answer chunk.** "Complexity per Layer" first gave dense a false rank 1, because Section 4's text also says "the total computational complexity per layer". A row label only in Table 1 ("Self-Attention (restricted)") gave the true rank (15).

---

## Step 2.5: Visual retrieval (ColPali family)

**Setup**
- Model: `vidore/colSmol-256M` (ColPali-style late interaction, via `colpali-engine`), on the Mac GPU (MPS) in float32
- Only pages where text fails are indexed: scanned pages + pages with figures (IMF: 22 of 46 pages); digital text pages are left to hybrid search
- Each page image → 1,139 patch vectors (128-d) → one Qdrant point in collection `docai_pages`, scored with MaxSim
- Indexing speed: ~7 s per page on a MacBook Air (22 IMF pages in 158 s)
- Answering: `ask.py --images N` sends the top-N page images to Gemini together with the text chunks

**Engineering findings**
- ColQwen2.5 (3B, ~7.5 GB) and ColSmol-256M both crashed (segfault) while loading their adapter straight onto MPS in bfloat16. Fix: load in float32 on CPU first, then move to MPS. (Not a memory problem: the 500 MB model crashed the same way.)
- ColSmol-256M is used locally; ColQwen2.5 remains an option on a CUDA GPU (Kaggle/Colab)

**First results (IMF charts, page 9)**

| Test | Dense | BM25 | Hybrid | Rerank | Visual |
|---|---|---|---|---|---|
| T8 commodity peak | ❌ 17 | ✅ 2 | ✅ 5 | ✅ 1 | ❌ 6 |
| T9 structural balance 2025 | ❌ 13 | ✅ 1 | ✅ 2 | ✅ 1 | ✅ 1 |

- Text modes find page 9 through its caption and panel titles, but those chunks do not contain the chart values → for charts, a text hit = page found, not answer found

**First end-to-end chart answer (T9, `--images 2`)**
- Visual search put page 9 at #1 (score 17.4); Gemini read Figure 1.9 panel 3
- ✅ "Emerging market and developing economies, approximately −0.3 percentage points [p. 9]"
- ❌ also claimed advanced economies were slightly negative in 2025; their 2025 bar is slightly positive (~+0.01), the 2026 bar is slightly negative
- **Finding:** the VLM reads main values correctly but confuses near-zero bars in adjacent years (144 DPI full-page image). Retrieval was correct, the answer only half correct → evaluate final answers, not only retrieval

---

## Retrieval evaluation (55 questions, 6 documents)

**Setup**
- `backend/eval/questions.jsonl`: 55 questions over 6 documents (Apple 10-K, Attention, BERT, IMF WEO ch. 1, Gazette 1960 faded scan, EPA letter clean scan); 9 categories
- Each question: answer, a keyword unique to the answer chunk, and the answer page; answers drafted from the PDF text/charts and checked against the page
- `backend/eval/run_retrieval.py`: runs every question in every mode (models loaded once), validates keywords against the chunks first, reports hit@5 and MRR per category, saves `results_retrieval.csv`
- Judging: text modes by keyword (charts/T11: by page = "page found", not "answer found"); visual by page, only for chart/scan questions (only those pages are in the visual index)

**Results**

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
| **ALL** | **89% / 0.71** | **85% / 0.73** | **95% / 0.77** | **96% / 0.86** | **94% / 0.87** (18) |

(hit@5 / MRR)

**Findings**
- **The reranker helps, which reverses the 11-question result.** On 11 questions rerank tied hybrid (mean rank 1.73); on 55 it is clearly best (MRR 0.77 → 0.86), mostly by ranking the answer higher: tables 0.30 → 0.67 (hit@5 80% → 100%; Q03 Greater China found only by rerank), numbers 0.79 → 0.95, synonyms (Q22: hybrid 8 → rerank 1). A small test set gave a wrong conclusion.
- The reranker is still weaker on noisy text: faded scans 88% vs hybrid 100% (T11 rank 6), formula tables MRR 0.75 vs BM25 1.00
- Dense vs BM25 complementarity confirmed at scale: BM25 fails synonyms (57%) and tables (40%); dense fails formula tables (67%); hybrid 95%
- **Q12 fails in every mode** ("How many attention heads?" → `h = 8` at rank 8–10): many chunks discuss "heads" (multi-head attention, Table 3 head-count variations), while the `h = 8` sentence sits in an equation-heavy chunk. Open issue.
- Charts: text modes found the chart page as often as visual search (rerank 100%, T8 at #1 vs visual #6), but text chunks lack chart values; visual found pages whose text OCR broke (T11: visual 2 vs rerank 6)

**Decisions**
- Default text mode → **rerank** (justified by MRR 0.86 at ~1 GB model + extra latency; hybrid kept as the faster option)
- Page images for answers → **combined**: top visual page + pages of the top rerank chunks that are in the visual index (figure/scanned pages), capped at 3 images (`ask.py --images`, default 3)

**First answers with the combined page images**
- T8 ("Which commodity price index peaks highest and when?"): page 9 sent via the text hit. Gemini answered **Precious metals** (~2025:Q4–2026:Q1). **This exposed an error in our ground truth:** Precious metals is on the right-hand scale (0–250); we had compared lines visually and picked Energy (~150 on the left scale). Precious metals' solid line peaks ~185–190 → Gemini picked the right series but overestimated the value (225–230). T8 answer corrected.
- T11 ("Who was granted earned leave for 17 days?"): ❌ "couldn't find". Page 4 was not sent: the top rerank hit was low-confidence (score −3.6) but its pages (8, 9) took 2 of 3 image slots and pushed out visual #2 (page 4).
- Fix: text-guided pages only from rerank hits with score > 0 (T8 top hit 0.159 → used; T11 −3.6 → skipped). Threshold chosen from two examples → confirm in the Step 4 answer evaluation. (In hybrid mode the RRF scores are always > 0, so the filter does not apply there.)

- T11 after the fix: ✅ "Shri S. S. Saith, a permanent Librarian in the Ministry of External Affairs, was granted earned leave for 17 days from January 12, 1960, to January 28, 1960 [p. 4]." All 5 text hits scored below 0 (−3.6 to −4.9) → their pages skipped → images = visual pages 3, 4, 9. Gemini read "17 days" and the dates from the faded page image; OCR had produced "ear   s       ty" and "fanuary". (Jan 12–28 inclusive = 17 days.)
- **Key result: an answer no text-based RAG can give.** The information does not exist in the extracted text; only page-image retrieval + a VLM recovers it.

**Caveats:** questions written by one person (drafted from the PDFs); judged by keyword/page, not by the final answer; 55 questions is still small → answer-level evaluation in Step 4

---

## Step 3: Backend API + frontend

**Backend (FastAPI, `backend/api/`)**
- All models loaded once at startup (bge-small, BM25, bge-reranker-base, ColSmol, Gemini client) + one shared local-Qdrant connection with a lock
- `POST /documents` (upload → background worker: parse → chunk → index → visual index, one PDF at a time; status in `status.json`; duplicates detected by file hash), `GET /documents`, `GET /documents/{id}`, `/status`, `/file`, `/pages/{n}`
- `POST /chat` streams Server-Sent Events: `sources` (chunks with pages + bounding boxes, page images) → `token` … → `done` (timing)
- Gemini: 30 s timeout per call, retry on 429/500/503/timeout, fallback model
- Visual search is skipped for documents with no page images (~1–2 s saved)

**Latency (one question, 10-K)**
- 184.6 s with `gemini-3.8-flash` as main model (overloaded: repeated 503s and hanging calls) → **11.1 s** after switching main to `gemini-3.6-flash` (retrieval 4.7 s, first token at 11.0 s)
- 11-page handwritten upload: ready in 63 s (parse 30 s, visual 33 s ≈ 3 s/page); 51-page handwritten upload: parse 131 s + ~3 s/page visual

**Frontend (Next.js 16, `frontend/`)**
- Documents column: drag-and-drop upload, live progress (polls status every 1.5 s)
- PDF viewer (react-pdf / PDF.js): lazy page rendering (only pages near the view are drawn, max 40 kept), zoom, fallback to page images if the PDF file is missing
- Chat: streamed answers (fetch + stream reader; EventSource only supports GET), Markdown, sources list, Stop, mode (rerank/hybrid) and images toggle
- **Clickable citations:** `[p. 8]`, `[p. 3, 4]`, `[p. 3, p. 4]`, `[pp. 3-5]`, `[page 12]`, `[pages 3 and 5]`, `[p. 3; p. 9]` → one button per page; click → viewer scrolls to the page and highlights the source chunk's bounding boxes (PDF points scaled by rendered width / page width)

**Frontend bugs found and fixed (good interview stories)**
1. **Safari crash in PDF.js text layer:** PDF.js uses `for await (… of readableStream)`; Safari's ReadableStream is not async-iterable → polyfill + fallback that switches the text layer off if it still fails
2. **Whole window scrolled instead of the viewer; reopened at "page 41":** a CSS grid row grows to fit its content, so the middle column became as tall as the whole PDF and Safari restored the old window scroll on reload → `grid-rows-[minmax(0,1fr)]` + `overflow-hidden`
3. **Viewer jumped back to page 1 while scrolling:** logs showed the viewer unmounting/remounting in a loop. Newly visible pages suspended while PDF.js loaded them, and the nearest Suspense boundary was the `dynamic()` wrapper around the whole viewer → React hid the viewer (scroll lost) and re-ran its effects. Fix: a Suspense boundary per page + adding pages inside `startTransition`
4. **Scroll jank:** state updated on every scroll event → now once per animation frame, only when the visible page set or current page changes; pages memoized
5. **Citation formats:** Gemini writes citations in many shapes; the parser accepts all of the above and leaves non-citations (`[Table 2]`) alone

**VLM limitation seen again:** chart peak values overestimated (Energy peak ~305 read as 330–350 on IMF Fig. 1.SF.1; earlier 187 read as 225–230). Series and timing were right.

---

## Step 4.1: Answer evaluation (final answers, not just retrieval)

**Setup (`backend/eval/run_answers.py`)**
- Same pipeline as the app: rerank (top 5 chunks) + up to 3 page images → answer
- Answer model `gemini-3.6-flash`, no fallback (one model for every answer)
- Judge: a **different** model, `gemini-3.8-flash`, temperature 0, JSON verdict: correct / partial / wrong / abstained + short reason, against the reference answer
- Model-free citation checks: **evidence** (a cited page holds the answer: keyword in a retrieved chunk on that page, or the expected page for charts/scans) and **grounded** (every cited page was given to the model)
- Resumable: results saved after each question (`eval/answers.jsonl`); summary → `eval/results_answers.csv`

**Results (55 questions, 6 documents)**

| Category (n) | Correct | Partial | Wrong | Abstained | Evidence | Grounded | Avg s |
|---|---|---|---|---|---|---|---|
| chart (7) | 86% | 14% | 0% | 0% | 100% | 100% | 12.1 |
| exact-term (6) | 83% | 0% | 0% | 17% | 100% | 100% | 7.0 |
| formula-table (3) | 100% | 0% | 0% | 0% | 100% | 100% | 7.5 |
| number (11) | 100% | 0% | 0% | 0% | 100% | 100% | 5.3 |
| prose (4) | 100% | 0% | 0% | 0% | 100% | 100% | 4.5 |
| scan-clean (4) | 100% | 0% | 0% | 0% | 100% | 100% | 5.9 |
| scan-faded (8) | 100% | 0% | 0% | 0% | 100% | 100% | 9.1 |
| synonym (7) | 100% | 0% | 0% | 0% | 100% | 100% | 5.2 |
| table (5) | 100% | 0% | 0% | 0% | 100% | 100% | 4.9 |
| **ALL (55)** | **96%** | **2%** | **0%** | **2%** | **100%** | **100%** | **7.0** |

**Findings**
- **No wrong answers and no hallucinated citations** (100% grounded): when the answer was not retrieved, the system abstained (Q12, "How many attention heads?", the one question retrieval missed in every mode)
- **Faded scans 100% correct**, including T11 ("17 days" destroyed by OCR, recovered from the page image)
- **Only weakness: exact values read from charts** (T9 partial: one group right, a near-zero bar misread)
- **Run-to-run variation:** T8 was partial in run 1 and correct in run 2 (temperature 0.2; the chart is read slightly differently each time). Chart answers fluctuate between partial and correct; a single run per question is not fully stable

**Bug found by the evaluation: citation format on scanned documents**
- Run 1 showed "evidence ✗" on 5 correct scan answers (T10, T12, Q32, Q34, Q38). Cause: with page images in the prompt, the model copied the source labels and wrote `[Source 4, p. 2, 3; Page image, p. 3]`; both parsers only read brackets starting with "p.", so they found **no** citations, and in the app those answers had no clickable citations
- Fix: the prompt asks for `[p. N]` only, and both parsers (app + evaluation) read `p. N` anywhere inside a bracket
- Run 2: evidence 91% → **100%**; correct 95% → 96%; partial 4% → 2%; avg time 9.3 s → 7.0 s

**Human spot-check**
- 15 random answers graded by hand without seeing the judge's verdict: **15/15 agreement** with the judge
- All 15 sampled answers were "correct" (expected when 96% are), so this validates the judge's "correct" verdicts only; the non-correct verdicts (T9 partial, Q12 abstained) checked separately

**Caveats**
- Questions were written by one person from the same documents, mostly single-fact lookups → an easy test; real questions are vaguer
- Judge and answerer are different models but the same family (Gemini) → human spot-check below

## Where dense retrieval fails: test questions

Run each question in every retrieval mode as it gets built, and fill in the result columns
(✅ = right chunk/page in the top 5, ❌ = missed). "Expected" marked *verify* still needs checking against the PDF.

| ID | Failure type | Question | Doc | Expected answer / page | Dense | Hybrid | +Rerank | ColPali |
|---|---|---|---|---|---|---|---|---|
| T1 | Formula table | What is the complexity per layer of self-attention? | Attention | O(n²·d), Table 1, p. 6 | ❌ rank 15 | ✅ 3 | ✅ 4 | |
| T2 | Tiny chunks | How many employees does Apple have? | 10-K | Human Capital, p. 8 | ✅ #1, but #2/#3 noise | | | |
| T3 | Exact term | What does Note 2 say about how revenue is recognized? | 10-K | Note 2 – Revenue, p. 41 | ✅ #1 (term is in heading) | | | |
| T4 | Exact term | What are the risk factors in Item 1A? | 10-K | Item 1A, p. 8 | ✅ #1, 0.828 (term is in heading) | | | |
| T5 | Exact term | What BLEU score did the big Transformer get on English-German? | Attention | 28.4, Table 2, p. 8 | ✅ #1 text (6.1), Table 2 #3 | | | |
| T6 | Number | Which year had total net sales of 391,035? | 10-K | 2024, p. 35 / p. 29 | ⚠️ ✅ weak (net-sales tables found by topic, not by the number; scores 0.62–0.65) | | | |
| T7 | Number | What were Services net sales in 2025? | 10-K | 109,158, p. 29 / p. 35 | ✅ table #2 (p. 29) | | | |
| T8 | Chart | Which commodity price index peaks highest and when? | IMF | Energy, ~2026:Q4, index ~150 (Fig 1.9 panel 1, p. 9; chart only) | ❌ p. 9 not in top 5 (found Commodity Special Feature text, pp. 31–39) | | | |
| T9 | Chart | Which country group had a negative structural primary balance change in 2025? | IMF | Emerging market and developing economies, ~−0.3 pp (Fig 1.9 panel 3, p. 9; chart only) | ❌ p. 9 not in top 5 (found Fig 1.8 Fiscal Policy caption p. 8, trade text, annex table ×3) | | | |
| T10 | Faded scan | Who was appointed to the Indian Administrative Service? | Gazette | Shri Dilsukh Ram, p. 3 | ❌ rank 21 (#1 was the Central Information Service) | ✅ 2 | ✅ 4 | |
| T11 | Faded scan | Who was granted earned leave for 17 days? | Gazette | Shri S. S. Saith, p. 4 | ❌ not in top 30: OCR destroyed "earned leave for 17 days" → text-only methods cannot answer; ColPali-only test | | | |
| T6b | Number only | 391,035 | 10-K | p. 29 / p. 35 tables | ⚠️ weak: p. 35 at #3 by chance, p. 29 missing, flat scores 0.60–0.64 | | | |
| T13 | Wording sensitivity | What were Apple's total net sales in 2026? | 10-K | $478,834M, p. 35 / p. 42 | ❌ before rule 6 → ✅ after (table #2) | | | |
| T12 | Clean scan (control) | What does the EPA ask states to post on their websites? | EPA letter | LCR sampling protocols and guidance, p. 1 | ✅ rank 1 | | | |

**Finding (T3–T7):** dense passed more than expected because the exact terms ("Note 2", "Item 1A") were also in the chunk headings, another benefit of prepending headings. These questions are too easy to separate dense from hybrid → added harder variants where the exact word/number is the only clue (T6b, plus a body-only term T3b to pick).

**Finding (T6b):** with only the number as the query, dense scores were nearly flat (0.60–0.64): it matched "number-heavy chunks", not the number. Clearest case for BM25 exact matching.

**Finding (T6b):** a second kind of tiny chunk survived rule 7: heading + a lone year ("Note 14 … 2025", p. 55; p. 43), where Docling read a table's column label as text. They sit between two tables, so rule 7 cannot merge them (it never merges into tables).

**Finding (scans): clean scan ✅ vs. faded scan ❌.** Dense found the EPA answer at rank 1, but the Gazette answers ranked 21 (T10) and >30 (T11). Gazette notifications all look alike ("Shri X … appointed / granted leave"), so meaning alone cannot separate them, and OCR errors (`Scrvice`, `carned`) blur the exact words.

**Finding (T11): OCR can destroy the answer itself.** The chunk reads "Shri S. S. Saith a permanent Librarian … has been granted ear   s       ty 1960 to the 28th fanuary, 1960"; "earned leave for 17 days" was lost to faded ink. No text-based method (dense, BM25, reranker) can retrieve information that is not in the text → only page-image retrieval (ColPali + VLM) can answer. Strongest motivation for ColPali.

**Finding (charts, T8–T9):** both failed. Dense found related text (commodity section; a different fiscal figure) but not Figure 1.9 on p. 9. Even at rank 1 it would not help: chart values were never turned into text, so chunks hold only captions and surrounding prose. With T11, these are the ColPali-only cases.

**Finding (T9): split-table parts crowd the results.** #3–#5 were three parts of the same Annex Table 1.1.5 with identical scores (0.698), because every part starts with the same heading, intro and header row. One table took 3 of 5 slots.

**How to check rank of the answer chunk (any test):** search with k=30 and find the first hit containing a keyword from the answer (see the snippet used for T10–T12).

**How to run (dense):**
`python ingest/index.py search "<question>" --doc <doc_id>` (index the IMF, Gazette and EPA docs first)

## To improve later
- ~~Merge tiny chunks (< ~30 words) into a neighbouring chunk~~ → done (rule 7)
- ~~Attach a table's intro sentence to the table chunk~~ → done (rule 6)
- ~~Long split-table headers repeated in every part~~ → fixed: Gazette p. 12 (a two-column name list) was parsed as one giant table whose "header row" was full of names; splitting it into 41 parts repeated that header in each, so "Menon" was in 44 of 138 chunks. Now captions/headers are repeated only if ≤ 40 words; otherwise first part only, later parts get "(table continued)".
- Send page images only when likely needed (currently 3 images for every question on documents with visual pages, even when the text chunk already has the answer, e.g. "global growth in 2026" → "3.1 percent"). Candidate rules: question sounds visual (chart/figure/trend), all text hits weak (rerank < 0, as in T11), or high visual score. Test in the Step 4 answer evaluation (accuracy vs. cost/latency).
- Keep only the best-scoring part of a split table in the final results (avoid one table filling several slots)
- Drop text chunks whose body is only a number or year (e.g. "2025" between two tables)

## To evaluate later
- Text-only RAG vs. ColPali on faded scans (Gazette) and chart questions (IMF)
- Dense vs. hybrid + reranker on table questions (e.g., Attention Table 1)
- Chunk size 150 / 300 / 500 on retrieval accuracy (not just structure)
- Later: dense vs. hybrid vs. ColPali vs. GraphRAG vs. agentic on one test set
- Optional: RapidOCR vs. ocrmac on the faded Gazette scan

---

# Step 5: GraphRAG

## The plan (what GraphRAG is here)
Plain search finds chunks that *sound like* the question. It fails when the answer needs two facts in two places and the second one does not sound like the question. GraphRAG builds a map of the document (things and how they are connected) and walks that map from what search found to what it missed.

1. **5.1 Multi-hop test set**: questions that need two facts from two places, plus a baseline with today's search.
2. **5.2 Extraction**: Gemini reads every chunk and returns entities and relationships, each relationship with an exact evidence quote that the code checks.
3. **5.3 Merging + structural edges**: the same entity from many chunks becomes one node; add edges from the document's layout. Store as a NetworkX graph per document; entity vectors in Qdrant (`docai_entities`).
4. **5.4 Graph search**: question → matching entities → walk 1–2 hops → their chunks → reranker → Gemini. Two modes: `graph` and `graph+hybrid` (graph chunks added to the usual reranked chunks).
5. **5.5 Evaluation** against the baseline, plus a hand check of merges.
6. **5.6 "Graph" mode in the UI.**
7. **5.7 (optional)** communities (clusters of entities with summaries) for broad questions.

**Edges come from two sources only:** relationships stated in the text (with an evidence quote) and structural edges from the document layout.

## Step 5.1: Multi-hop test set and baseline

**What a multi-hop question is here:** the answer needs two facts that sit in two different chunks (usually two pages). Each question has `keywords` (one per fact) and `pages`.

**How keywords are used (eval only, never sent to the RAG model):**
- Search check (`run_retrieval.py`): a question counts as found only when **every** keyword is in the top 5. Rank = the rank at which the last fact appears.
- Answer check (`run_answers.py`): "cites evidence" only if the answer cites a page for **every** fact. Correctness is judged by a different model against a hand-written reference answer.
- `--check` validates the questions file: keyword not in any chunk (typo or OCR), keyword in more than 3 chunks (too common), and **one chunk holds every keyword** (not really multi-hop).

**Two kinds of multi-hop question:**
- **Bridge (12):** you only know what to search for second after finding fact 1 ("Which product grew fastest, and *why*?" → iPhone → "Pro models"). This is where GraphRAG should help.
- **Compare (14):** the question names both facts ("What was R&D expense, and how many employees?"). Search can match each half on its own words.

**Challenges and fixes:**
- **M06 was not really multi-hop:** the Attention abstract says "28.4 BLEU ... 3.5 days" in one sentence. `--check` caught it; replaced with "parameters (Table 3, p. 9) + training time (p. 7)".
- **The second batch was too easy:** 9 of 10 new questions were found by rerank, because they were two questions joined with "and". This led to splitting multi-hop into bridge and compare, and writing 5 more real bridge questions (M22–M26).
- **The summary grouped old answers under the old label:** each saved answer keeps the category it had when it ran. Fixed: the summary reads categories from `questions.jsonl`.
- No Gazette or handwritten questions in the new batches: OCR text is too unreliable for exact keywords.
- Keyword rule: short, exact text from the answer's location, unique (1–3 chunks); "intensify for memory" is in 4 chunks because Apple repeats it in risk factors; kept, since any of them answers that half.

**Baseline (rerank, before any graph):**

| | n | Search hit@5 | Answers correct | Partial | Abstain | Wrong |
|---|---|---|---|---|---|---|
| **Bridge** | 12 | **42%** | **50%** | 33% | 17% | 0% |
| Compare | 14 | 79% | 86% | 14% | 0% | 0% |
| Single-fact | 55 | 96% | ~96% | | | |

**Findings:**
- Fact 1 is almost always found; **fact 2 is what is missing** (often rank 6–9, or not in the top 30 when it does not sound like the question: M01 "why" vs. "Pro models", M09 "hardware" vs. "16 Cloud TPUs").
- **Wrong = 0%**: when a fact is missing, the model leaves it out or says it cannot find it. So more evidence should turn partials directly into corrects.
- Answers beat search (50% vs 42%) because page images sometimes cover the missing fact (M10: oil price).
- On multi-hop, hybrid was not better than BM25 (fusion pushed second facts down); rerank was best.

**Targets for GraphRAG:** bridge answers up from 50%; compare stays at ~86%; single-fact must not drop from ~96%.

Baseline saved as `eval/answers_baseline_multihop.jsonl`, `eval/results_retrieval_baseline.csv`, `eval/results_answers_baseline.csv`.

## Step 5.2: Entity and relationship extraction (`backend/graph/extract.py`)

**What happens:** each chunk (≥ 8 words) is sent to Gemini (`gemini-3.6-flash`, temperature 0, JSON output). Results are saved per chunk in `data/processed/<doc_id>/graph/extractions.jsonl`; the run resumes where it stopped.

**Entity rules (prompt):**
- Important named things only, name as written, short.
- One of 8 fixed types: Organization, Person, Product, Place, Date, Metric, Concept, Event (generic, so it works for any PDF).
- Metric = a named measurement, with the number in the description ("iPhone net sales 2026 → $254,098M, up 21%").
- Description: one sentence, facts from this chunk only.
- Skip generic words ("the Company", "this section") and labels of document parts ("Table 3", "Section 3.2"): they are not entities.

**Relationship rules (prompt):**
- Only links the text states; no outside knowledge.
- Source and target must be from the entity list.
- Description (how they are related), strength 1–10, and an evidence quote of 5–25 words copied word for word. No quote → no relationship.

**Checks in code (the model cannot get past these):**
- Evidence quote must be in the chunk (ignoring case, spacing, table pipes, "$ 254" vs "$254"); at least 8 characters. Otherwise dropped as invented.
- Both ends must be entities of that chunk; no self-links.
- Unknown type → Concept; duplicate names in a chunk kept once; strength clamped to 1–10.

So every edge can be traced to an exact sentence in the PDF.

**Description vs. evidence:** the description is the model's own words (used for search and merging; could be slightly off); the evidence is an exact quote (proof, checked). At answer time Gemini still gets the original chunks, not just descriptions.

**First test (Attention paper, 5 chunks):** 42 entities, 24 relationships, **100% kept** (0 bad evidence, 0 unknown entities). Quality good ("Transformer → Self-attention", "Transformer → Multi-Head Attention"; ByteNet / ConvS2S / Extended Neural GPU described in comparison). About 11 s per chunk: fine for 46 chunks (~8 min), slow for the Apple 10-K (hundreds of chunks); parallel calls may be needed. Still to check: table chunks (numbers in rows).