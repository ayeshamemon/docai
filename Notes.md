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
- Keep only the best-scoring part of a split table in the final results (avoid one table filling several slots)
- Drop text chunks whose body is only a number or year (e.g. "2025" between two tables)

## To evaluate later
- Text-only RAG vs. ColPali on faded scans (Gazette) and chart questions (IMF)
- Dense vs. hybrid + reranker on table questions (e.g., Attention Table 1)
- Chunk size 150 / 300 / 500 on retrieval accuracy (not just structure)
- Later: dense vs. hybrid vs. ColPali vs. GraphRAG vs. agentic on one test set
- Optional: RapidOCR vs. ocrmac on the faded Gazette scan