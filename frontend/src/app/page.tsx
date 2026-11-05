// Landing page (/). The workspace lives at /app.
// Every example on this page is a real answer from the project's own tests.

import Link from "next/link";
import Logo from "@/components/Logo";
import ThemeToggle from "@/components/ThemeToggle";

const REPO = "https://github.com/ayeshamemon/docai";

const HARD_CASES = [
  {
    kind: "Financial tables",
    doc: "Apple 10-K, 123 pages",
    q: "What were Apple's total net sales in 2026?",
    a: "$478,834 million",
    page: 29,
    note: "Table intro sentences stay attached to their tables, so the right table is found even when the question names a year.",
  },
  {
    kind: "Charts",
    doc: "IMF World Economic Outlook, Figure 1.9",
    q: "Which country group had a negative structural primary balance change in 2025?",
    a: "Emerging market and developing economies, about −0.3 points, read from the chart",
    page: 9,
    note: "The value exists only as a bar. The page image is retrieved and read by a vision model.",
  },
  {
    kind: "Faded scans",
    doc: "Gazette of India, March 1960",
    q: "Who was granted earned leave for 17 days?",
    a: "Shri S. S. Saith, from 12 to 28 January 1960",
    page: 4,
    note: "OCR turned this line into “ear   s   ty”. No text search can find it; reading the page image can.",
  },
  {
    kind: "Handwritten notes",
    doc: "Class notes, 11 scanned pages",
    q: "What is RAG and why is it used?",
    a: "Retrieval-Augmented Generation: retrieval, augmentation, generation, with reasons drawn from the notes",
    page: 3,
    note: "Every page is treated as a scan: OCR for the text, page images for everything else.",
  },
];

const STEPS = [
  ["Parse", "Docling reads layout, reading order and tables; scanned pages get OCR; every page is saved as an image."],
  ["Chunk", "Structure-aware chunks: headings stay with their text, tables stay whole and keep their headers."],
  ["Index", "Each chunk gets a meaning vector and a keyword vector; chart and scanned pages get patch-level image vectors."],
  ["Retrieve", "Hybrid search and a reranker find the passages; visual search finds the pages where text fails."],
  ["Answer", "Gemini answers only from those sources and cites pages. Click a citation to see the passage highlighted."],
];

const RESULTS = [
  ["Dense (meaning)", "89%", "0.71"],
  ["BM25 (keywords)", "85%", "0.73"],
  ["Hybrid", "95%", "0.77"],
  ["Hybrid + reranker", "96%", "0.86"],
];

export default function Landing() {
  return (
    <div className="min-h-full bg-paper text-ink">
      {/* header */}
      <header className="border-b border-rule">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-5 py-3">
          <Logo />
          <nav className="flex items-center gap-1 text-sm text-ink-soft sm:gap-3">
            <a href="#how" className="hidden rounded px-2 py-1 hover:text-ink sm:inline">How it works</a>
            <a href="#results" className="hidden rounded px-2 py-1 hover:text-ink sm:inline">Results</a>
            <a href={REPO} target="_blank" rel="noreferrer" className="rounded px-2 py-1 hover:text-ink">GitHub</a>
            <ThemeToggle className="text-ink-soft hover:bg-desk hover:text-ink" />
            <Link href="/app"
              className="rounded-md bg-blueprint px-3 py-1.5 font-medium text-paper hover:opacity-90">
              Open the app
            </Link>
          </nav>
        </div>
      </header>

      {/* hero: the idea, shown rather than told */}
      <section className="mx-auto grid max-w-6xl items-center gap-14 px-5 py-16 md:grid-cols-[1fr_1.05fr] md:py-24">
        <div>
          <h1 className="text-[2.6rem] font-semibold leading-[1.08] tracking-tight sm:text-5xl">
            Ask a PDF anything. See the exact passage behind every answer.
          </h1>
          <p className="mt-6 max-w-[34rem] text-lg leading-relaxed text-ink-soft">
            Upload reports, research papers, scanned letters or handwritten notes. DocAI reads
            text, tables, charts and faded scans, answers with page citations, and highlights the
            evidence on the page.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Link href="/app" className="rounded-md bg-blueprint px-5 py-2.5 font-medium text-paper hover:opacity-90">
              Open the app
            </Link>
            <a href="#how" className="rounded-md border border-rule px-5 py-2.5 font-medium text-ink hover:bg-desk">
              How it works
            </a>
          </div>
        </div>

        <Demo />
      </section>

      {/* hard documents */}
      <section className="border-t border-rule bg-desk/60">
        <div className="mx-auto max-w-6xl px-5 py-16">
          <h2 className="max-w-2xl text-2xl font-semibold tracking-tight sm:text-3xl">
            Built for the documents that break chat-with-PDF tools
          </h2>
          <p className="mt-3 max-w-2xl text-ink-soft">
            Each example below is a real answer from the project’s test set.
          </p>

          <div className="mt-10 divide-y divide-rule border-y border-rule">
            {HARD_CASES.map((c) => (
              <article key={c.kind} className="grid gap-4 py-7 md:grid-cols-[13rem_1fr]">
                <div>
                  <h3 className="font-semibold">{c.kind}</h3>
                  <p className="mt-1 text-sm text-ink-faint">{c.doc}</p>
                </div>
                <div>
                  <p className="text-sm text-ink-soft">{c.q}</p>
                  <p className="mt-1.5 font-serif text-xl leading-snug">
                    {c.a} <span className="cite align-middle">p.&nbsp;{c.page}</span>
                  </p>
                  <p className="mt-2 max-w-2xl text-sm text-ink-faint">{c.note}</p>
                </div>
              </article>
            ))}
          </div>
        </div>
      </section>

      {/* how it works: a real sequence, so it is numbered */}
      <section id="how" className="mx-auto max-w-6xl scroll-mt-16 px-5 py-16">
        <h2 className="text-2xl font-semibold tracking-tight sm:text-3xl">How it works</h2>
        <ol className="mt-10 grid gap-8 md:grid-cols-5 md:gap-6">
          {STEPS.map(([title, text], i) => (
            <li key={title} className="border-t-2 border-ink pt-4">
              <p className="text-sm text-ink-faint">Step {i + 1}</p>
              <h3 className="mt-1 font-semibold">{title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-ink-soft">{text}</p>
            </li>
          ))}
        </ol>
      </section>

      {/* results */}
      <section id="results" className="scroll-mt-16 border-t border-rule">
        <div className="mx-auto grid max-w-6xl gap-10 px-5 py-16 md:grid-cols-[1fr_1.2fr]">
          <div>
            <h2 className="text-2xl font-semibold tracking-tight sm:text-3xl">Measured, not assumed</h2>
            <p className="mt-4 leading-relaxed text-ink-soft">
              55 questions over 6 documents: financial tables, formula-heavy tables, synonyms,
              exact numbers, charts, clean scans and a 1960 faded scan. For each retrieval mode:
              how often the passage with the answer is among the five sources the model reads,
              and how high it ranks.
            </p>
            <p className="mt-4 text-sm leading-relaxed text-ink-faint">
              Visual search found the answer page for 94% of the 18 chart and scan questions.
            </p>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="text-ink-faint">
                <tr className="border-b border-rule">
                  <th className="py-2 pr-4 font-medium">Retrieval mode</th>
                  <th className="py-2 pr-4 font-medium">Answer in top 5</th>
                  <th className="py-2 font-medium">Mean reciprocal rank</th>
                </tr>
              </thead>
              <tbody>
                {RESULTS.map(([mode, hit, mrr], i) => {
                  const best = i === RESULTS.length - 1;
                  return (
                    <tr key={mode} className={`border-b border-rule ${best ? "font-semibold" : ""}`}>
                      <td className="py-3 pr-4">{mode}</td>
                      <td className="py-3 pr-4 tabular-nums">{hit}</td>
                      <td className="py-3 tabular-nums">{mrr}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* closing */}
      <section className="border-t border-rule bg-side text-side-text">
        <div className="mx-auto flex max-w-6xl flex-col items-start justify-between gap-6 px-5 py-14 sm:flex-row sm:items-center">
          <p className="max-w-xl text-2xl font-semibold leading-snug tracking-tight">
            Try it on your own PDF.
          </p>
          <Link href="/app" className="rounded-md bg-marker px-5 py-2.5 font-medium text-[#1f2a37] hover:opacity-90">
            Open the app
          </Link>
        </div>
      </section>

      <footer className="border-t border-rule">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-2 px-5 py-6 text-sm text-ink-faint">
          <span>Built by Ayesha Memon</span>
          <a href={REPO} target="_blank" rel="noreferrer" className="hover:text-ink">Source on GitHub</a>
        </div>
      </footer>
    </div>
  );
}

/** One orchestrated moment: question → marker sweeps the answering line → cited answer. */
function Demo() {
  return (
    <div className="relative mx-2 my-6 sm:mx-6">
      <div className="pdf-page rounded-sm px-7 pb-10 pt-8 text-[#1f2a37]">
        <p className="text-[11px] text-[#8794a3]">Apple Inc. Form 10-K, page 8</p>
        <p className="mt-3 text-sm font-semibold">Human Capital</p>
        <div className="mt-2 space-y-2 font-serif text-[15px] leading-relaxed text-[#4b5867]">
          <p>
            The Company believes that its people play an important role in its success, and
            strives to attract, develop and retain the best talent.
          </p>
          <p>
            <span className="marker-sweep text-[#1f2a37]">
              As of September 26, 2026, the Company had approximately 167,000 full-time
              equivalent employees.
            </span>
          </p>
          <p>The Company believes that compensation should be competitive and equitable.</p>
        </div>
      </div>

      {/* the question and the answer, layered over the page */}
      <div className="appear absolute -left-2 -top-5 max-w-[80%] rounded-lg bg-ink px-3 py-2 text-sm text-paper shadow-lg sm:-left-6"
           style={{ animationDelay: "0.3s" }}>
        How many employees does Apple have?
      </div>
      <div className="appear absolute -bottom-7 -right-2 max-w-[85%] rounded-lg border border-rule bg-paper px-4 py-3 shadow-lg sm:-right-6"
           style={{ animationDelay: "2.2s" }}>
        <p className="font-serif text-[15px] leading-snug">
          Approximately 167,000 full-time equivalent employees{" "}
          <span className="cite">p.&nbsp;8</span>
        </p>
      </div>
    </div>
  );
}
