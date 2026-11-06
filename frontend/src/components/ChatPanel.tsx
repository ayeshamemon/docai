"use client";

// Right column: ask questions about the open document; answers stream in word by word.

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { Box, Mode, Source, Timing, pageImageUrl, streamChat } from "@/lib/api";

interface Message {
  role: "user" | "assistant";
  text: string;
  question?: string;          // assistant: the question it answers (for "Ask again")
  sources?: Source[];
  timing?: Timing;
  error?: string;
  streaming?: boolean;
}

/** Called when a citation or a source is clicked: jump to `page` and highlight `boxes`. */
export type CiteHandler = (page: number, boxes: Box[]) => void;

interface Props {
  docId: string | null;
  docName?: string;
  onCite?: CiteHandler;
}

// Citations: any [...] bracket; inside it, every number list after p./pp./page counts.
// [p. 8]  [p. 3, 4]  [p. 3, p. 4]  [pp. 3-5]  [page 12]  [pages 3 and 5]  [p. 3; p. 9]
// and labelled forms like [Source 4, p. 2, 3; Page image, p. 3]  →  one chip per page.
// Brackets without p./page ("[Table 2]", "[1]") and real Markdown links stay as they are.
// (The evaluation in backend/eval/run_answers.py parses citations with the same rules.)
const BRACKET = /\[([^\]\n]{1,120})\]/g;
const PAGE_LIST = /(?:\bpp?\.|\bpages?\b)\s*(\d+(?:\s*(?:[–-]|,|and|&)\s*\d+)*)/gi;

function bracketPages(inner: string): number[] {
  const pages: number[] = [];
  for (const m of inner.matchAll(PAGE_LIST)) {
    for (const part of m[1].split(/,|\band\b|&/i)) {
      const nums = (part.match(/\d+/g) ?? []).map(Number);
      if (nums.length === 2 && nums[1] - nums[0] >= 0 && nums[1] - nums[0] < 20) {
        for (let p = nums[0]; p <= nums[1]; p++) pages.push(p);
      } else if (nums.length) {
        pages.push(nums[0]);
      }
    }
  }
  return [...new Set(pages)];
}

/** Turn citations into Markdown links (#cite-8) that the renderer shows as buttons. */
function linkCitations(text: string): string {
  return text.replace(BRACKET, (whole: string, inner: string, offset: number, str: string) => {
    if (str[offset + whole.length] === "(") return whole;      // already a Markdown link
    const pages = bracketPages(inner);
    return pages.length
      ? pages.map((p) => `[p.\u00a0${p}](#cite-${p})`).join(" ")
      : whole;
  });
}

/** Boxes of this answer's source chunks that lie on `page`. */
function boxesOnPage(sources: Source[] | undefined, page: number): Box[] {
  return (sources ?? []).flatMap((s) =>
    s.kind === "chunk" ? s.boxes.filter((b) => b.page === page) : []);
}

const EXAMPLES = [
  "Summarize this document in five points",
  "What are the most important numbers?",
  "What do the charts or tables show?",
];

export default function ChatPanel({ docId, docName, onCite }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [mode, setMode] = useState<Mode>("rerank");
  const [images, setImages] = useState(true);
  const [busy, setBusy] = useState(false);
  const abort = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement>(null);

  // Keep the newest message in view while the answer streams.
  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "end" });
  }, [messages]);

  // Stop a running answer if the panel goes away (e.g. another document is opened).
  useEffect(() => () => abort.current?.abort(), []);

  /** Update the last (assistant) message. */
  const patchLast = (fn: (m: Message) => Message) =>
    setMessages((ms) => [...ms.slice(0, -1), fn(ms[ms.length - 1])]);

  function send() {
    const question = input.trim();
    if (!question) return;
    setInput("");
    ask(question);
  }

  async function ask(question: string) {
    if (!docId || busy) return;
    setBusy(true);
    setMessages((ms) => [
      ...ms,
      { role: "user", text: question },
      { role: "assistant", text: "", question, streaming: true },
    ]);

    abort.current = new AbortController();
    await streamChat(
      { question, doc_id: docId, mode, images: images ? 3 : 0 },
      {
        onSources: (sources) => patchLast((m) => ({ ...m, sources })),
        onToken: (t) => patchLast((m) => ({ ...m, text: m.text + t })),
        onDone: (timing) => patchLast((m) => ({ ...m, timing })),
        onError: (error) => patchLast((m) => ({ ...m, error })),
      },
      abort.current.signal,
    );
    patchLast((m) => ({ ...m, streaming: false }));
    setBusy(false);
  }

  function stop() {
    abort.current?.abort();
    patchLast((m) => ({ ...m, streaming: false, error: m.text ? undefined : "Stopped." }));
    setBusy(false);
  }

  if (!docId) {
    return (
      <div className="flex h-full items-center justify-center p-8 text-center text-sm text-ink-faint">
        Open a document from your library to ask questions about it.
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* header with settings */}
      <div className="flex items-center justify-between gap-2 border-b border-rule px-4 py-1.5 text-sm">
        <span className="truncate font-medium text-ink" title={docName}>
          {docName ? docName.replace(/\.pdf$/i, "").replace(/[_-]+/g, " ") : "Chat"}
        </span>
        <div className="flex shrink-0 items-center gap-3 text-xs text-ink-soft">
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value as Mode)}
            className="rounded border border-rule bg-paper px-1 py-0.5 text-ink"
            title="Retrieval mode"
          >
            <option value="rerank">Rerank</option>
            <option value="hybrid">Hybrid</option>
          </select>
          <label className="flex items-center gap-1" title="Also send page images (charts, scans) to Gemini">
            <input type="checkbox" checked={images} onChange={(e) => setImages(e.target.checked)} className="accent-blueprint" />
            Read page images
          </label>
        </div>
      </div>

      {/* messages */}
      <div className="min-h-0 flex-1 space-y-6 overflow-y-auto px-4 py-5">
        {messages.length === 0 && (
          <div className="mt-6">
            <p className="text-sm text-ink-soft">
              Ask anything about this document. Answers cite their pages, and clicking a
              citation shows the exact passage.
            </p>
            <div className="mt-4 flex flex-col items-start gap-2">
              {EXAMPLES.map((q) => (
                <button
                  key={q}
                  type="button"
                  onClick={() => setInput(q)}
                  className="rounded-md border border-rule px-3 py-1.5 text-left text-sm text-ink hover:border-ink-faint hover:bg-desk/50"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m, i) =>
          m.role === "user" ? (
            <div key={i} className="ml-10 rounded-lg bg-ink px-3 py-2 text-sm leading-relaxed text-paper">
              {m.text}
            </div>
          ) : (
            <AssistantMessage key={i} m={m} docId={docId} onCite={onCite}
              onAskAgain={busy ? undefined : () => m.question && ask(m.question)} />
          ),
        )}
        <div ref={bottom} />
      </div>

      {/* input */}
      <div className="border-t border-rule p-3">
        <div className="flex items-end gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
            }}
            rows={2}
            placeholder="Ask a question about this document"
            className="min-h-0 flex-1 resize-none rounded-md border border-rule px-2.5 py-2 text-sm text-ink outline-none placeholder:text-ink-faint focus:border-blueprint"
          />
          {busy ? (
            <button type="button" onClick={stop}
              className="rounded-md border border-rule px-3 py-2 text-sm text-ink hover:bg-desk">
              Stop
            </button>
          ) : (
            <button type="button" onClick={send} disabled={!input.trim()}
              className="rounded-md bg-blueprint px-3 py-2 text-sm font-medium text-paper hover:bg-blueprint/90 disabled:opacity-40">
              Ask
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function AssistantMessage({ m, docId, onCite, onAskAgain }: {
  m: Message; docId: string; onCite?: CiteHandler; onAskAgain?: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(m.text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch { /* clipboard blocked: ignore */ }
  }
  const cite = (page: number) => onCite?.(page, boxesOnPage(m.sources, page));
  const chunks = m.sources?.filter((s) => s.kind === "chunk") ?? [];
  const pages = m.sources?.filter((s) => s.kind === "page") ?? [];

  const phase: Phase = m.text ? "writing" : m.sources ? "reading" : "searching";

  return (
    <div className="text-sm text-ink">
      {m.streaming && !m.error && (
        <Steps phase={phase} chunks={chunks.length} pages={pages.length} />
      )}

      {m.text && (
        <div className={`prose-chat ${m.streaming ? "mt-3" : ""}`}>
          <ReactMarkdown
            components={{
              // citation links (#cite-N) become buttons that jump to the page
              a: ({ href, children }) => {
                const page = href?.startsWith("#cite-") ? parseInt(href.slice(6), 10) : NaN;
                if (Number.isNaN(page)) {
                  return <a href={href} target="_blank" rel="noreferrer" className="underline">{children}</a>;
                }
                return (
                  <button
                    type="button"
                    onClick={() => cite(page)}
                    title={`Show page ${page} in the document`}
                    className="cite"
                  >
                    {children}
                  </button>
                );
              },
            }}
          >
            {linkCitations(m.text)}
          </ReactMarkdown>
        </div>
      )}

      {m.error && <p className="mt-2 rounded bg-danger-bg px-2.5 py-1.5 text-xs text-danger-text">{m.error}</p>}

      {/* page images the model read (charts, scans) */}
      {pages.length > 0 && (
        <div className="mt-3 flex gap-2">
          {pages.map((s) => s.kind === "page" && (
            <button
              key={s.page}
              type="button"
              onClick={() => onCite?.(s.page, [])}
              title={`Page ${s.page}: the model read this page image`}
              className="group relative overflow-hidden rounded border border-rule bg-white"
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={pageImageUrl(docId, s.page)} alt={`Page ${s.page}`} loading="lazy"
                   className="h-[72px] w-[54px] object-cover object-top transition-transform group-hover:scale-105" />
              <span className="absolute inset-x-0 bottom-0 bg-ink/75 py-px text-center text-[10px] text-paper">
                p. {s.page}
              </span>
            </button>
          ))}
        </div>
      )}

      {/* actions + sources */}
      {!m.streaming && (m.text || chunks.length > 0) && (
        <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-ink-faint">
          {chunks.length > 0 && (
            <button type="button" onClick={() => setOpen((o) => !o)} aria-expanded={open} className="hover:text-ink">
              {open ? "Hide" : "Show"} {chunks.length} sources
              {m.timing && ` (${m.timing.total_s} s)`}
            </button>
          )}
          {m.text && (
            <button type="button" onClick={copy} className="hover:text-ink">
              {copied ? "Copied" : "Copy"}
            </button>
          )}
          {onAskAgain && (
            <button type="button" onClick={onAskAgain} className="hover:text-ink">
              Ask again
            </button>
          )}
        </div>
      )}

      {open && (
        <ul className="mt-2 space-y-2 border-l-2 border-marker pl-3 text-xs">
          {chunks.map((s) =>
            s.kind === "chunk" && (
              <li key={s.chunk_id}>
                <button
                  type="button"
                  onClick={() => onCite?.(s.pages[0], s.boxes)}
                  title="Show this source in the document"
                  className="font-medium text-ink underline-offset-2 hover:underline"
                >
                  Page {s.pages.join(", ")}{s.type === "table" ? ", table" : ""}
                </button>
                {s.heading && <span className="text-ink-soft">: {s.heading}</span>}
                <p className="line-clamp-2 text-ink-faint">{s.text.replace(s.heading, "").trim()}</p>
              </li>
            ),
          )}
        </ul>
      )}
    </div>
  );
}

type Phase = "searching" | "reading" | "writing";

/** Live progress while an answer is being made: Search → Read → Write. */
function Steps({ phase, chunks, pages }: { phase: Phase; chunks: number; pages: number }) {
  const order: Phase[] = ["searching", "reading", "writing"];
  const at = order.indexOf(phase);
  const read = `Reading ${chunks} passages${pages ? ` and ${pages} page images` : ""}`;
  const labels: Record<Phase, string> = {
    searching: "Searching the document",
    reading: chunks || pages ? read : "Reading the sources",
    writing: "Writing the answer",
  };
  return (
    <ol className="space-y-1 text-xs" aria-live="polite">
      {order.map((p, i) => (
        <li key={p} className={`flex items-center gap-2 ${i > at ? "text-ink-faint/60" : i === at ? "text-ink" : "text-ink-faint"}`}>
          <span className={`flex h-3.5 w-3.5 items-center justify-center rounded-full border
            ${i < at ? "border-blueprint bg-blueprint" : i === at ? "border-blueprint" : "border-rule"}`}>
            {i < at && (
              <svg width="8" height="8" viewBox="0 0 12 12" fill="none" stroke="white" strokeWidth="2.2" aria-hidden="true">
                <path d="M2.5 6.2l2.3 2.3 4.7-5" />
              </svg>
            )}
            {i === at && <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-blueprint" />}
          </span>
          {labels[p]}
        </li>
      ))}
    </ol>
  );
}