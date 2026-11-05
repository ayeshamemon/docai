"use client";

// Right column: ask questions about the open document; answers stream in word by word.

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { Box, Mode, Source, Timing, streamChat } from "@/lib/api";

interface Message {
  role: "user" | "assistant";
  text: string;
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

// Citations Gemini writes in many shapes: [p. 8]  [p. 3, 4]  [p. 3, p. 4]  [pp. 3-5]
// [p. 3–4, 9]  [page 12]  [pages 3 and 5]  [p. 3; p. 9]  →  one clickable chip per page.
// Any bracket starting with p./pp./page is checked strictly, so "[Table 2]" or
// "[p. 3 for details]" stay plain text.
const CITATION = /\[\s*((?:pp?\.|pages?)\s*[^\]\n]{1,60})\]/gi;
const VALID = /^(?:(?:pp?\.|pages?)?\s*\d+(?:\s*[–-]\s*\d+)?\s*(?:,|;|and|&)?\s*)+$/i;

function citePages(list: string): number[] {
  const pages: number[] = [];
  const clean = list.replace(/pp?\.|pages?/gi, " ");
  for (const part of clean.split(/,|;|\band\b|&/i)) {
    const [a, b] = part.split(/[–-]/).map((x) => parseInt(x.trim(), 10));
    if (Number.isNaN(a)) continue;
    if (b && b >= a && b - a < 20) for (let p = a; p <= b; p++) pages.push(p);
    else pages.push(a);
  }
  return [...new Set(pages)];
}

/** Turn citations into Markdown links (#cite-8) that the renderer shows as buttons. */
function linkCitations(text: string): string {
  return text.replace(CITATION, (whole: string, inner: string) => {
    if (!VALID.test(inner.trim())) return whole;
    const pages = citePages(inner);
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

  async function send() {
    const question = input.trim();
    if (!question || !docId || busy) return;
    setInput("");
    setBusy(true);
    setMessages((ms) => [
      ...ms,
      { role: "user", text: question },
      { role: "assistant", text: "", streaming: true },
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
      <div className="flex h-full items-center justify-center p-6 text-center text-sm text-neutral-400">
        Open a document to ask questions about it
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* header with settings */}
      <div className="flex items-center justify-between gap-2 border-b border-neutral-200 px-3 py-1.5 text-sm">
        <span className="truncate font-medium text-neutral-700" title={docName}>
          {docName ?? "Chat"}
        </span>
        <div className="flex shrink-0 items-center gap-2 text-xs text-neutral-600">
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value as Mode)}
            className="rounded border border-neutral-200 bg-white px-1 py-0.5"
            title="Retrieval mode"
          >
            <option value="rerank">Rerank</option>
            <option value="hybrid">Hybrid</option>
          </select>
          <label className="flex items-center gap-1" title="Also send page images (charts, scans) to Gemini">
            <input type="checkbox" checked={images} onChange={(e) => setImages(e.target.checked)} />
            Images
          </label>
        </div>
      </div>

      {/* messages */}
      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-3 py-4">
        {messages.length === 0 && (
          <p className="mt-8 text-center text-sm text-neutral-400">
            Ask anything about this document.
          </p>
        )}
        {messages.map((m, i) =>
          m.role === "user" ? (
            <div key={i} className="ml-8 rounded-lg bg-neutral-900 px-3 py-2 text-sm text-white">
              {m.text}
            </div>
          ) : (
            <AssistantMessage key={i} m={m} onCite={onCite} />
          ),
        )}
        <div ref={bottom} />
      </div>

      {/* input */}
      <div className="border-t border-neutral-200 p-2">
        <div className="flex items-end gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
            }}
            rows={2}
            placeholder="Ask a question…  (Enter to send, Shift+Enter for a new line)"
            className="min-h-0 flex-1 resize-none rounded-md border border-neutral-200 px-2 py-1.5 text-sm outline-none focus:border-neutral-400"
          />
          {busy ? (
            <button type="button" onClick={stop}
              className="rounded-md bg-neutral-200 px-3 py-2 text-sm text-neutral-800 hover:bg-neutral-300">
              Stop
            </button>
          ) : (
            <button type="button" onClick={send} disabled={!input.trim()}
              className="rounded-md bg-neutral-900 px-3 py-2 text-sm text-white hover:bg-neutral-700 disabled:opacity-40">
              Send
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function AssistantMessage({ m, onCite }: { m: Message; onCite?: CiteHandler }) {
  const [open, setOpen] = useState(false);
  const cite = (page: number) => onCite?.(page, boxesOnPage(m.sources, page));
  const chunks = m.sources?.filter((s) => s.kind === "chunk") ?? [];
  const pages = m.sources?.filter((s) => s.kind === "page") ?? [];

  return (
    <div className="text-sm text-neutral-800">
      {m.text ? (
        <div className="prose-chat">
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
                    className="mx-0.5 inline-flex items-center rounded bg-amber-100 px-1 py-px align-baseline text-[11px] font-medium text-amber-800 hover:bg-amber-200"
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
      ) : m.streaming && !m.error ? (
        <p className="animate-pulse text-neutral-400">
          {m.sources ? "Writing the answer…" : "Searching the document…"}
        </p>
      ) : null}

      {m.error && <p className="mt-1 rounded bg-red-50 px-2 py-1 text-xs text-red-700">{m.error}</p>}

      {(chunks.length > 0 || pages.length > 0) && (
        <div className="mt-2 text-xs text-neutral-500">
          <button type="button" onClick={() => setOpen((o) => !o)} className="hover:text-neutral-800">
            {open ? "▾" : "▸"} {chunks.length} sources
            {pages.length > 0 && ` · ${pages.length} page images`}
            {m.timing && ` · ${m.timing.total_s}s`}
          </button>
          {open && (
            <ul className="mt-1 space-y-1.5 border-l-2 border-neutral-200 pl-2">
              {chunks.map((s) =>
                s.kind === "chunk" && (
                  <li key={s.chunk_id}>
                    <button
                      type="button"
                      onClick={() => onCite?.(s.pages[0], s.boxes)}
                      title="Show this source in the document"
                      className="font-medium text-neutral-700 underline-offset-2 hover:underline"
                    >
                      p. {s.pages.join(", ")} · {s.type}
                    </button>
                    {s.heading && <span className="text-neutral-500"> — {s.heading}</span>}
                    <p className="line-clamp-2 text-neutral-500">{s.text.replace(s.heading, "").trim()}</p>
                  </li>
                ),
              )}
              {pages.map((s) =>
                s.kind === "page" && (
                  <li key={`p${s.page}`} className="text-neutral-600">
                    <button type="button" onClick={() => onCite?.(s.page, [])}
                      className="underline-offset-2 hover:underline">
                      🖼 page image p. {s.page} {s.reason && `(${s.reason})`}
                    </button>
                  </li>
                ),
              )}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}