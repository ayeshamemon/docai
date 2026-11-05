"use client";

// Right column: ask questions about the open document; answers stream in word by word.

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { Mode, Source, Timing, streamChat } from "@/lib/api";

interface Message {
  role: "user" | "assistant";
  text: string;
  sources?: Source[];
  timing?: Timing;
  error?: string;
  streaming?: boolean;
}

interface Props {
  docId: string | null;
  docName?: string;
}

export default function ChatPanel({ docId, docName }: Props) {
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
            <AssistantMessage key={i} m={m} />
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

function AssistantMessage({ m }: { m: Message }) {
  const [open, setOpen] = useState(false);
  const chunks = m.sources?.filter((s) => s.kind === "chunk") ?? [];
  const pages = m.sources?.filter((s) => s.kind === "page") ?? [];

  return (
    <div className="text-sm text-neutral-800">
      {m.text ? (
        <div className="prose-chat">
          <ReactMarkdown>{m.text}</ReactMarkdown>
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
                    <span className="font-medium text-neutral-700">
                      p. {s.pages.join(", ")} · {s.type}
                    </span>
                    {s.heading && <span className="text-neutral-500"> — {s.heading}</span>}
                    <p className="line-clamp-2 text-neutral-500">{s.text.replace(s.heading, "").trim()}</p>
                  </li>
                ),
              )}
              {pages.map((s) =>
                s.kind === "page" && (
                  <li key={`p${s.page}`} className="text-neutral-600">
                    🖼 page image p. {s.page} {s.reason && `(${s.reason})`}
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