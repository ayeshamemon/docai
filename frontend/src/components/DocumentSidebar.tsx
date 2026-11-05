"use client";

// Left column: drop zone for uploads + list of documents with live status.

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import Logo from "@/components/Logo";
import ThemeToggle from "@/components/ThemeToggle";
import {
  DocumentSummary, Status, getStatus, isBusy, listDocuments,
  stageLabel, stageProgress, uploadDocument,
} from "@/lib/api";

interface Props {
  selectedId: string | null;
  onSelect: (doc: DocumentSummary) => void;
}

export default function DocumentSidebar({ selectedId, onSelect }: Props) {
  const [docs, setDocs] = useState<DocumentSummary[]>([]);
  const [live, setLive] = useState<Record<string, Status>>({}); // status of docs being processed
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    try {
      setDocs(await listDocuments());
      setError(null);
    } catch {
      setError("Cannot reach the backend. Is the server running on port 8000?");
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  // Poll every 1.5 s while any document is still being processed.
  useEffect(() => {
    const busy = docs.filter((d) => isBusy(d.status)).map((d) => d.doc_id);
    if (busy.length === 0) return;
    const timer = setInterval(async () => {
      const updates = await Promise.all(busy.map((id) => getStatus(id).catch(() => null)));
      const next: Record<string, Status> = {};
      updates.forEach((s) => { if (s) next[s.doc_id] = s; });
      setLive((old) => ({ ...old, ...next }));
      if (updates.some((s) => s && !isBusy(s.stage))) refresh(); // one finished → reload the list
    }, 1500);
    return () => clearInterval(timer);
  }, [docs, refresh]);

  async function handleFiles(files: FileList | null) {
    if (!files) return;
    for (const file of Array.from(files)) {
      if (!file.name.toLowerCase().endsWith(".pdf")) {
        setError(`${file.name} is not a PDF`);
        continue;
      }
      try {
        const st = await uploadDocument(file);
        setLive((old) => ({ ...old, [st.doc_id]: st }));
      } catch (e) {
        setError(`Upload failed: ${(e as Error).message}`);
      }
    }
    refresh();
  }

  return (
    <aside className="flex h-full min-h-0 flex-col bg-side text-side-text">
      {/* brand */}
      <div className="flex items-center justify-between px-4 pb-4 pt-4">
        <Link href="/" className="rounded text-side-text hover:opacity-90" title="DocAI home">
          <Logo />
        </Link>
        <ThemeToggle className="text-side-muted hover:bg-side-raised hover:text-side-text" />
      </div>

      {/* upload */}
      <div className="px-3">
        <button
          type="button"
          onClick={() => fileInput.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => { e.preventDefault(); setDragging(false); handleFiles(e.dataTransfer.files); }}
          className={`group w-full rounded-lg border border-dashed px-3 py-4 text-left text-sm transition-colors
            ${dragging ? "border-marker bg-marker/10"
                       : "border-side-rule hover:border-side-muted hover:bg-side-raised/50"}`}
        >
          <span className="flex items-center gap-2 font-medium text-side-text">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
                 strokeLinecap="round" aria-hidden="true" className="text-marker">
              <path d="M12 5v14M5 12h14" />
            </svg>
            Add a PDF
          </span>
          <span className="mt-0.5 block pl-6 text-xs text-side-muted">Drop it here or click to browse</span>
        </button>
        <input
          ref={fileInput} type="file" accept="application/pdf" multiple hidden
          onChange={(e) => { handleFiles(e.target.files); e.target.value = ""; }}
        />
        {error && <p className="mt-2 rounded bg-danger-bg px-2 py-1.5 text-xs text-danger-text">{error}</p>}
      </div>

      {/* library */}
      <p className="mt-6 px-4 pb-1.5 text-xs font-medium text-side-muted">Library</p>
      <ul className="min-h-0 flex-1 space-y-0.5 overflow-y-auto px-2 pb-3">
        {docs.map((d) => {
          const st = live[d.doc_id] ?? { doc_id: d.doc_id, stage: d.status };
          const busy = isBusy(st.stage);
          const ready = st.stage === "ready";
          const failed = st.stage === "failed";
          const active = d.doc_id === selectedId;
          return (
            <li key={d.doc_id}>
              <button
                type="button"
                disabled={!ready}
                onClick={() => onSelect(d)}
                title={d.filename}
                aria-current={active ? "true" : undefined}
                className={`relative flex w-full items-start gap-2.5 rounded-md py-2 pl-3 pr-2 text-left transition-colors
                  ${active ? "bg-side-raised" : ready ? "hover:bg-side-raised/60" : "cursor-default"}`}
              >
                {active && <span className="absolute inset-y-2 left-0 w-[3px] rounded bg-marker" />}
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"
                     aria-hidden="true" className={`mt-0.5 shrink-0 ${active ? "text-side-text" : "text-side-muted"}`}>
                  <path d="M7 3h7l5 5v12a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1z" />
                  <path d="M14 3v5h5" />
                </svg>
                <span className="min-w-0 flex-1">
                  <span className={`block truncate text-sm ${active ? "font-medium text-side-text" : "text-side-text/90"}`}>
                    {prettyName(d.filename ?? d.doc_id)}
                  </span>
                  <span className={`block text-xs ${failed ? "text-danger-text" : "text-side-muted"}`}>
                    {ready && d.num_pages ? `${d.num_pages} pages` : stageLabel(st)}
                  </span>
                  {busy && (
                    <span className="mt-1.5 block h-0.5 overflow-hidden rounded bg-side-rule">
                      <span
                        className="block h-full bg-marker transition-[width] duration-500"
                        style={{ width: `${stageProgress(st)}%` }}
                      />
                    </span>
                  )}
                </span>
              </button>
            </li>
          );
        })}
        {docs.length === 0 && !error && (
          <li className="px-3 text-xs text-side-muted">Your PDFs will appear here.</li>
        )}
      </ul>
    </aside>
  );
}

/** "Class-29-handwritten-notes.pdf" → "Class 29 handwritten notes" */
function prettyName(name: string): string {
  return name.replace(/\.pdf$/i, "").replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim();
}