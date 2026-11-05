"use client";

// Left column: drop zone for uploads + list of documents with live status.

import { useCallback, useEffect, useRef, useState } from "react";
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
    <aside className="flex h-full flex-col gap-3 border-r border-neutral-200 bg-neutral-50 p-3">
      <h2 className="px-1 text-xs font-semibold uppercase tracking-wide text-neutral-500">
        Documents
      </h2>

      {/* Drop zone */}
      <button
        type="button"
        onClick={() => fileInput.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); handleFiles(e.dataTransfer.files); }}
        className={`rounded-lg border-2 border-dashed px-3 py-5 text-center text-sm transition
          ${dragging ? "border-blue-500 bg-blue-50 text-blue-700"
                     : "border-neutral-300 text-neutral-500 hover:border-neutral-400"}`}
      >
        <span className="block font-medium">Drop a PDF here</span>
        <span className="text-xs">or click to choose</span>
      </button>
      <input
        ref={fileInput} type="file" accept="application/pdf" multiple hidden
        onChange={(e) => { handleFiles(e.target.files); e.target.value = ""; }}
      />

      {error && <p className="rounded bg-red-50 px-2 py-1 text-xs text-red-700">{error}</p>}

      {/* Document list */}
      <ul className="flex-1 space-y-1 overflow-y-auto">
        {docs.map((d) => {
          const st = live[d.doc_id] ?? { doc_id: d.doc_id, stage: d.status };
          const busy = isBusy(st.stage);
          const ready = st.stage === "ready";
          const active = d.doc_id === selectedId;
          return (
            <li key={d.doc_id}>
              <button
                type="button"
                disabled={!ready}
                onClick={() => onSelect(d)}
                title={d.filename}
                className={`w-full rounded-md px-2 py-2 text-left text-sm transition
                  ${active ? "bg-white shadow-sm ring-1 ring-neutral-200" : "hover:bg-white/70"}
                  ${!ready ? "cursor-default opacity-80" : ""}`}
              >
                <span className="block truncate font-medium text-neutral-800">
                  {d.filename ?? d.doc_id}
                </span>
                <span className={`block text-xs ${st.stage === "failed" ? "text-red-600" : "text-neutral-500"}`}>
                  {ready && d.num_pages ? `${d.num_pages} pages` : stageLabel(st)}
                </span>
                {busy && (
                  <span className="mt-1 block h-1 overflow-hidden rounded bg-neutral-200">
                    <span
                      className="block h-full bg-blue-500 transition-all duration-500"
                      style={{ width: `${stageProgress(st)}%` }}
                    />
                  </span>
                )}
              </button>
            </li>
          );
        })}
        {docs.length === 0 && !error && (
          <li className="px-2 text-xs text-neutral-400">No documents yet</li>
        )}
      </ul>
    </aside>
  );
}