"use client";

// Workspace (/app): documents | PDF viewer | chat.
// 3.4a documents ✓ · 3.4b viewer ✓ · 3.4c chat ✓ · 3.4d clickable citations ✓

import dynamic from "next/dynamic";
import { useCallback, useEffect, useState } from "react";
import ChatPanel from "@/components/ChatPanel";
import DocumentSidebar from "@/components/DocumentSidebar";
import { Box, DocumentDetails, DocumentSummary, getDetails } from "@/lib/api";

// PDF.js only runs in the browser, so the viewer is loaded on the client only.
const PdfViewer = dynamic(() => import("@/components/PdfViewer"), {
  ssr: false,
  loading: () => <p className="p-6 text-sm text-ink-faint">Opening the viewer…</p>,
});

export default function Home() {
  const [doc, setDoc] = useState<DocumentSummary | null>(null);
  const [details, setDetails] = useState<DocumentDetails | null>(null);
  const [error, setError] = useState<string | null>(null);
  // What the viewer should show for the last clicked citation / source.
  const [highlights, setHighlights] = useState<Box[]>([]);
  const [goTo, setGoTo] = useState<{ page: number; nonce: number } | null>(null);

  // React to a different document ID only. A refreshed copy of the same document must not
  // clear the details: that would unmount the viewer and send it back to page 1.
  const docId = doc?.doc_id ?? null;
  useEffect(() => {
    if (!docId) return;
    let cancelled = false;
    setDetails((d) => (d?.doc_id === docId ? d : null));
    setError(null);
    setHighlights([]);
    setGoTo(null);
    getDetails(docId)
      .then((d) => { if (!cancelled) setDetails((old) => (old?.doc_id === d.doc_id ? old : d)); })
      .catch((e) => { if (!cancelled) setError(e.message); });
    return () => { cancelled = true; };
  }, [docId]);

  // A citation was clicked: highlight its boxes and jump there.
  // nonce lets the same page be clicked twice and still scroll.
  const onCite = useCallback((page: number, boxes: Box[]) => {
    setHighlights(boxes);
    setGoTo({ page, nonce: Date.now() });
  }, []);

  return (
    // grid-rows-[minmax(0,1fr)] + overflow-hidden: the row is exactly the screen height, so only the
    // viewer scrolls (without it the row grew to the full PDF height and the whole window scrolled)
    <main className="grid h-screen grid-cols-[264px_1fr_420px] grid-rows-[minmax(0,1fr)] overflow-hidden bg-paper text-ink">
      <DocumentSidebar selectedId={docId} onSelect={setDoc} />

      <section className="min-h-0 min-w-0 border-r border-rule bg-desk">
        {!doc && (
          <div className="flex h-full items-center justify-center p-10">
            <div className="max-w-sm">
              <p className="text-2xl font-semibold leading-snug tracking-tight text-ink">
                Ask a PDF anything.
                <br />
                See the exact passage behind every answer.
              </p>
              <p className="mt-4 text-sm leading-relaxed text-ink-soft">
                Reports, papers, scans and handwritten notes all work. Tables and charts are
                read too. Add a PDF on the left, or open one from your library.
              </p>
            </div>
          </div>
        )}
        {doc && error && <p className="m-6 rounded bg-danger-bg px-3 py-2 text-sm text-danger-text">Could not open this document: {error}</p>}
        {doc && details && (
          <PdfViewer
            key={details.doc_id}
            docId={details.doc_id}
            pages={details.pages}
            hasFile={details.has_file}
            goTo={goTo}
            highlights={highlights}
          />
        )}
      </section>

      <section className="min-h-0 min-w-0">
        {/* key: a new document starts a fresh chat */}
        <ChatPanel key={docId ?? "none"} docId={docId} docName={doc?.filename} onCite={onCite}
          hasGraph={!!details?.has_graph && details.doc_id === docId} />
      </section>
    </main>
  );
}