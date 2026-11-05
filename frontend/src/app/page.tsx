"use client";

// Main screen: documents | PDF viewer | chat.
// 3.4a documents ✓ · 3.4b viewer ✓ · 3.4c chat (placeholder)

import dynamic from "next/dynamic";
import { useEffect, useState } from "react";
import DocumentSidebar from "@/components/DocumentSidebar";
import { DocumentDetails, DocumentSummary, getDetails } from "@/lib/api";

// PDF.js only runs in the browser, so the viewer is loaded on the client only.
const PdfViewer = dynamic(() => import("@/components/PdfViewer"), {
  ssr: false,
  loading: () => <p className="p-6 text-sm text-neutral-400">Loading viewer…</p>,
});

export default function Home() {
  const [doc, setDoc] = useState<DocumentSummary | null>(null);
  const [details, setDetails] = useState<DocumentDetails | null>(null);
  const [error, setError] = useState<string | null>(null);

  // React to a different document ID only. A refreshed copy of the same document must not
  // clear the details: that would unmount the viewer and send it back to page 1.
  const docId = doc?.doc_id ?? null;
  useEffect(() => {
    if (!docId) return;
    let cancelled = false;
    setDetails((d) => (d?.doc_id === docId ? d : null));
    setError(null);
    getDetails(docId)
      .then((d) => { if (!cancelled) setDetails((old) => (old?.doc_id === d.doc_id ? old : d)); })
      .catch((e) => { if (!cancelled) setError(e.message); });
    return () => { cancelled = true; };
  }, [docId]);

  return (
    // grid-rows-[minmax(0,1fr)] + overflow-hidden: the row is exactly the screen height, so only the
    // viewer scrolls (without it the row grew to the full PDF height and the whole window scrolled)
    <main className="grid h-screen grid-cols-[260px_1fr_400px] grid-rows-[minmax(0,1fr)] overflow-hidden bg-white text-neutral-900">
      <DocumentSidebar selectedId={docId} onSelect={setDoc} />

      <section className="min-h-0 min-w-0 border-r border-neutral-200 bg-neutral-100">
        {!doc && (
          <div className="flex h-full items-center justify-center text-sm text-neutral-400">
            Select or upload a document
          </div>
        )}
        {doc && error && <p className="p-6 text-sm text-red-600">{error}</p>}
        {doc && details && (
          <PdfViewer
            key={details.doc_id}
            docId={details.doc_id}
            pages={details.pages}
            hasFile={details.has_file}
          />
        )}
      </section>

      <section className="flex min-h-0 items-center justify-center">
        <p className="text-sm text-neutral-400">Chat (Step 3.4c)</p>
      </section>
    </main>
  );
}