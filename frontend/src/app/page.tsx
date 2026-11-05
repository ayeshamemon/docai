"use client";

// Main screen: documents | PDF viewer | chat.
// 3.4a: only the documents column is real; viewer (3.4b) and chat (3.4c) are placeholders.

import { useState } from "react";
import DocumentSidebar from "@/components/DocumentSidebar";
import type { DocumentSummary } from "@/lib/api";

export default function Home() {
  const [doc, setDoc] = useState<DocumentSummary | null>(null);

  return (
    <main className="grid h-screen grid-cols-[260px_1fr_400px] bg-white text-neutral-900">
      <DocumentSidebar selectedId={doc?.doc_id ?? null} onSelect={setDoc} />

      <section className="flex items-center justify-center border-r border-neutral-200 bg-neutral-100">
        {doc ? (
          <p className="text-sm text-neutral-500">
            PDF viewer for <b>{doc.filename}</b> (Step 3.4b)
          </p>
        ) : (
          <p className="text-sm text-neutral-400">Select or upload a document</p>
        )}
      </section>

      <section className="flex items-center justify-center">
        <p className="text-sm text-neutral-400">Chat (Step 3.4c)</p>
      </section>
    </main>
  );
}