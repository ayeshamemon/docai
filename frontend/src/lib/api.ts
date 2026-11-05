// Talks to the FastAPI backend (Step 3.1–3.3).
// The address comes from .env.local: NEXT_PUBLIC_API_URL=http://localhost:8000

export const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Stage =
  | "queued"
  | "parsing"
  | "chunking"
  | "indexing"
  | "visual_indexing"
  | "ready"
  | "failed"
  | "not indexed";

export interface DocumentSummary {
  doc_id: string;
  filename: string;
  num_pages?: number;
  chunks_indexed?: number;
  pages_indexed?: number;
  status: Stage;
}

export interface Status {
  doc_id: string;
  stage: Stage;
  filename?: string;
  pages_done?: number;
  pages_total?: number;
  chunks?: number;
  seconds?: number;
  error?: string;
  duplicate?: boolean;
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `${res.status} ${res.statusText}`);
  }
  return res.json();
}

export async function listDocuments(): Promise<DocumentSummary[]> {
  return json(await fetch(`${API}/documents`, { cache: "no-store" }));
}

export async function uploadDocument(file: File): Promise<Status> {
  const form = new FormData();
  form.append("file", file);
  return json(await fetch(`${API}/documents`, { method: "POST", body: form }));
}

export async function getStatus(docId: string): Promise<Status> {
  return json(await fetch(`${API}/documents/${docId}/status`, { cache: "no-store" }));
}

export const isBusy = (s: Stage) =>
  ["queued", "parsing", "chunking", "indexing", "visual_indexing"].includes(s);

/** Human-readable stage, e.g. "Visual indexing 6/11". */
export function stageLabel(st: Pick<Status, "stage" | "pages_done" | "pages_total" | "error">): string {
  switch (st.stage) {
    case "queued": return "Queued";
    case "parsing": return "Parsing…";
    case "chunking": return "Chunking…";
    case "indexing": return "Indexing text…";
    case "visual_indexing":
      return st.pages_total
        ? `Indexing pages ${st.pages_done ?? 0}/${st.pages_total}`
        : "Indexing pages…";
    case "ready": return "Ready";
    case "failed": return st.error ? `Failed: ${st.error}` : "Failed";
    default: return "Not indexed";
  }
}

/** Rough overall progress 0–100 for the progress bar. */
export function stageProgress(st: Pick<Status, "stage" | "pages_done" | "pages_total">): number {
  const base: Record<string, number> = {
    queued: 3, parsing: 15, chunking: 45, indexing: 55, visual_indexing: 65, ready: 100,
  };
  if (st.stage === "visual_indexing" && st.pages_total) {
    return 65 + Math.round(35 * ((st.pages_done ?? 0) / st.pages_total));
  }
  return base[st.stage] ?? 0;
}