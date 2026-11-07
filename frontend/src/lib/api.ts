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

// ---------- document details (Step 3.3) ----------

export interface PageInfo {
  page: number;
  type: "digital" | "scanned" | "blank";
  width: number;   // PDF points
  height: number;  // PDF points
}

export interface DocumentDetails {
  doc_id: string;
  filename: string;
  num_pages: number;
  has_file: boolean;
  has_graph?: boolean;   // GraphRAG built for this document
  pages: PageInfo[];
}

/** A region to highlight: page number + [left, top, right, bottom] in PDF points (top-left origin). */
export interface Box {
  page: number;
  bbox: [number, number, number, number];
}

export async function getDetails(docId: string): Promise<DocumentDetails> {
  return json(await fetch(`${API}/documents/${docId}`, { cache: "no-store" }));
}

export const fileUrl = (docId: string) => `${API}/documents/${docId}/file`;
export const pageImageUrl = (docId: string, page: number) => `${API}/documents/${docId}/pages/${page}`;

// ---------- chat (Step 3.1 /chat, streamed as Server-Sent Events) ----------

export type Mode = "rerank" | "hybrid" | "dense" | "bm25" | "graph+rerank";

export interface ChunkSource {
  kind: "chunk";
  doc_id: string;
  chunk_id: string;
  type: "text" | "table" | "figure";
  heading: string;
  pages: number[];
  boxes: Box[];
  text: string;
  score: number;
  from_graph?: boolean;      // added by the graph walk (Graph mode)
  graph_why?: string | null; // why the graph brought it, e.g. "P100 GPU -> big models: trained on"
}

export interface PageSource {
  kind: "page";
  doc_id: string;
  page: number;
  reason?: string;
  score: number;
}

export type Source = ChunkSource | PageSource;

export interface Timing {
  retrieve_s: number;
  first_token_s: number | null;
  total_s: number;
}

export interface ChatRequest {
  question: string;
  doc_id: string;
  mode?: Mode;
  images?: number;
}

export interface ChatHandlers {
  onSources: (s: Source[]) => void;
  onToken: (text: string) => void;
  onDone: (t: Timing) => void;
  onError: (message: string) => void;
}

/**
 * POST /chat and read the stream. (The browser's EventSource only supports GET,
 * so we read the response body ourselves and split it into events.)
 */
export async function streamChat(req: ChatRequest, h: ChatHandlers, signal?: AbortSignal) {
  let res: Response;
  try {
    res = await fetch(`${API}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
      signal,
    });
  } catch (e) {
    if ((e as Error).name !== "AbortError") h.onError("Cannot reach the backend.");
    return;
  }
  if (!res.ok || !res.body) {
    const body = await res.json().catch(() => ({}));
    h.onError(body.detail ?? `${res.status} ${res.statusText}`);
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      // events are separated by a blank line
      let sep;
      while ((sep = buffer.indexOf("\n\n")) !== -1) {
        const raw = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        let event = "message";
        let data = "";
        for (const line of raw.split("\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          else if (line.startsWith("data:")) data += line.slice(5).trim();
        }
        if (!data) continue;
        const payload = JSON.parse(data);
        if (event === "sources") h.onSources(payload);
        else if (event === "token") h.onToken(payload);
        else if (event === "done") h.onDone(payload);
        else if (event === "error") h.onError(payload);
      }
    }
  } catch (e) {
    if ((e as Error).name !== "AbortError") h.onError("The connection was interrupted.");
  }
}