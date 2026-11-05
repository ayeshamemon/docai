"use client";

// Middle column: the PDF itself, with page navigation, zoom and highlight boxes.
//
// - Pages are drawn only when they are near the visible area (fast for 100+ page PDFs).
// - Highlight boxes come in PDF points (top-left origin) and are scaled to screen size:
//   scale = rendered page width / page width in points.
// - If the original PDF file is missing, the saved page images are shown instead.

import { Suspense, memo, startTransition, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/TextLayer.css";
import { Box, PageInfo, fileUrl, pageImageUrl } from "@/lib/api";

// PDF.js renders in a web worker; the file is copied into /public at install time.
pdfjs.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.mjs";

// Safari polyfill: PDF.js reads the text layer with `for await (… of readableStream)`,
// but Safari's ReadableStream is not async-iterable yet. Add it if it is missing.
if (typeof ReadableStream !== "undefined" &&
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    typeof (ReadableStream.prototype as any)[Symbol.asyncIterator] !== "function") {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (ReadableStream.prototype as any)[Symbol.asyncIterator] = async function* (this: ReadableStream) {
    const reader = this.getReader();
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) return;
        yield value;
      }
    } finally {
      reader.releaseLock();
    }
  };
}

const GAP = 16;          // px between pages
const OVERSCAN = 1500;   // px above/below the viewport where pages get drawn
const MAX_RENDERED = 40; // drawn pages kept in memory; farther ones are released
const NO_BOXES: Box[] = [];
const LOADING = <p className="p-6 text-sm text-ink-faint">Opening the PDF…</p>;
const PAGE_PLACEHOLDER = <div className="h-full w-full bg-white" />;

interface Props {
  docId: string;
  pages: PageInfo[];
  hasFile: boolean;
  /** Jump to this page; change `nonce` to jump again to the same page. */
  goTo?: { page: number; nonce: number } | null;
  highlights?: Box[];
}

interface Slot { page: number; top: number; height: number; scale: number }

export default function PdfViewer({ docId, pages, hasFile, goTo, highlights = NO_BOXES }: Props) {
  const scroller = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState(800);
  const [viewHeight, setViewHeight] = useState(800);
  const [zoom, setZoom] = useState(1);                 // 1 = fit width
  const [pdfFailed, setPdfFailed] = useState(false);
  const [textLayer, setTextLayer] = useState(true);    // switched off if this browser can't draw it
  const [currentPage, setCurrentPage] = useState(1);
  const [rendered, setRendered] = useState<Set<number>>(() => new Set([1, 2, 3]));

  const useImages = !hasFile || pdfFailed;

  // Measure the column so "fit width" works and follows window resizes.
  useEffect(() => {
    const el = scroller.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      setContainerWidth(el.clientWidth);
      setViewHeight(el.clientHeight);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const pageWidth = Math.max(200, Math.round((containerWidth - 48) * zoom));

  // Top position and height of each page.
  const layout: Slot[] = useMemo(() => {
    let y = GAP;
    return pages.map((p) => {
      const height = Math.round(pageWidth * (p.height / p.width));
      const slot = { page: p.page, top: y, height, scale: pageWidth / p.width };
      y += height + GAP;
      return slot;
    });
  }, [pages, pageWidth]);
  const totalHeight = layout.length ? layout[layout.length - 1].top + layout[layout.length - 1].height + GAP : 0;

  // Highlights grouped by page (stable arrays, so unchanged pages don't re-render).
  const boxesByPage = useMemo(() => {
    const m = new Map<number, Box[]>();
    highlights.forEach((h) => m.set(h.page, [...(m.get(h.page) ?? []), h]));
    return m;
  }, [highlights]);

  // Called on scroll (once per animation frame): which pages are near, which page is current.
  // State only changes when the answer changes, so scrolling does not re-render every page.
  const frame = useRef<number | null>(null);
  const update = useCallback(() => {
    frame.current = null;
    const el = scroller.current;
    if (!el || layout.length === 0) return;
    const top = el.scrollTop;
    const probe = top + el.clientHeight / 3;
    const cur = layout.find((l) => probe >= l.top && probe <= l.top + l.height + GAP)?.page ?? 1;
    setCurrentPage(cur);

    const near = layout
      .filter((l) => l.top + l.height > top - OVERSCAN && l.top < top + el.clientHeight + OVERSCAN)
      .map((l) => l.page);
    // startTransition: drawing newly visible pages may "suspend" (wait for PDF.js). In a
    // transition React keeps the current screen instead of hiding the whole viewer, which
    // reset the scroll to page 1 and re-mounted the viewer in a loop.
    startTransition(() => setRendered((prev) => {
      if (near.every((p) => prev.has(p))) return prev;          // nothing new → no re-render
      const next = new Set(prev);
      near.forEach((p) => next.add(p));
      if (next.size > MAX_RENDERED) {                           // release the farthest pages
        [...next].sort((a, b) => Math.abs(b - cur) - Math.abs(a - cur))
          .slice(0, next.size - MAX_RENDERED).forEach((p) => next.delete(p));
      }
      return next;
    }));
  }, [layout]);

  const onScroll = () => {
    if (frame.current === null) frame.current = requestAnimationFrame(update);
  };
  useEffect(() => { update(); }, [update, viewHeight]);        // after layout/zoom/resize

  const scrollToPage = useCallback((page: number) => {
    const l = layout.find((x) => x.page === page);
    if (l) scroller.current?.scrollTo({ top: l.top - GAP, behavior: "smooth" });
  }, [layout]);

  // Jump when the parent asks (a clicked citation): to the first highlight on that page.
  useEffect(() => {
    if (!goTo) return;
    const l = layout.find((x) => x.page === goTo.page);
    const first = boxesByPage.get(goTo.page)?.[0];
    if (l && scroller.current) {
      const offset = first ? Math.max(0, first.bbox[1] * l.scale - 80) : -GAP;
      scroller.current.scrollTo({ top: l.top + offset, behavior: "smooth" });
    }
  }, [goTo]); // eslint-disable-line react-hooks/exhaustive-deps

  const onTextError = useCallback(() => setTextLayer(false), []);

  // Stable props for <Document>: if these looked "new" on every render, react-pdf could
  // reload the PDF, the content would collapse to "Loading…" and the scroll would jump to page 1.
  const file = useMemo(() => fileUrl(docId), [docId]);
  const onLoadError = useCallback((e: Error) => {
    console.warn("[viewer] PDF failed to load, using page images:", e.message);
    setPdfFailed(true);
  }, []);


  const pagesView = (
    <div className="relative mx-auto" style={{ height: totalHeight, width: pageWidth }}>
      {layout.map((l) => (
        <PageSlot
          key={l.page}
          docId={docId}
          slot={l}
          width={pageWidth}
          show={rendered.has(l.page)}
          useImages={useImages}
          textLayer={textLayer}
          boxes={boxesByPage.get(l.page) ?? NO_BOXES}
          flash={boxesByPage.has(l.page) ? goTo?.nonce ?? 0 : 0}
          onTextError={onTextError}
        />
      ))}
    </div>
  );

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* toolbar */}
      <div className="flex items-center justify-between gap-2 border-b border-rule bg-paper px-3 py-1.5 text-sm">
        <div className="flex items-center gap-1">
          <ToolButton label="Previous page" onClick={() => scrollToPage(Math.max(1, currentPage - 1))}>‹</ToolButton>
          <span className="w-20 text-center tabular-nums text-ink-soft">
            {currentPage} / {pages.length}
          </span>
          <ToolButton label="Next page" onClick={() => scrollToPage(Math.min(pages.length, currentPage + 1))}>›</ToolButton>
        </div>
        <div className="flex items-center gap-1">
          {useImages && (
            <span className="mr-2 text-xs text-ink-faint" title="The original PDF was not found, so the saved page images are shown">
              Showing page images
            </span>
          )}
          <ToolButton label="Zoom out" onClick={() => setZoom((z) => Math.max(0.5, +(z - 0.15).toFixed(2)))}>−</ToolButton>
          <span className="w-12 text-center tabular-nums text-ink-soft">{Math.round(zoom * 100)}%</span>
          <ToolButton label="Zoom in" onClick={() => setZoom((z) => Math.min(2.5, +(z + 0.15).toFixed(2)))}>+</ToolButton>
          <ToolButton label="Fit width" onClick={() => setZoom(1)}>Fit</ToolButton>
        </div>
      </div>

      {/* pages: only this area scrolls */}
      <div
        ref={scroller}
        onScroll={onScroll}
        className="min-h-0 flex-1 overflow-auto overscroll-contain bg-desk"
      >
        {useImages ? (
          pagesView
        ) : (
          <Document
            file={file}
            loading={LOADING}
            onLoadError={onLoadError}
          >
            {pagesView}
          </Document>
        )}
      </div>
    </div>
  );
}

// One page: a fixed-size white box (so the scroll height never jumps), the page drawn
// inside it when `show` is true, and the highlight boxes on top.
// memo → re-renders only when its own props change, not on every scroll.
const PageSlot = memo(function PageSlot({
  docId, slot, width, show, useImages, textLayer, boxes, flash, onTextError,
}: {
  docId: string; slot: Slot; width: number; show: boolean; useImages: boolean;
  textLayer: boolean; boxes: Box[]; flash: number; onTextError: () => void;
}) {
  return (
    <div
      className="pdf-page absolute left-0 overflow-hidden"
      style={{ top: slot.top, width, height: slot.height }}
    >
      {show && (
        useImages ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={pageImageUrl(docId, slot.page)} alt={`page ${slot.page}`}
               width={width} height={slot.height} className="block" loading="lazy" />
        ) : (
          // Own Suspense boundary per page: if a page waits, only this page shows a placeholder.
          <Suspense fallback={PAGE_PLACEHOLDER}>
            <Page
              pageNumber={slot.page}
              width={width}
              renderAnnotationLayer={false}
              renderTextLayer={textLayer}
              onGetTextError={onTextError}
              onRenderTextLayerError={onTextError}
              loading={PAGE_PLACEHOLDER}
            />
          </Suspense>
        )
      )}

      {boxes.map((h, i) => {
        const [x0, y0, x1, y1] = h.bbox;
        return (
          <div
            key={`${flash}-${i}`}
            className="marker-box pointer-events-none absolute"
            style={{
              left: x0 * slot.scale - 2,
              top: y0 * slot.scale - 2,
              width: (x1 - x0) * slot.scale + 4,
              height: (y1 - y0) * slot.scale + 4,
            }}
          />
        );
      })}

      <span className="pointer-events-none absolute bottom-1 right-2 rounded bg-white/80 px-1 text-[10px] text-[#8794a3]">
        {slot.page}
      </span>
    </div>
  );
});

function ToolButton({ label, onClick, children }: { label: string; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      title={label}
      aria-label={label}
      onClick={onClick}
      className="min-w-7 rounded px-2 py-0.5 text-ink-soft hover:bg-desk hover:text-ink active:bg-rule"
    >
      {children}
    </button>
  );
}