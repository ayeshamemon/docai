// DocAI mark: a page with a folded corner and one marker stroke across it.
// The stroke is the brand idea: an answer is only as good as its highlighted evidence.

export function LogoMark({ size = 26, className = "" }: { size?: number; className?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true" className={className}>
      <path d="M8 3h11l7 7v17a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z"
            fill="currentColor" opacity="0.14" />
      <path d="M8 3h11l7 7v17a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z"
            fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" />
      <path d="M19 3v5a2 2 0 0 0 2 2h5" fill="none" stroke="currentColor" strokeWidth="1.8"
            strokeLinejoin="round" />
      <rect x="9.5" y="14" width="11" height="1.8" rx="0.9" fill="currentColor" opacity="0.55" />
      <path d="M8.6 18.6c4.6-1.1 9.6-1.3 15-.6l-.5 3.4c-5.2-.6-9.9-.4-14.2.6z" fill="#ffe14d" />
      <rect x="9.5" y="24" width="8" height="1.8" rx="0.9" fill="currentColor" opacity="0.55" />
    </svg>
  );
}

export default function Logo({ className = "" }: { className?: string }) {
  return (
    <span className={`inline-flex items-center gap-2 ${className}`}>
      <LogoMark />
      <span className="text-[17px] font-semibold tracking-tight">DocAI</span>
    </span>
  );
}