import { BookOpen, FileText, Compass } from 'lucide-react';
import type { Citation } from '../types';

function scoreClass(score: number): string {
  if (score >= 0.9) return 'text-emerald-500 bg-emerald-500/10 border-emerald-500/20';
  if (score >= 0.75) return 'text-accent-ink bg-accent/10 border-accent/20';
  return 'text-ink-muted bg-surface-2 border-line';
}

export default function SourcesPanel({ citations }: { citations: Citation[] }) {
  return (
    <div className="h-full flex flex-col bg-surface/20 border-l border-line">
      <div className="p-4 border-b border-line flex items-center justify-between">
        <div className="flex items-center gap-2">
          <BookOpen className="w-4 h-4 text-accent-ink" />
          <h3 className="text-sm font-semibold font-display text-ink">References ({citations.length})</h3>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-3.5">
        {citations.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center py-20 px-4">
            <Compass className="w-8 h-8 text-ink-muted/40 mb-2.5 stroke-[1.5]" />
            <p className="text-xs font-medium text-ink-muted">No active sources</p>
            <p className="text-[10px] text-ink-muted/70 mt-1 max-w-[200px] leading-relaxed">
              Sources appear here after you ask a question.
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            <p className="text-[11px] text-ink-muted leading-relaxed">
              Passages supporting the current response:
            </p>
            {citations.map((c, i) => (
              <div
                key={`${c.file}-${c.page}-${i}`}
                id={`source-card-${c.file}-${c.page}`}
                className="rounded-xl border border-line bg-surface p-4"
              >
                <div className="flex items-start justify-between gap-2 mb-2.5">
                  <div className="flex items-start gap-2 min-w-0">
                    <span className="flex items-center justify-center w-5 h-5 rounded-md text-[11px] font-mono font-bold bg-surface-2 text-ink-muted shrink-0">
                      {i + 1}
                    </span>
                    <div className="min-w-0">
                      <div className="flex items-center gap-1.5">
                        <FileText className="w-3 h-3 text-ink-muted shrink-0" />
                        <span className="text-xs font-semibold text-ink truncate" title={c.file}>
                          {c.file}
                        </span>
                      </div>
                      {c.page != null && (
                        <span className="text-[10px] font-mono text-ink-muted">Page {c.page}</span>
                      )}
                      {c.header && <span className="text-[10px] italic text-ink-muted ml-1">{c.header}</span>}
                    </div>
                  </div>
                  {typeof c.score === 'number' && (
                    <span className={`px-2 py-0.5 rounded-full text-[9px] font-semibold border shrink-0 ${scoreClass(c.score)}`}>
                      Match {(c.score * 100).toFixed(0)}%
                    </span>
                  )}
                </div>
                <div className="text-[11.5px] leading-relaxed text-ink-muted bg-surface-2/50 p-2.5 rounded-lg border border-line line-clamp-4">
                  {c.text}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
