import { FileText } from 'lucide-react';
import type { Citation } from '../types';

export default function SourcesPanel({ citations }: { citations: Citation[] }) {
  return (
    <div className="h-full overflow-y-auto border-l border-line bg-cream-card/50 p-5">
      <h2 className="font-display text-sm font-semibold mb-3">Sources</h2>
      {citations.length === 0 && (
        <p className="text-xs text-charcoal-muted">Sources appear here after you ask a question.</p>
      )}
      <div className="flex flex-col gap-3">
        {citations.map((c, i) => (
          <div
            key={`${c.file}-${c.page}-${i}`}
            id={`source-card-${c.file}-${c.page}`}
            className="rounded-xl border border-line bg-cream-card p-3"
          >
            <div className="flex items-center gap-2 text-xs font-medium">
              <FileText className="w-3.5 h-3.5 text-accent shrink-0" />
              <span className="truncate" title={c.file}>{c.file}</span>
            </div>
            <div className="text-[11px] text-charcoal-muted mt-1 flex flex-wrap gap-x-2">
              {c.page != null && <span>Page {c.page}</span>}
              {c.header && <span className="italic">{c.header}</span>}
              {typeof c.score === 'number' && <span>Relevance {c.score.toFixed(2)}</span>}
            </div>
            <p className="text-xs text-charcoal-muted mt-2 line-clamp-4">{c.text}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
