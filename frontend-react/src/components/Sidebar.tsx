import { Info } from 'lucide-react';
import UploadPanel from './UploadPanel';
import KnowledgeBaseList from './KnowledgeBaseList';

export default function Sidebar() {
  return (
    <div className="flex flex-col gap-7">
      <UploadPanel />
      <div className="border-t border-line pt-5">
        <KnowledgeBaseList />
      </div>
      <div className="rounded-xl bg-surface-2/50 border border-line p-4 flex items-start gap-2.5">
        <Info className="w-4 h-4 text-ink-muted shrink-0 mt-0.5" />
        <p className="text-xs text-ink-muted leading-relaxed">
          Numbered badges like <span className="font-semibold text-ink">[1]</span> in an answer link to
          the exact passage it came from. &ldquo;Match&rdquo; shows how closely that passage fits your question.
        </p>
      </div>
    </div>
  );
}
