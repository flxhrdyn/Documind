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
        <p className="text-[10px] text-ink-muted leading-relaxed">
          Ask questions about your uploaded documents. Answers are grounded strictly in the parsed files.
        </p>
      </div>
    </div>
  );
}
