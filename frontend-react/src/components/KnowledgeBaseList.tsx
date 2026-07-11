import { useState } from 'react';
import { FileText, Trash2, Database } from 'lucide-react';
import { useDocuments, useDeleteDocument, useDeleteAllDocuments } from '../hooks/useDocuments';
import ConfirmDialog from './ConfirmDialog';

export default function KnowledgeBaseList() {
  const { data, isLoading, isError } = useDocuments();
  const deleteOne = useDeleteDocument();
  const deleteAll = useDeleteAllDocuments();
  const [confirmAll, setConfirmAll] = useState(false);

  const docs = data?.documents ?? [];

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Database className="w-3.5 h-3.5 text-accent" />
          <h2 className="text-xs font-semibold uppercase tracking-wide text-ink-muted">Knowledge Base</h2>
        </div>
        {docs.length > 0 && (
          <button
            onClick={() => setConfirmAll(true)}
            className="text-xs font-medium text-rose-500 hover:text-rose-400 transition-colors"
          >
            Clear All
          </button>
        )}
      </div>

      {isLoading && <p className="text-xs text-ink-muted">Loading...</p>}
      {isError && <p className="text-xs text-rose-500">Cannot reach backend.</p>}

      {!isLoading && !isError && docs.length === 0 && (
        <div className="flex flex-col items-center justify-center py-8 px-4 rounded-xl border border-line bg-surface-2/20 text-center">
          <Database className="w-6 h-6 text-ink-muted/40 mb-2" />
          <p className="text-xs font-medium text-ink-muted">No documents indexed yet</p>
          <p className="text-xs text-ink-muted/70 mt-0.5">Upload a PDF file above to train the knowledge base</p>
        </div>
      )}

      <div className="space-y-1.5 max-h-80 overflow-y-auto pr-0.5">
        {docs.map((name) => (
          <div
            key={name}
            className="group flex items-center justify-between p-2.5 rounded-xl border border-line bg-surface hover:bg-surface-2/60 transition-colors duration-200"
          >
            <div className="flex items-center gap-2.5 min-w-0">
              <span className="p-1.5 bg-accent/10 text-accent-ink rounded-lg shrink-0">
                <FileText className="w-3.5 h-3.5" />
              </span>
              <span className="text-xs font-medium text-ink truncate" title={name}>
                {name}
              </span>
            </div>
            <button
              onClick={() => deleteOne.mutate(name)}
              aria-label={`Delete ${name}`}
              className="opacity-0 group-hover:opacity-100 focus-visible:opacity-100 p-1.5 hover:bg-rose-500/10 text-ink-muted hover:text-rose-500 rounded-lg transition-all"
            >
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          </div>
        ))}
      </div>

      <ConfirmDialog
        open={confirmAll}
        title="Delete all documents?"
        message="This permanently deletes all indexed documents and cannot be undone."
        confirmLabel="Delete all"
        onConfirm={() => {
          deleteAll.mutate();
          setConfirmAll(false);
        }}
        onCancel={() => setConfirmAll(false)}
      />
    </div>
  );
}
