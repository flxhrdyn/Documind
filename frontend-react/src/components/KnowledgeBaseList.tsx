import { useState } from 'react';
import { FileText, Trash2 } from 'lucide-react';
import { useDocuments, useDeleteDocument, useDeleteAllDocuments } from '../hooks/useDocuments';
import ConfirmDialog from './ConfirmDialog';

export default function KnowledgeBaseList() {
  const { data, isLoading, isError } = useDocuments();
  const deleteOne = useDeleteDocument();
  const deleteAll = useDeleteAllDocuments();
  const [confirmAll, setConfirmAll] = useState(false);

  const docs = data?.documents ?? [];

  return (
    <div>
      <div className="flex items-center justify-between mb-2.5">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-ink-muted">
          Knowledge Base
        </h2>
        {docs.length > 0 && (
          <span className="text-[11px] font-mono text-ink-muted">{docs.length}</span>
        )}
      </div>
      {isLoading && <p className="text-xs text-ink-muted">Loading...</p>}
      {isError && <p className="text-xs text-accent-ink">Cannot reach backend.</p>}
      {!isLoading && !isError && docs.length === 0 && (
        <p className="text-xs text-ink-muted">No documents yet.</p>
      )}
      <ul className="flex flex-col gap-0.5">
        {docs.map((name) => (
          <li
            key={name}
            className="group flex items-center gap-2.5 text-sm px-2 py-1.5 -mx-2 rounded-lg hover:bg-surface transition-colors"
          >
            <span className="w-6 h-6 rounded-md bg-surface-2 flex items-center justify-center shrink-0">
              <FileText className="w-3.5 h-3.5 text-ink-muted" />
            </span>
            <span className="truncate flex-1" title={name}>{name}</span>
            <button
              onClick={() => deleteOne.mutate(name)}
              aria-label={`Delete ${name}`}
              className="opacity-0 group-hover:opacity-100 focus-visible:opacity-100 text-ink-muted hover:text-accent-ink transition-opacity"
            >
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          </li>
        ))}
      </ul>
      {docs.length > 0 && (
        <button
          onClick={() => setConfirmAll(true)}
          className="mt-3 text-xs text-accent-ink hover:underline"
        >
          Delete all documents
        </button>
      )}
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
