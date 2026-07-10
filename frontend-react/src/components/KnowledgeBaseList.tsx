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
      <h2 className="font-display text-sm font-semibold mb-2">Knowledge Base</h2>
      {isLoading && <p className="text-xs text-charcoal-muted">Loading...</p>}
      {isError && <p className="text-xs text-accent">Cannot reach backend.</p>}
      {!isLoading && !isError && docs.length === 0 && (
        <p className="text-xs text-charcoal-muted">No documents yet.</p>
      )}
      <ul className="flex flex-col gap-1">
        {docs.map((name) => (
          <li key={name} className="flex items-center gap-2 text-sm group">
            <FileText className="w-4 h-4 text-charcoal-muted shrink-0" />
            <span className="truncate flex-1" title={name}>{name}</span>
            <button
              onClick={() => deleteOne.mutate(name)}
              aria-label={`Delete ${name}`}
              className="opacity-0 group-hover:opacity-100 text-charcoal-muted hover:text-accent"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </li>
        ))}
      </ul>
      {docs.length > 0 && (
        <button
          onClick={() => setConfirmAll(true)}
          className="mt-3 text-xs text-accent hover:underline"
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
