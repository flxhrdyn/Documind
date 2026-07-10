import { useRef, useState } from 'react';
import { UploadCloud } from 'lucide-react';
import { useUploadJob } from '../hooks/useUploadJob';

export default function UploadPanel() {
  const { upload, status, isUploading } = useUploadJob();
  const inputRef = useRef<HTMLInputElement>(null);
  const [error, setError] = useState<string | null>(null);

  const onFile = async (file: File | undefined) => {
    if (!file) return;
    setError(null);
    try {
      await upload(file);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Upload failed');
    }
  };

  return (
    <div>
      <h2 className="text-xs font-semibold uppercase tracking-wide text-ink-muted mb-2.5">
        Upload PDF
      </h2>
      <button
        onClick={() => inputRef.current?.click()}
        disabled={isUploading}
        className="group w-full flex flex-col items-center gap-2 border border-dashed border-line rounded-xl py-5 text-sm bg-surface hover:border-accent hover:bg-surface transition-colors disabled:opacity-60 disabled:hover:border-line"
      >
        <span className="w-9 h-9 rounded-full bg-surface-2 group-hover:bg-accent/15 flex items-center justify-center transition-colors">
          <UploadCloud className="w-4 h-4 text-ink-muted group-hover:text-accent-ink transition-colors" />
        </span>
        <span className="text-ink-muted">
          {isUploading ? `Indexing... (${status})` : 'Add a document'}
        </span>
      </button>
      <input
        ref={inputRef}
        type="file"
        accept="application/pdf"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          e.target.value = '';
          onFile(file);
        }}
      />
      {error && <p className="text-xs text-accent-ink mt-2">{error}</p>}
    </div>
  );
}
