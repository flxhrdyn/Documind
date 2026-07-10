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
      <h2 className="font-display text-sm font-semibold mb-2">Upload PDF</h2>
      <button
        onClick={() => inputRef.current?.click()}
        disabled={isUploading}
        className="w-full flex items-center justify-center gap-2 border border-dashed border-line rounded-xl py-4 text-sm text-charcoal-muted hover:bg-cream-muted disabled:opacity-60"
      >
        <UploadCloud className="w-4 h-4" />
        {isUploading ? `Indexing... (${status})` : 'Add a document'}
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
      {error && <p className="text-xs text-accent mt-2">{error}</p>}
    </div>
  );
}
