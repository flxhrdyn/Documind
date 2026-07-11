import { useRef, useState } from 'react';
import { Upload, AlertCircle, Loader2 } from 'lucide-react';
import { useUploadJob } from '../hooks/useUploadJob';

const STATUS_LABELS: Record<string, string> = {
  uploading: 'Uploading document...',
  pending: 'Queued for indexing...',
  running: 'Starting indexing job...',
  parsing: 'Parsing document...',
  indexing: 'Saving to knowledge base...',
};

function statusLabel(status: string | null): string {
  if (!status) return 'Processing...';
  return STATUS_LABELS[status] ?? 'Processing...';
}

const MAX_UPLOAD_MB = 15;

export default function UploadPanel() {
  const { upload, status, isUploading } = useUploadJob();
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onFile = async (file: File | undefined) => {
    if (!file) return;
    if (file.type !== 'application/pdf' && !file.name.toLowerCase().endsWith('.pdf')) {
      setError('Only PDF document files are supported by the InvenioAI indexing engine.');
      return;
    }
    if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
      setError(`"${file.name}" is ${(file.size / (1024 * 1024)).toFixed(1)}MB, over the ${MAX_UPLOAD_MB}MB limit. Try a smaller file.`);
      return;
    }
    setError(null);
    try {
      await upload(file);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Upload failed');
    }
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold font-display text-ink">Upload Document</h2>
        <span className="text-xs text-ink-muted">Max {MAX_UPLOAD_MB}MB</span>
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setIsDragging(false);
          onFile(e.dataTransfer.files?.[0]);
        }}
        onClick={() => !isUploading && inputRef.current?.click()}
        className={`group relative flex flex-col items-center justify-center border border-dashed rounded-xl p-6 text-center cursor-pointer transition-all duration-300 ${
          isDragging
            ? 'border-accent bg-accent/10'
            : 'border-line hover:border-accent/60 bg-surface-2/30 hover:bg-surface-2/60'
        } ${isUploading ? 'opacity-70 pointer-events-none' : ''}`}
      >
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
        <div className="p-3 bg-surface-2 rounded-xl mb-3 shadow-sm border border-line group-hover:scale-105 transition-transform duration-200">
          {isUploading ? (
            <Loader2 className="w-5 h-5 text-accent animate-spin" />
          ) : (
            <Upload className="w-5 h-5 text-ink-muted group-hover:text-accent transition-colors duration-200" />
          )}
        </div>
        {isUploading ? (
          <p className="text-sm font-medium text-ink-muted">{statusLabel(status)}</p>
        ) : (
          <>
            <p className="text-sm font-medium text-ink">
              Drag &amp; drop PDF here, or <span className="text-accent-ink hover:underline">browse file</span>
            </p>
            <p className="text-xs text-ink-muted mt-1">Supports standard PDF files</p>
          </>
        )}
      </div>

      {error && (
        <div className="flex items-start gap-2 p-3 rounded-lg bg-rose-500/10 text-rose-500 border border-rose-500/20">
          <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
          <span className="text-xs font-medium">{error}</span>
        </div>
      )}
    </div>
  );
}
