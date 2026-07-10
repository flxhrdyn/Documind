import { useCallback, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { API_BASE_URL, apiFetch, formatError } from '../lib/apiClient';
import type { UploadJob } from '../types';

export function nextPollInterval(current: number): number {
  return Math.min(current * 2, 5000);
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function createUploadJob(file: File): Promise<string> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${API_BASE_URL}/upload/jobs`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(`Upload failed (${res.status})`);
  const body = await res.json();
  if (!body.job_id) throw new Error('Missing job id from backend');
  return String(body.job_id);
}

function fetchJob(jobId: string): Promise<UploadJob> {
  return apiFetch<UploadJob>(`/upload/jobs/${jobId}`);
}

export function useUploadJob() {
  const qc = useQueryClient();
  const [status, setStatus] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  const upload = useCallback(
    async (file: File) => {
      setIsUploading(true);
      setStatus('uploading');
      try {
        const jobId = await createUploadJob(file);
        let interval = 1000;
        while (true) {
          const job = await fetchJob(jobId);
          setStatus(job.status);
          if (job.status === 'succeeded') {
            qc.invalidateQueries({ queryKey: ['documents'] });
            qc.invalidateQueries({ queryKey: ['metrics'] });
            break;
          }
          if (job.status === 'failed') {
            throw new Error(job.error || 'Indexing failed');
          }
          await sleep(interval);
          interval = nextPollInterval(interval);
        }
      } catch (e) {
        setStatus('failed');
        throw new Error(formatError(e));
      } finally {
        setIsUploading(false);
      }
    },
    [qc],
  );

  return { upload, status, isUploading };
}
