import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiFetch } from '../lib/apiClient';
import type { DocumentsResponse } from '../types';

export function useDocuments() {
  return useQuery({
    queryKey: ['documents'],
    queryFn: () => apiFetch<DocumentsResponse>('/documents'),
  });
}

export function useDeleteDocument() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (filename: string) =>
      apiFetch<unknown>(`/documents/delete?filename=${encodeURIComponent(filename)}`, {
        method: 'DELETE',
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['documents'] }),
  });
}

export function useDeleteAllDocuments() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => apiFetch<unknown>('/documents', { method: 'DELETE' }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['documents'] });
      qc.invalidateQueries({ queryKey: ['metrics'] });
    },
  });
}
