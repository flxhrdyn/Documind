import { useQuery } from '@tanstack/react-query';
import { apiFetch } from '../lib/apiClient';
import type { MetricsResponse } from '../types';

export function useMetrics() {
  return useQuery({
    queryKey: ['metrics'],
    queryFn: () => apiFetch<MetricsResponse>('/metrics'),
    refetchInterval: 15000,
  });
}
