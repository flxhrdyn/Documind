import { API_BASE_URL } from './apiClient';
import type { Citation } from '../types';

export interface StreamHandlers {
  onStep?: (step: string) => void;
  onThinking?: (content: string) => void;
  onToken?: (content: string) => void;
  onDone?: (payload: { answer: string; sources: Citation[]; thoughts?: string }) => void;
  onError?: (message: string) => void;
}

export async function streamQuery(
  question: string,
  history: string[],
  handlers: StreamHandlers,
): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/query/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, history }),
  });
  if (!res.ok || !res.body) {
    handlers.onError?.(`Request failed (${res.status})`);
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const events = buffer.split('\n\n');
    buffer = events.pop() ?? '';
    for (const evt of events) {
      const line = evt.trim();
      if (!line.startsWith('data:')) continue;
      let data: Record<string, unknown>;
      try {
        data = JSON.parse(line.slice(5).trim());
      } catch {
        continue;
      }
      const step = String(data.step ?? '');
      handlers.onStep?.(step);
      if (step === 'thinking') handlers.onThinking?.(String(data.content ?? ''));
      else if (step === 'token') handlers.onToken?.(String(data.content ?? ''));
      else if (step === 'done')
        handlers.onDone?.({
          answer: String(data.answer ?? ''),
          sources: (data.sources as Citation[]) ?? [],
          thoughts: data.thoughts as string | undefined,
        });
      else if (step === 'error') handlers.onError?.(String(data.message ?? 'Unknown error'));
    }
  }
}
