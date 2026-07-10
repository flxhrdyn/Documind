export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
).replace(/\/+$/, '');

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers || {}) },
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export function formatError(e: unknown): string {
  const msg = e instanceof Error ? e.message : String(e);
  if (/rate_?limit|quota|429/i.test(msg)) {
    return 'Groq API rate limit exceeded. Wait a moment and try again.';
  }
  if (/failed to fetch|networkerror/i.test(msg)) {
    return `Could not reach the backend at ${API_BASE_URL}.`;
  }
  return msg;
}
