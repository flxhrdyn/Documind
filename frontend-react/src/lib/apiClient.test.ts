import { describe, it, expect, vi, afterEach } from 'vitest';
import { apiFetch, formatError } from './apiClient';

afterEach(() => vi.restoreAllMocks());

describe('apiFetch', () => {
  it('returns parsed JSON on 200', async () => {
    vi.stubGlobal('fetch', vi.fn(async () =>
      new Response(JSON.stringify({ ok: true }), { status: 200 }),
    ));
    await expect(apiFetch('/documents')).resolves.toEqual({ ok: true });
  });

  it('throws with backend detail on non-2xx', async () => {
    vi.stubGlobal('fetch', vi.fn(async () =>
      new Response(JSON.stringify({ detail: 'boom' }), { status: 500 }),
    ));
    await expect(apiFetch('/documents')).rejects.toThrow('boom');
  });
});

describe('formatError', () => {
  it('maps rate limit text to a friendly message', () => {
    expect(formatError(new Error('rate_limit exceeded'))).toMatch(/rate limit/i);
  });
});
