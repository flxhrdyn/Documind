import { describe, it, expect, vi, afterEach } from 'vitest';
import { streamQuery } from './sse';

function sseResponse(lines: string[]): Response {
  const body = new ReadableStream({
    start(controller) {
      const enc = new TextEncoder();
      for (const l of lines) controller.enqueue(enc.encode(`data: ${l}\n\n`));
      controller.close();
    },
  });
  return new Response(body, { status: 200 });
}

afterEach(() => vi.restoreAllMocks());

describe('streamQuery', () => {
  it('dispatches tokens and a done payload', async () => {
    vi.stubGlobal('fetch', vi.fn(async () =>
      sseResponse([
        JSON.stringify({ step: 'token', content: 'Hello' }),
        JSON.stringify({ step: 'token', content: ' world' }),
        JSON.stringify({ step: 'done', answer: 'Hello world', sources: [] }),
      ]),
    ));

    const tokens: string[] = [];
    let done: { answer: string } | null = null;
    await streamQuery('q', [], {
      onToken: (c) => tokens.push(c),
      onDone: (p) => { done = p; },
    });

    expect(tokens).toEqual(['Hello', ' world']);
    expect(done!.answer).toBe('Hello world');
  });

  it('invokes onError on error step', async () => {
    vi.stubGlobal('fetch', vi.fn(async () =>
      sseResponse([JSON.stringify({ step: 'error', message: 'pipeline boom' })]),
    ));
    let err = '';
    await streamQuery('q', [], { onError: (m) => { err = m; } });
    expect(err).toBe('pipeline boom');
  });
});
