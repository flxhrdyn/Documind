import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import * as sse from '../lib/sse';
import { useChat } from './useChat';

beforeEach(() => localStorage.clear());
afterEach(() => vi.restoreAllMocks());

describe('useChat', () => {
  it('appends user + assistant messages and persists to localStorage', async () => {
    vi.spyOn(sse, 'streamQuery').mockImplementation(async (_q, _h, handlers) => {
      handlers.onToken?.('Hi');
      handlers.onDone?.({ answer: 'Hi there', sources: [], thoughts: undefined });
    });

    const { result } = renderHook(() => useChat());
    await act(async () => {
      await result.current.send('hello');
    });

    await waitFor(() => expect(result.current.messages.length).toBe(2));
    expect(result.current.messages[0].role).toBe('user');
    expect(result.current.messages[1].content).toBe('Hi there');
    expect(localStorage.getItem('invenioai_chat')).toContain('Hi there');
  });
});
