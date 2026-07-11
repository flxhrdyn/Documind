import { useCallback, useEffect, useRef, useState } from 'react';
import { streamQuery } from '../lib/sse';
import type { ChatMessage } from '../types';

const STORAGE_KEY = 'invenioai_chat';

function load(): ChatMessage[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as ChatMessage[]) : [];
  } catch {
    return [];
  }
}

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>(load);
  const [isGenerating, setIsGenerating] = useState(false);
  const messagesRef = useRef(messages);
  messagesRef.current = messages;

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
  }, [messages]);

  const runQuery = useCallback(async (trimmed: string, assistantId: string, historyBefore: string[]) => {
    setIsGenerating(true);
    const patch = (fn: (m: ChatMessage) => ChatMessage) =>
      setMessages((prev) => prev.map((m) => (m.id === assistantId ? fn(m) : m)));

    try {
      await streamQuery(trimmed, historyBefore, {
        onToken: (c) => patch((m) => ({ ...m, content: m.content + c, isError: false })),
        onDone: (p) =>
          patch((m) => ({ ...m, content: p.answer, sources: p.sources, thoughts: p.thoughts, isError: false })),
        onError: (msg) => patch((m) => ({ ...m, content: msg, isError: true })),
      });
    } catch {
      patch((m) => ({
        ...m,
        content: 'We couldn’t reach the server. Check your connection and try again.',
        isError: true,
      }));
    } finally {
      setIsGenerating(false);
    }
  }, []);

  const send = useCallback(async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || isGenerating) return;

    const now = () => new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const userMsg: ChatMessage = { id: `u-${Date.now()}`, role: 'user', content: trimmed, timestamp: now() };
    const assistantId = `a-${Date.now()}`;
    const assistantMsg: ChatMessage = { id: assistantId, role: 'assistant', content: '', timestamp: now() };

    const history = messagesRef.current.map((m) => `${m.role}: ${m.content}`);
    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    await runQuery(trimmed, assistantId, history);
  }, [isGenerating, runQuery]);

  const retry = useCallback(async (assistantId: string) => {
    if (isGenerating) return;
    const idx = messagesRef.current.findIndex((m) => m.id === assistantId);
    if (idx < 1) return;
    const userMsg = messagesRef.current[idx - 1];
    if (userMsg.role !== 'user') return;

    const history = messagesRef.current.slice(0, idx - 1).map((m) => `${m.role}: ${m.content}`);
    setMessages((prev) => prev.map((m) => (m.id === assistantId ? { ...m, content: '', isError: false } : m)));
    await runQuery(userMsg.content, assistantId, history);
  }, [isGenerating, runQuery]);

  const clear = useCallback(() => setMessages([]), []);

  return { messages, isGenerating, send, retry, clear };
}
