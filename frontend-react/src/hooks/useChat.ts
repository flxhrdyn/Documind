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

  const send = useCallback(async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || isGenerating) return;

    const now = () => new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const userMsg: ChatMessage = { id: `u-${Date.now()}`, role: 'user', content: trimmed, timestamp: now() };
    const assistantId = `a-${Date.now()}`;
    const assistantMsg: ChatMessage = { id: assistantId, role: 'assistant', content: '', timestamp: now() };

    const history = messagesRef.current.map((m) => `${m.role}: ${m.content}`);
    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setIsGenerating(true);

    const patch = (fn: (m: ChatMessage) => ChatMessage) =>
      setMessages((prev) => prev.map((m) => (m.id === assistantId ? fn(m) : m)));

    try {
      await streamQuery(trimmed, history, {
        onToken: (c) => patch((m) => ({ ...m, content: m.content + c })),
        onDone: (p) =>
          patch((m) => ({ ...m, content: p.answer, sources: p.sources, thoughts: p.thoughts })),
        onError: (msg) => patch((m) => ({ ...m, content: `Error: ${msg}` })),
      });
    } finally {
      setIsGenerating(false);
    }
  }, [isGenerating]);

  const clear = useCallback(() => setMessages([]), []);

  return { messages, isGenerating, send, clear };
}
