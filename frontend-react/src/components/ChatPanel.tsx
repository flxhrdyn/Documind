import { useEffect, useRef, useState } from 'react';
import { Send } from 'lucide-react';
import ChatMessageView from './ChatMessage';
import type { ChatMessage } from '../types';

interface Props {
  messages: ChatMessage[];
  isGenerating: boolean;
  hasDocuments: boolean;
  onSend: (text: string) => void;
}

export default function ChatPanel({ messages, isGenerating, hasDocuments, onSend }: Props) {
  const [input, setInput] = useState('');
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const submit = () => {
    const text = input.trim();
    if (!text || isGenerating) return;
    onSend(text);
    setInput('');
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {messages.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center text-center">
            <h2 className="font-display text-4xl font-bold text-accent">InvenioAI</h2>
            <p className="text-charcoal-muted mt-2">Ask anything about your knowledge base.</p>
          </div>
        )}
        {messages.map((m) => (
          <ChatMessageView key={m.id} message={m} />
        ))}
        <div ref={endRef} />
      </div>
      <div className="border-t border-line p-4">
        {!hasDocuments && (
          <p className="text-xs text-accent mb-2">Upload a PDF first to start asking questions.</p>
        )}
        <div className="flex items-end gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            rows={1}
            placeholder="Ask something about your documents..."
            className="flex-1 resize-none rounded-xl border border-line bg-cream-card px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-accent/40"
          />
          <button
            onClick={submit}
            disabled={isGenerating}
            className="p-3 rounded-xl bg-accent text-accent-fg hover:bg-accent-soft disabled:opacity-60"
            aria-label="Send"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
