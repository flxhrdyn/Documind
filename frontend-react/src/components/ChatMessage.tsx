import { useState } from 'react';
import { Brain, ChevronDown } from 'lucide-react';
import type { ChatMessage as Msg } from '../types';

export default function ChatMessage({ message }: { message: Msg }) {
  const [showThoughts, setShowThoughts] = useState(false);
  const isUser = message.role === 'user';

  return (
    <div className={`flex flex-col ${isUser ? 'items-end' : 'items-start'} motion-safe:animate-[message-in_0.25s_ease-out]`}>
      <div
        className={`max-w-[85%] md:max-w-[75%] px-5 py-4 rounded-2xl text-sm leading-relaxed border ${
          isUser
            ? 'bg-surface text-ink border-line rounded-tr-sm'
            : 'bg-transparent text-ink border-none px-0 py-2'
        }`}
      >
        <div className="flex items-center justify-between gap-4 mb-1.5 text-[10px] text-ink-muted">
          <span className="font-semibold">{isUser ? 'You' : 'Assistant'}</span>
          {message.timestamp && <span>{message.timestamp}</span>}
        </div>

        {!isUser && message.thoughts && (
          <div className="mb-3">
            <button
              onClick={() => setShowThoughts((v) => !v)}
              className="flex items-center gap-1.5 text-xs text-ink-muted hover:text-ink transition-colors"
            >
              <Brain className="w-3.5 h-3.5" /> Thought Process
              <ChevronDown className={`w-3.5 h-3.5 transition-transform ${showThoughts ? 'rotate-180' : ''}`} />
            </button>
            {showThoughts && (
              <pre className="mt-2 text-xs whitespace-pre-wrap text-ink-muted bg-surface-2 rounded-lg p-3 font-mono">
                {message.thoughts}
              </pre>
            )}
          </div>
        )}

        <p className="whitespace-pre-wrap break-words">{message.content}</p>
      </div>
    </div>
  );
}
