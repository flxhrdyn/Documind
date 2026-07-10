import { useState } from 'react';
import { Brain, ChevronDown } from 'lucide-react';
import type { ChatMessage as Msg } from '../types';

export default function ChatMessage({ message }: { message: Msg }) {
  const [showThoughts, setShowThoughts] = useState(false);
  const isUser = message.role === 'user';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-2xl rounded-2xl px-4 py-3 ${
          isUser ? 'bg-accent text-accent-fg' : 'bg-cream-card border border-line'
        }`}
      >
        {!isUser && message.thoughts && (
          <div className="mb-2">
            <button
              onClick={() => setShowThoughts((v) => !v)}
              className="flex items-center gap-1.5 text-xs text-charcoal-muted hover:text-charcoal"
            >
              <Brain className="w-3.5 h-3.5" /> Thought Process
              <ChevronDown className={`w-3.5 h-3.5 transition-transform ${showThoughts ? 'rotate-180' : ''}`} />
            </button>
            {showThoughts && (
              <pre className="mt-2 text-xs whitespace-pre-wrap text-charcoal-muted bg-cream-muted rounded-lg p-3">
                {message.thoughts}
              </pre>
            )}
          </div>
        )}
        <p className="whitespace-pre-wrap text-sm leading-relaxed">{message.content}</p>
      </div>
    </div>
  );
}
