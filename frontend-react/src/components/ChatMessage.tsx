import { useState } from 'react';
import { Brain, ChevronDown, AlertCircle, RotateCcw } from 'lucide-react';
import type { ChatMessage as Msg, Citation } from '../types';

function focusSource(citation: Citation) {
  const el = document.getElementById(`source-card-${citation.file}-${citation.page}`);
  if (!el) return;
  el.scrollIntoView({ behavior: 'smooth', block: 'center' });
  el.classList.add('ring-2', 'ring-accent', 'border-accent');
  window.setTimeout(() => {
    el.classList.remove('ring-2', 'ring-accent', 'border-accent');
  }, 1500);
}

function AnswerText({ content, sources }: { content: string; sources?: Citation[] }) {
  if (!sources?.length) return <>{content}</>;

  const parts = content.split(/(\[\d+\])/g);
  return (
    <>
      {parts.map((part, idx) => {
        const match = part.match(/^\[(\d+)\]$/);
        if (!match) return <span key={idx}>{part}</span>;
        const n = Number(match[1]);
        const citation = sources[n - 1];
        if (!citation) return <span key={idx}>{part}</span>;
        return (
          <button
            key={idx}
            type="button"
            onClick={() => focusSource(citation)}
            className="inline-flex items-center justify-center align-super mx-0.5 min-w-[1.1rem] h-[1.1rem] px-1 rounded text-[10px] font-mono font-bold bg-accent text-accent-fg hover:bg-accent-soft transition-colors"
            aria-label={`View source ${n}: ${citation.file}`}
          >
            {n}
          </button>
        );
      })}
    </>
  );
}

export default function ChatMessage({
  message, onRetry,
}: { message: Msg; onRetry: (assistantId: string) => void }) {
  const [showThoughts, setShowThoughts] = useState(false);
  const isUser = message.role === 'user';

  if (message.isError) {
    return (
      <div className="flex flex-col items-start motion-safe:animate-[message-in_0.25s_ease-out]">
        <div className="max-w-[85%] md:max-w-[75%] px-4 py-3 rounded-2xl text-sm border border-rose-500/30 bg-rose-500/10 text-rose-500">
          <div className="flex items-start gap-2">
            <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
            <div className="min-w-0">
              <p className="font-medium">Couldn&rsquo;t get an answer</p>
              <p className="text-xs text-rose-500/80 mt-0.5 break-words">{message.content}</p>
              <button
                onClick={() => onRetry(message.id)}
                className="inline-flex items-center gap-1.5 mt-2.5 px-2.5 py-1 rounded-lg text-xs font-medium bg-rose-500/15 hover:bg-rose-500/25 transition-colors"
              >
                <RotateCcw className="w-3 h-3" />
                Try again
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

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

        <p className="whitespace-pre-wrap break-words">
          <AnswerText content={message.content} sources={message.sources} />
        </p>
      </div>
    </div>
  );
}
