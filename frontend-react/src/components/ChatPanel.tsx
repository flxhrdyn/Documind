import { useEffect, useRef, useState } from 'react';
import { Send, Sparkles, AlertCircle, Loader2 } from 'lucide-react';
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
  }, [messages, isGenerating]);

  const submit = () => {
    const text = input.trim();
    if (!text || isGenerating) return;
    onSend(text);
    setInput('');
  };

  const isLastAssistantStreaming =
    isGenerating && messages.length > 0 && messages[messages.length - 1].role === 'assistant' &&
    messages[messages.length - 1].content === '';

  return (
    <div className="h-full flex flex-col bg-bg">
      <div className="flex-1 overflow-y-auto px-4 py-6 md:px-8 space-y-8">
        {messages.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center py-10 text-center max-w-xl mx-auto">
            <div className="p-4 bg-accent/10 rounded-full border border-accent/20 mb-6">
              <Sparkles className="w-8 h-8 text-accent-ink" />
            </div>
            <h2 className="text-lg font-display font-medium text-ink">Document Q&amp;A</h2>
            <p className="text-xs text-ink-muted mt-2 leading-relaxed max-w-md">
              Ask questions about your uploaded PDF documents. The assistant searches through the
              content and provides answers based directly on the text.
            </p>
            {!hasDocuments && (
              <div className="mt-4 p-3 rounded-xl bg-accent/5 border border-accent/20 flex items-center gap-2 text-xs text-accent-ink">
                <AlertCircle className="w-4 h-4 shrink-0" />
                <span>No documents added yet. Upload a PDF in the sidebar to get started.</span>
              </div>
            )}
          </div>
        )}

        {messages.length > 0 && (
          <div className="max-w-3xl mx-auto space-y-6">
            {messages.map((m) => (
              <ChatMessageView key={m.id} message={m} />
            ))}

            {isLastAssistantStreaming && (
              <div className="flex flex-col items-start">
                <div className="max-w-[75%] px-0 py-2">
                  <div className="flex items-center gap-2 text-[11px] text-ink-muted mb-2">
                    <Loader2 className="w-3.5 h-3.5 animate-spin text-accent-ink" />
                    <span>Thinking...</span>
                  </div>
                  <div className="flex gap-1 py-1 px-3 bg-surface border border-line rounded-full">
                    <span className="w-1.5 h-1.5 bg-accent rounded-full animate-bounce [animation-delay:0ms]" />
                    <span className="w-1.5 h-1.5 bg-accent rounded-full animate-bounce [animation-delay:150ms]" />
                    <span className="w-1.5 h-1.5 bg-accent rounded-full animate-bounce [animation-delay:300ms]" />
                  </div>
                </div>
              </div>
            )}
            <div ref={endRef} />
          </div>
        )}
      </div>

      <div className="p-4 md:p-6 border-t border-line bg-surface/40">
        <div className="max-w-3xl mx-auto relative flex items-end">
          <div className="w-full relative rounded-2xl border border-line bg-surface focus-within:border-accent focus-within:ring-1 focus-within:ring-accent/40 transition-all duration-200">
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
              placeholder={
                hasDocuments
                  ? 'Ask a question about the documents...'
                  : 'Upload a PDF in the sidebar to begin...'
              }
              className="w-full resize-none bg-transparent py-4 pl-4 pr-12 text-sm text-ink placeholder:text-ink-muted focus:outline-none min-h-[52px] max-h-40 leading-relaxed"
            />
            <div className="absolute right-3 bottom-3">
              <button
                onClick={submit}
                disabled={!input.trim() || isGenerating}
                className="p-1.5 bg-accent hover:bg-accent-soft disabled:bg-surface-2 text-accent-fg disabled:text-ink-muted rounded-xl transition-all duration-150 disabled:cursor-not-allowed flex items-center justify-center"
                aria-label="Send query"
              >
                <Send className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
        <div className="max-w-3xl mx-auto text-center mt-2.5 text-[11px] text-ink-muted">
          <span>
            Press <kbd className="px-1.5 py-0.5 bg-surface-2 rounded text-[10px]">Enter</kbd> to ask,{' '}
            <kbd className="px-1.5 py-0.5 bg-surface-2 rounded text-[10px]">Shift + Enter</kbd> for a new line
          </span>
        </div>
      </div>
    </div>
  );
}
