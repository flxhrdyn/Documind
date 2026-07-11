import { useState } from 'react';
import { BookOpen } from 'lucide-react';
import { useChat } from '../hooks/useChat';
import { useDocuments } from '../hooks/useDocuments';
import ChatPanel from '../components/ChatPanel';
import SourcesPanel from '../components/SourcesPanel';

export default function ChatPage() {
  const { messages, isGenerating, send, retry } = useChat();
  const { data } = useDocuments();
  const [sourcesOpen, setSourcesOpen] = useState(false);

  const lastAssistant = [...messages].reverse().find((m) => m.role === 'assistant');
  const sources = lastAssistant?.sources ?? [];

  return (
    <div className="relative flex h-full min-h-0">
      <div className="flex-1 min-w-0 h-full">
        <ChatPanel
          messages={messages}
          isGenerating={isGenerating}
          hasDocuments={(data?.count ?? 0) > 0}
          onSend={send}
          onRetry={retry}
        />
      </div>

      {sources.length > 0 && (
        <button
          onClick={() => setSourcesOpen(true)}
          className="lg:hidden fixed bottom-24 right-4 z-30 flex items-center gap-1.5 px-3 py-2 rounded-full bg-accent text-accent-fg text-xs font-semibold shadow-lg"
        >
          <BookOpen className="w-3.5 h-3.5" />
          References ({sources.length})
        </button>
      )}

      <div className="hidden lg:block w-96 shrink-0 h-full">
        <SourcesPanel citations={sources} />
      </div>

      {sourcesOpen && (
        <div className="lg:hidden fixed inset-0 z-40 flex justify-end">
          <button
            aria-label="Close references"
            onClick={() => setSourcesOpen(false)}
            className="absolute inset-0 bg-ink/40 motion-safe:animate-[fade-in_0.2s_ease-out]"
          />
          <div className="relative w-96 max-w-[90vw] h-full motion-safe:animate-[slide-in-right_0.25s_ease-out]">
            <SourcesPanel citations={sources} onClose={() => setSourcesOpen(false)} />
          </div>
        </div>
      )}
    </div>
  );
}
