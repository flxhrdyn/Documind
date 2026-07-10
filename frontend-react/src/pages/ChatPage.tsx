import { useChat } from '../hooks/useChat';
import { useDocuments } from '../hooks/useDocuments';
import ChatPanel from '../components/ChatPanel';
import SourcesPanel from '../components/SourcesPanel';

export default function ChatPage() {
  const { messages, isGenerating, send } = useChat();
  const { data } = useDocuments();

  const lastAssistant = [...messages].reverse().find(
    (m) => m.role === 'assistant' && m.sources && m.sources.length > 0,
  );

  return (
    <div className="flex h-screen">
      <div className="flex-1 min-w-0">
        <ChatPanel
          messages={messages}
          isGenerating={isGenerating}
          hasDocuments={(data?.count ?? 0) > 0}
          onSend={send}
        />
      </div>
      <div className="w-96 shrink-0">
        <SourcesPanel citations={lastAssistant?.sources ?? []} />
      </div>
    </div>
  );
}
