/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef, useEffect } from 'react';
import { Send, Sparkles, AlertCircle, Loader2, Play } from 'lucide-react';
import { ChatMessage, Citation } from '../types';

interface ChatPanelProps {
  chatHistory: ChatMessage[];
  isGenerating: boolean;
  onSendMessage: (content: string) => void;
  onCitationClick: (id: string) => void;
  activeCitationId: string | null;
  documentsCount: number;
}

export default function ChatPanel({
  chatHistory,
  isGenerating,
  onSendMessage,
  onCitationClick,
  activeCitationId,
  documentsCount
}: ChatPanelProps) {
  const [inputValue, setInputValue] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const sampleQueries = [
    { text: 'How does InvenioAI combine dense and sparse document retrieval?', label: 'Hybrid Search' },
    { text: 'What is the self-attention mechanism in the Transformer paper?', label: 'Transformer' },
    { text: 'What percentage of carbon emissions did Acme Corp reduce in FY2026?', label: 'Sustainability' },
    { text: 'How does the dual-layer semantic cache operate at the edge?', label: 'Performance' },
  ];

  // Auto-scroll to bottom of conversation
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory, isGenerating]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isGenerating) return;
    onSendMessage(inputValue);
    setInputValue('');
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  // Custom regex parser to render bold text and interactive citation links beautifully
  const renderFormattedContent = (content: string, citations: Citation[] = []) => {
    if (!content) return null;

    // First, split lines to handle paragraph breaks
    const lines = content.split('\n');

    return lines.map((line, lineIdx) => {
      // Process bold markers "**text**"
      let match;
      const citationRegex = /\[(\d+)\]/g;

      // A simple tokenizer to handle bold markers AND citation badges inside each line
      const boldRegex = /\*\*([^*]+)\*\*/g;

      // Check if there are citations
      let subParts: { text: string; isCitation: boolean; citationId?: string }[] = [];
      
      let lastMatchEnd = 0;
      while ((match = citationRegex.exec(line)) !== null) {
        // Add preceding text
        if (match.index > lastMatchEnd) {
          subParts.push({ text: line.substring(lastMatchEnd, match.index), isCitation: false });
        }
        subParts.push({ text: match[0], isCitation: true, citationId: match[1] });
        lastMatchEnd = citationRegex.lastIndex;
      }
      if (lastMatchEnd < line.length) {
        subParts.push({ text: line.substring(lastMatchEnd), isCitation: false });
      }

      if (subParts.length === 0) {
        subParts.push({ text: line, isCitation: false });
      }

      const parsedElements = subParts.map((part, partIdx) => {
        if (part.isCitation && part.citationId) {
          const hasMatchingCitation = citations.some(c => c.id === part.citationId);
          const isHighlighted = activeCitationId === part.citationId;
          
          return (
            <button
              key={`cite-${partIdx}`}
              onClick={() => onCitationClick(part.citationId!)}
              className={`inline-flex items-center justify-center font-mono text-[10.5px] font-bold px-1.5 py-0.5 mx-0.5 rounded-md border cursor-pointer transition-all duration-150 ${
                isHighlighted
                  ? 'bg-teal-600 text-white border-teal-700 ring-2 ring-teal-500/30'
                  : hasMatchingCitation
                  ? 'bg-teal-50 text-teal-700 hover:bg-teal-600 hover:text-white border-teal-200/60 dark:bg-teal-950/20 dark:text-teal-400 dark:border-teal-900/50 dark:hover:bg-teal-600 dark:hover:text-white'
                  : 'bg-gray-100 text-gray-500 border-gray-200 dark:bg-gray-900 dark:text-gray-400 dark:border-gray-800'
              }`}
              title={`View source excerpt [${part.citationId}]`}
            >
              {part.text}
            </button>
          );
        } else {
          // Render bold text in non-citation strings
          // Split by bold regex
          const boldParts = part.text.split(boldRegex);
          return boldParts.map((bText, bIdx) => {
            // odd indexes are matched bold segments
            if (bIdx % 2 === 1) {
              return <strong key={`b-${bIdx}`} className="font-semibold text-gray-900 dark:text-gray-50">{bText}</strong>;
            }
            return <span key={`s-${bIdx}`}>{bText}</span>;
          });
        }
      });

      return (
        <p key={lineIdx} className="min-h-[1.25rem] mb-2 last:mb-0 leading-relaxed text-gray-700 dark:text-gray-300 font-sans">
          {parsedElements}
        </p>
      );
    });
  };

  return (
    <div className="h-full flex flex-col bg-gray-50/20 dark:bg-gray-950/5">
      {/* Messages Feed Viewport */}
      <div className="flex-1 overflow-y-auto px-4 py-6 md:px-8 space-y-8">
        {chatHistory.length === 0 ? (
          /* Empty Chat state */
          <div className="h-full flex flex-col items-center justify-center py-10 text-center max-w-xl mx-auto animate-in fade-in duration-300">
            <div className="p-4 bg-teal-50 dark:bg-teal-950/30 rounded-full border border-teal-100/60 dark:border-teal-900/40 mb-6 animate-pulse">
              <Sparkles className="w-8 h-8 text-teal-600 dark:text-teal-400" />
            </div>
            
            <h2 className="text-lg font-display font-medium text-gray-900 dark:text-gray-100">
              Document Q&A
            </h2>
            
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-2 leading-relaxed font-sans max-w-md">
              Ask questions about your uploaded PDF documents. The assistant searches through the content and provides answers based directly on the text.
            </p>

            {documentsCount === 0 && (
              <div className="mt-4 p-3 rounded-xl bg-teal-50/30 dark:bg-teal-950/10 border border-teal-200/30 dark:border-teal-900/30 flex items-center space-x-2 text-xs text-teal-800 dark:text-teal-400">
                <AlertCircle className="w-4 h-4 shrink-0 text-teal-600 dark:text-teal-400" />
                <span>No documents added yet. Upload files in the sidebar or ask questions about the preloaded templates below.</span>
              </div>
            )}

            {/* Quick Starter Queries Grid */}
            <div className="mt-8 w-full space-y-2">
              <span className="text-[10px] font-semibold text-gray-400 dark:text-gray-500 uppercase tracking-wider block mb-2">
                Sample Questions
              </span>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-left">
                {sampleQueries.map((q, idx) => (
                  <button
                    key={idx}
                    onClick={() => {
                      if (!isGenerating) onSendMessage(q.text);
                    }}
                    disabled={isGenerating}
                    className="p-3 text-xs bg-white dark:bg-gray-900 hover:border-teal-500/40 dark:hover:border-teal-500/40 hover:bg-gray-50/40 dark:hover:bg-gray-800/20 border border-gray-100 dark:border-gray-800/80 rounded-xl text-gray-700 dark:text-gray-300 transition-all duration-200 shadow-xs flex items-start space-x-2 group cursor-pointer disabled:opacity-50"
                  >
                    <Play className="w-3 h-3 text-teal-600 dark:text-teal-400 mt-0.5 shrink-0 opacity-40 group-hover:opacity-100 transition-opacity duration-150" />
                    <div>
                      <span className="font-medium group-hover:text-gray-900 dark:group-hover:text-gray-50 transition-colors duration-150">
                        {q.text}
                      </span>
                      <span className="block text-[10px] text-gray-400 dark:text-gray-500 mt-0.5">
                        {q.label}
                      </span>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          /* Active Chat Conversation Feed */
          <div className="max-w-2.5xl mx-auto space-y-6">
            {chatHistory.map((message) => {
              const isUser = message.role === 'user';
              
              return (
                <div
                  key={message.id}
                  className={`flex flex-col ${isUser ? 'items-end' : 'items-start'} group`}
                >
                  {/* Message Bubble Container */}
                  <div
                    className={`max-w-[85%] md:max-w-[75%] px-5 py-4 rounded-2xl text-sm leading-relaxed transition-all duration-200 shadow-xs border ${
                      isUser
                        ? 'bg-white text-gray-800 border-gray-100 dark:bg-gray-900 dark:text-gray-100 dark:border-gray-800/80 rounded-tr-sm'
                        : 'bg-transparent text-gray-800 dark:text-gray-100 border-none px-0 py-2 shadow-none'
                    }`}
                  >
                    {/* Header showing sender and time */}
                    <div className="flex items-center justify-between mb-1.5 text-[10px] text-gray-400 dark:text-gray-500">
                      <span className="font-semibold">{isUser ? 'You' : 'Assistant'}</span>
                      <span>{message.timestamp}</span>
                    </div>

                    {/* Actual message text formatted */}
                    <div className="max-w-[65ch] font-sans break-words">
                      {isUser ? (
                        <p className="text-gray-800 dark:text-gray-100 font-medium">{message.content}</p>
                      ) : (
                        renderFormattedContent(message.content, message.citations)
                      )}
                    </div>

                    {/* Assistant metadata inline */}
                    {!isUser && message.metrics && (
                      <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-gray-100/80 dark:border-gray-800/40 pt-2.5 text-[10px] text-gray-400 dark:text-gray-500">
                        <span>Search: <strong>{message.metrics.retrievalTimeMs}ms</strong></span>
                        <span className="opacity-40">•</span>
                        <span>Answer: <strong>{message.metrics.generationTimeMs}ms</strong></span>
                        <span className="opacity-40">•</span>
                        <span>References: <strong>{message.metrics.chunksRetrieved} {message.metrics.chunksRetrieved === 1 ? 'chunk' : 'chunks'}</strong></span>
                        {message.metrics.cacheHit && (
                          <>
                            <span className="opacity-40">•</span>
                            <span className="px-1.5 py-0.5 bg-emerald-50 dark:bg-emerald-950/30 text-emerald-600 dark:text-emerald-400 rounded font-medium text-[9px]">
                              Cached
                            </span>
                          </>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}

            {/* Simulated generation loading dot stream */}
            {isGenerating && (
              <div className="flex flex-col items-start">
                <div className="max-w-[75%] px-0 py-2 bg-transparent text-gray-800 dark:text-gray-100">
                  <div className="flex items-center space-x-2 text-[11px] text-gray-500 dark:text-gray-400 mb-2">
                    <Loader2 className="w-3.5 h-3.5 animate-spin text-teal-600 dark:text-teal-400" />
                    <span>Thinking...</span>
                  </div>
                  
                  <div className="flex space-x-1 py-1 px-3 bg-white dark:bg-gray-900 border border-gray-100 dark:border-gray-800 rounded-full shadow-xs">
                    <span className="w-1.5 h-1.5 bg-teal-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                    <span className="w-1.5 h-1.5 bg-teal-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                    <span className="w-1.5 h-1.5 bg-teal-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Message Input Form Panel */}
      <div className="p-4 md:p-6 border-t border-gray-100 dark:border-gray-900/80 bg-white/40 dark:bg-gray-950/10 backdrop-blur-md">
        <form onSubmit={handleSubmit} className="max-w-2.5xl mx-auto relative flex items-end">
          <div className="w-full relative rounded-2xl border border-gray-200/80 bg-white shadow-sm dark:border-gray-800/80 dark:bg-gray-900 focus-within:border-teal-500 dark:focus-within:border-teal-500 focus-within:ring-1 focus-within:ring-teal-500/40 transition-all duration-200">
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                documentsCount === 0
                  ? "Upload a PDF in the sidebar to begin..."
                  : "Ask a question about the documents..."
              }
              rows={1}
              className="w-full resize-none bg-transparent py-4 pl-4 pr-12 text-sm text-gray-800 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none min-h-[52px] max-h-[160px] leading-relaxed"
              style={{ height: 'auto' }}
            />
            
            <div className="absolute right-3 bottom-3 flex items-center space-x-2">
              <button
                type="submit"
                disabled={!inputValue.trim() || isGenerating}
                className="p-1.5 bg-teal-600 hover:bg-teal-700 disabled:bg-gray-100 dark:disabled:bg-gray-800 text-white disabled:text-gray-400 dark:disabled:text-gray-600 rounded-xl shadow-xs transition-all duration-150 cursor-pointer disabled:cursor-not-allowed flex items-center justify-center"
                aria-label="Send query"
              >
                <Send className="w-4 h-4" />
              </button>
            </div>
          </div>
        </form>
        
        <div className="max-w-2.5xl mx-auto text-center mt-2.5 text-[11px] text-gray-450 dark:text-gray-500">
          <span>Press <kbd className="px-1.5 py-0.5 bg-gray-100 dark:bg-gray-800 rounded text-[10px] font-sans">Enter</kbd> to ask, <kbd className="px-1.5 py-0.5 bg-gray-100 dark:bg-gray-800 rounded text-[10px] font-sans">Shift + Enter</kbd> for a new line</span>
        </div>
      </div>
    </div>
  );
}
