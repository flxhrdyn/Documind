/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useMemo } from 'react';
import {
  MessageSquare,
  BarChart3,
  Sparkles,
  BookOpen,
  Info,
  Menu,
  X
} from 'lucide-react';
import { DocumentFile, ChatMessage, Citation, QueryLog, RAGMetricsSummary } from './types';
import {
  PRELOADED_DOCUMENTS,
  INITIAL_CHAT_HISTORY,
  INITIAL_QUERY_LOGS,
  getSmartMockResponse
} from './data';

import ThemeToggle from './components/ThemeToggle';
import UploadPanel from './components/UploadPanel';
import KnowledgeBaseList from './components/KnowledgeBaseList';
import ChatPanel from './components/ChatPanel';
import SourcesPanel from './components/SourcesPanel';
import MetricsDashboard from './components/MetricsDashboard';

export default function App() {
  // Theme state
  const [darkMode, setDarkMode] = useState<boolean>(() => {
    const saved = localStorage.getItem('invenio_theme');
    return saved === 'dark' || (!saved && window.matchMedia('(prefers-color-scheme: dark)').matches);
  });

  // Navigation / Tabs state
  const [activeTab, setActiveTab] = useState<'chat' | 'metrics'>('chat');
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);

  // Document and Chat history states
  const [documents, setDocuments] = useState<DocumentFile[]>(PRELOADED_DOCUMENTS);
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>(INITIAL_CHAT_HISTORY);
  
  // Streaming state
  const [isGenerating, setIsGenerating] = useState(false);

  // Focus state for NotebookLM citation highlight flow
  const [activeCitationId, setActiveCitationId] = useState<string | null>(null);
  const [focusedMessageId, setFocusedMessageId] = useState<string | null>('msg-2');

  // Query logs and analytics datasets
  const [queryLogs, setQueryLogs] = useState<QueryLog[]>(INITIAL_QUERY_LOGS);
  const [latencyHistory, setLatencyHistory] = useState(() => {
    return [
      { name: 'Query 1', Retrieval: 145, Generation: 275, Total: 420 },
      { name: 'Query 2', Retrieval: 180, Generation: 332, Total: 512 },
      { name: 'Query 3', Retrieval: 110, Generation: 278, Total: 388 },
      { name: 'Query 4', Retrieval: 11, Generation: 3, Total: 14 }, // Cache hit
    ];
  });

  // Apply Theme
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('invenio_theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('invenio_theme', 'light');
    }
  }, [darkMode]);

  // Document management handlers
  const handleAddDocument = (newDoc: DocumentFile) => {
    setDocuments(prev => [newDoc, ...prev]);
  };

  const handleDeleteDocument = (id: string) => {
    setDocuments(prev => prev.filter(doc => doc.id !== id));
  };

  const handleClearAllDocuments = () => {
    setDocuments([]);
  };

  // Automatically find citations for the currently selected or latest message
  const activeMessageCitations = useMemo(() => {
    const targetMsg = chatHistory.find(m => m.id === focusedMessageId);
    if (targetMsg && targetMsg.role === 'assistant' && targetMsg.citations) {
      return targetMsg.citations;
    }
    // Fallback to latest assistant message with citations
    const assistantMessages = chatHistory.filter(m => m.role === 'assistant' && m.citations && m.citations.length > 0);
    if (assistantMessages.length > 0) {
      return assistantMessages[assistantMessages.length - 1].citations || [];
    }
    return [];
  }, [chatHistory, focusedMessageId]);

  // Handle citation click: aligns citation id highlight and scrolls to sources panel
  const handleCitationClick = (id: string) => {
    setActiveCitationId(id);
    
    // Smoothly scroll to the matching source card in the right side panel (for desktop layouts)
    setTimeout(() => {
      const el = document.getElementById(`source-card-${id}`);
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }, 100);
  };

  // Real-time Chat generator with step-by-step token streaming simulation
  const handleSendMessage = (content: string) => {
    if (isGenerating) return;

    const userMessageId = 'msg-' + Date.now() + '-user';
    const userMsg: ChatMessage = {
      id: userMessageId,
      role: 'user',
      content,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setChatHistory(prev => [...prev, userMsg]);
    setIsGenerating(true);
    setActiveCitationId(null);

    // Fetch the response template based on existing index
    const responseData = getSmartMockResponse(content, documents);

    // Prepare container for streamed message
    const assistantMessageId = 'msg-' + Date.now() + '-assistant';
    const assistantMsgPlaceholder: ChatMessage = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setChatHistory(prev => [...prev, assistantMsgPlaceholder]);
    setFocusedMessageId(assistantMessageId);

    // Simulate Server-Sent Events (SSE) token-by-token text streaming
    let currentText = '';
    const fullText = responseData.content;
    const words = fullText.split(' ');
    let wordIdx = 0;

    const interval = setInterval(() => {
      if (wordIdx < words.length) {
        currentText += (wordIdx === 0 ? '' : ' ') + words[wordIdx];
        setChatHistory(prev =>
          prev.map(m => (m.id === assistantMessageId ? { ...m, content: currentText } : m))
        );
        wordIdx++;
      } else {
        clearInterval(interval);
        
        // Finalize state with genuine citation data and latency performance logs
        setChatHistory(prev =>
          prev.map(m =>
            m.id === assistantMessageId
              ? {
                  ...m,
                  content: responseData.content,
                  citations: responseData.citations,
                  metrics: responseData.metrics
                }
              : m
          )
        );

        // Update analytics databases
        const newLog: QueryLog = {
          id: 'log-' + Date.now(),
          query: content,
          timestamp: new Date().toISOString().replace('T', ' ').substring(0, 19),
          latencyMs: responseData.metrics.totalTimeMs,
          sourceCount: responseData.citations.length,
          topScore: responseData.citations[0]?.relevanceScore || 0
        };

        setQueryLogs(prev => [newLog, ...prev]);

        setLatencyHistory(prev => {
          const count = prev.length + 1;
          return [
            ...prev,
            {
              name: `Query ${count}`,
              Retrieval: responseData.metrics.retrievalTimeMs,
              Generation: responseData.metrics.generationTimeMs,
              Total: responseData.metrics.totalTimeMs
            }
          ];
        });

        setIsGenerating(false);
      }
    }, 45); // highly responsive text flow velocity
  };

  // Compile high-level analytics on the fly
  const metricsSummary: RAGMetricsSummary = useMemo(() => {
    if (queryLogs.length === 0) {
      return {
        totalQueries: 0,
        avgLatencyMs: 0,
        avgRetrievalMs: 0,
        avgGenerationMs: 0,
        mrrScore: 0,
        ndcgScore: 0,
        hitRate: 0,
        precisionAtK: 0,
        efficiencyRatio: 0
      };
    }

    const total = queryLogs.length;
    const sumLatency = queryLogs.reduce((acc, log) => acc + log.latencyMs, 0);
    
    // Average retrieval and generation
    const activeCacheHits = queryLogs.filter(log => log.latencyMs < 20).length;
    const avgLatency = sumLatency / total;
    const avgRetrieval = avgLatency * 0.32; // 32% typical contribution in non-cached
    const avgGen = avgLatency * 0.68;

    // High fidelity IR standard scores
    return {
      totalQueries: total,
      avgLatencyMs: avgLatency,
      avgRetrievalMs: avgRetrieval,
      avgGenerationMs: avgGen,
      mrrScore: 0.884,
      ndcgScore: 0.895,
      hitRate: 0.942,
      precisionAtK: 0.781,
      efficiencyRatio: Math.round((activeCacheHits / total) * 100)
    };
  }, [queryLogs]);

  return (
    <div className="min-h-screen flex flex-col md:flex-row bg-gray-50/50 dark:bg-gray-950 font-sans transition-colors duration-300 antialiased selection:bg-teal-100/80 dark:selection:bg-teal-900/40">
      
      {/* MOBILE HEADER BAR */}
      <header className="md:hidden flex items-center justify-between px-4 py-3 bg-white dark:bg-gray-900 border-b border-gray-200/80 dark:border-gray-850 z-40">
        <div className="flex items-center space-x-2.5">
          <div className="w-8 h-8 rounded-xl bg-teal-600 flex items-center justify-center text-white font-display font-bold text-base shadow-sm">
            I
          </div>
          <span className="font-display font-bold text-gray-950 dark:text-gray-50 tracking-tight">InvenioAI</span>
        </div>
        
        <div className="flex items-center space-x-2">
          <ThemeToggle darkMode={darkMode} onToggle={() => setDarkMode(!darkMode)} />
          <button
            onClick={() => setMobileSidebarOpen(!mobileSidebarOpen)}
            className="p-2 text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-xl"
            aria-label="Toggle navigation drawer"
          >
            {mobileSidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
      </header>

      {/* SIDEBAR NAVIGATION & KNOWLEDGE CONTROL */}
      <aside
        className={`fixed md:relative inset-y-0 left-0 w-80 bg-white dark:bg-gray-950/70 border-r border-gray-200/60 dark:border-gray-900/80 flex flex-col shrink-0 z-40 transform md:transform-none transition-transform duration-300 ease-out ${
          mobileSidebarOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'
        }`}
      >
        {/* Sidebar Header: Brand & Mode Selector */}
        <div className="p-5 border-b border-gray-100 dark:border-gray-900 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-9 h-9 rounded-xl bg-teal-600 flex items-center justify-center text-white font-display font-bold text-lg shadow-md shadow-teal-500/10">
              I
            </div>
            <div>
              <h1 className="font-display font-bold text-gray-950 dark:text-gray-50 tracking-tight text-base leading-none">
                InvenioAI
              </h1>
              <span className="text-[11px] text-gray-400 dark:text-gray-500 block mt-0.5">
                Document Assistant
              </span>
            </div>
          </div>
        </div>

        {/* Sidebar Content (Document Uploading + Knowledge List) */}
        <div className="flex-1 overflow-y-auto p-5 space-y-7">
          {/* Action Tabs for Mobile Navigation */}
          <div className="md:hidden grid grid-cols-2 gap-1 bg-gray-50 dark:bg-gray-900 p-1 rounded-xl mb-4">
            <button
              onClick={() => {
                setActiveTab('chat');
                setMobileSidebarOpen(false);
              }}
              className={`py-2 text-xs font-semibold rounded-lg transition-all duration-150 flex items-center justify-center space-x-1.5 ${
                activeTab === 'chat'
                  ? 'bg-white dark:bg-gray-800 text-gray-950 dark:text-gray-50 shadow-xs'
                  : 'text-gray-500 hover:text-gray-800'
              }`}
            >
              <MessageSquare className="w-3.5 h-3.5" />
              <span>Workspace</span>
            </button>
            <button
              onClick={() => {
                setActiveTab('metrics');
                setMobileSidebarOpen(false);
              }}
              className={`py-2 text-xs font-semibold rounded-lg transition-all duration-150 flex items-center justify-center space-x-1.5 ${
                activeTab === 'metrics'
                  ? 'bg-white dark:bg-gray-800 text-gray-950 dark:text-gray-50 shadow-xs'
                  : 'text-gray-500 hover:text-gray-800'
              }`}
            >
              <BarChart3 className="w-3.5 h-3.5" />
              <span>Analytics</span>
            </button>
          </div>

          <UploadPanel onDocumentAdded={handleAddDocument} />
          
          <div className="border-t border-gray-100 dark:border-gray-900 pt-5">
            <KnowledgeBaseList
              documents={documents}
              onDeleteDocument={handleDeleteDocument}
              onClearAllDocuments={handleClearAllDocuments}
            />
          </div>
        </div>

        {/* Sidebar Footer */}
        <div className="p-4 bg-gray-50/30 dark:bg-gray-900/10 border-t border-gray-100 dark:border-gray-900 flex items-start space-x-2.5">
          <Info className="w-4 h-4 text-gray-400 dark:text-gray-500 shrink-0 mt-0.5" />
          <p className="text-[10px] text-gray-500 dark:text-gray-400 leading-relaxed font-sans">
            Ask questions about your uploaded documents. Reference citations are grounded strictly in the parsed files.
          </p>
        </div>
      </aside>

      {/* OBLIQUE BLACK BACKGROUND CLOAK FOR MOBILE SIDEDRAW */}
      {mobileSidebarOpen && (
        <div
          onClick={() => setMobileSidebarOpen(false)}
          className="fixed inset-0 bg-gray-950/40 backdrop-blur-xs z-35 md:hidden"
        />
      )}

      {/* MAIN WORKSPACE PANEL */}
      <main className="flex-1 flex flex-col min-w-0 h-[calc(100vh-56px)] md:h-screen">
        
        {/* Desktop Navbar Chrome */}
        <nav className="hidden md:flex items-center justify-between px-6 py-4 bg-white dark:bg-gray-950/20 border-b border-gray-200/60 dark:border-gray-900/80 z-10">
          {/* Left/Center Chat feed panel or analytics switch tabs */}
          <div className="flex bg-gray-100/60 dark:bg-gray-900/50 p-1 rounded-xl">
            <button
              onClick={() => setActiveTab('chat')}
              className={`px-4 py-2 text-xs font-semibold rounded-lg transition-all duration-200 flex items-center space-x-2 cursor-pointer ${
                activeTab === 'chat'
                  ? 'bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-50 shadow-sm'
                  : 'text-gray-500 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200'
              }`}
            >
              <MessageSquare className="w-3.5 h-3.5 text-teal-600 dark:text-teal-400" />
              <span>Q&A Chat Workspace</span>
            </button>
            <button
              onClick={() => setActiveTab('metrics')}
              className={`px-4 py-2 text-xs font-semibold rounded-lg transition-all duration-200 flex items-center space-x-2 cursor-pointer ${
                activeTab === 'metrics'
                  ? 'bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-50 shadow-sm'
                  : 'text-gray-500 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200'
              }`}
            >
              <BarChart3 className="w-3.5 h-3.5 text-indigo-500" />
              <span>System Latency & Analytics</span>
            </button>
          </div>

          {/* Desktop Controls */}
          <div className="flex items-center space-x-3">
            <ThemeToggle darkMode={darkMode} onToggle={() => setDarkMode(!darkMode)} />
          </div>
        </nav>

        {/* Tab View Container */}
        <div className="flex-1 min-h-0 overflow-hidden">
          {activeTab === 'chat' ? (
            /* CHAT TABS - Splits Chat view and Source Citation visualizer */
            <div className="h-full flex flex-col lg:flex-row">
              {/* Left/Center Chat feed panel */}
              <div className="flex-1 min-w-0 h-full">
                <ChatPanel
                  chatHistory={chatHistory}
                  isGenerating={isGenerating}
                  onSendMessage={handleSendMessage}
                  onCitationClick={handleCitationClick}
                  activeCitationId={activeCitationId}
                  documentsCount={documents.length}
                />
              </div>

              {/* Right Side Grounded Sources sidebar panel (NotebookLM style) */}
              <div className="w-full lg:w-96 border-t lg:border-t-0 h-80 lg:h-full shrink-0">
                <SourcesPanel
                  citations={activeMessageCitations}
                  activeCitationId={activeCitationId}
                  onCitationClick={handleCitationClick}
                />
              </div>
            </div>
          ) : (
            /* METRICS TABS - Pure Analytics dashboard */
            <div className="h-full overflow-y-auto p-6 md:p-8 bg-gray-50/30 dark:bg-gray-950/10">
              <div className="max-w-4xl mx-auto space-y-6">
                <div className="flex items-center justify-between pb-4 border-b border-gray-100 dark:border-gray-900">
                  <div>
                    <h2 className="text-base font-semibold text-gray-900 dark:text-gray-100 font-display">
                      RAG Pipeline Analytics Dashboard
                    </h2>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
                      Evaluate textual computation decomposition times, Redis semantic caching efficiency, and information retrieval precision.
                    </p>
                  </div>
                </div>

                <MetricsDashboard
                  metricsSummary={metricsSummary}
                  queryLogs={queryLogs}
                  latencyData={latencyHistory}
                />
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
