/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export interface DocumentChunk {
  id: string;
  text: string;
  pageNumber: number;
}

export interface DocumentFile {
  id: string;
  name: string;
  size: string;
  uploadedAt: string;
  chunksCount: number;
  status: 'pending' | 'parsing' | 'indexing' | 'succeeded' | 'failed';
  error?: string;
  chunks: DocumentChunk[];
}

export interface Citation {
  id: string; // e.g. "1", "2"
  documentId: string;
  documentName: string;
  pageNumber: number;
  relevanceScore: number; // e.g. 0.89 (reranker score)
  snippet: string;
}

export interface QueryMetrics {
  totalTimeMs: number;
  retrievalTimeMs: number;
  generationTimeMs: number;
  chunksRetrieved: number;
  cacheHit: boolean;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  citations?: Citation[];
  metrics?: QueryMetrics;
}

export interface RAGMetricsSummary {
  totalQueries: number;
  avgLatencyMs: number;
  avgRetrievalMs: number;
  avgGenerationMs: number;
  mrrScore: number; // Mean Reciprocal Rank
  ndcgScore: number; // Normalized Discounted Cumulative Gain
  hitRate: number; // Retrieval Hit Rate (0-1)
  precisionAtK: number; // Precision@K
  efficiencyRatio: number; // retrieval/generation efficiency (0-100)
}

export interface QueryLog {
  id: string;
  query: string;
  timestamp: string;
  latencyMs: number;
  sourceCount: number;
  topScore: number;
}
