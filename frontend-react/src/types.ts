export interface DocumentsResponse {
  documents: string[];
  count: number;
}

export interface Citation {
  file: string;
  page: number | null;
  header: string | null;
  score: number | null;
  text: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Citation[];
  thoughts?: string;
  timestamp?: string;
  isError?: boolean;
}

export interface UploadJob {
  status: string;
  result?: { filename?: string };
  error?: string;
}

export interface IRQuality {
  hit_rate: number;
  ndcg: number;
  mrr: number;
  precision: number;
  recall: number;
}

export interface MetricsResponse {
  total_queries: number;
  total_documents_indexed: number;
  avg_response_time: number;
  avg_retrieval_time: number;
  avg_generation_time: number;
  avg_docs_retrieved: number;
  ir_quality: IRQuality;
  query_history: Array<{
    question: string;
    response_time?: number;
    retrieval_time?: number;
    generation_time?: number;
    timestamp?: string;
  }>;
}
