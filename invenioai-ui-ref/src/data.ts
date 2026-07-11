/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { DocumentFile, ChatMessage, QueryMetrics, QueryLog, Citation } from './types';

export const PRELOADED_DOCUMENTS: DocumentFile[] = [
  {
    id: 'doc-1',
    name: 'Attention_Is_All_You_Need.pdf',
    size: '1.2 MB',
    uploadedAt: '2026-07-09 14:22',
    chunksCount: 3,
    status: 'succeeded',
    chunks: [
      {
        id: 'chunk-1-1',
        text: 'The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.',
        pageNumber: 1
      },
      {
        id: 'chunk-1-2',
        text: 'Self-attention, sometimes called intra-attention, is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. It has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations.',
        pageNumber: 2
      },
      {
        id: 'chunk-1-3',
        text: 'Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this. Multi-head attention consists of eight parallel attention layers, or heads, allowing the model to perform representation learning efficiently.',
        pageNumber: 5
      }
    ]
  },
  {
    id: 'doc-2',
    name: 'InvenioAI_Architecture_Guide.pdf',
    size: '840 KB',
    uploadedAt: '2026-07-09 15:45',
    chunksCount: 3,
    status: 'succeeded',
    chunks: [
      {
        id: 'chunk-2-1',
        text: 'InvenioAI implements a hybrid dense-sparse retriever. Dense embeddings are generated using BGE-M3 (yielding a dense vector of 1024 dimensions), while sparse lexical representations are obtained via BM25. This hybrid candidates set is then merged via Reciprocal Rank Fusion (RRF) and reranked using a Cross-Encoder (Cohere Rerank v3) to maximize precision.',
        pageNumber: 1
      },
      {
        id: 'chunk-2-2',
        text: 'Dual-layer semantic caching is deployed at the edge node. Exact matching queries are served in < 15ms via a key-value Redis store. Near-match semantic queries are evaluated using Cosine similarity of candidate query embeddings, yielding a cache hit and serving the cached response directly when similarity exceeds a threshold of 0.95.',
        pageNumber: 3
      },
      {
        id: 'chunk-2-3',
        text: 'The system incorporates RAG Fusion which performs multi-query expansion on incoming queries. For each incoming user query, four semantically similar query variations are generated via a lightweight LLM (Gemini 2.5 Flash). These variations are queried against the vector database in parallel, yielding a robust and thorough document candidates set.',
        pageNumber: 4
      }
    ]
  },
  {
    id: 'doc-3',
    name: 'AcmeCorp_FY2026_Sustainability_Report.pdf',
    size: '2.1 MB',
    uploadedAt: '2026-07-10 09:15',
    chunksCount: 2,
    status: 'succeeded',
    chunks: [
      {
        id: 'chunk-3-1',
        text: 'Acme Corp has achieved a 42% reduction in greenhouse gas emissions across all Level 1 and Level 2 operations as of Q2 2026, ahead of our 2030 net-zero milestone roadmap. This was achieved primarily through the electrification of our logistics fleet and the deployment of solar arrays on our manufacturing rooftops.',
        pageNumber: 12
      },
      {
        id: 'chunk-3-2',
        text: 'Water reclamation facilities at our primary manufacturing plants in Oregon and Selangor recycled over 1.2 million cubic meters of industrial water in FY2026, representing a water reuse rate of 82%, and significantly reducing pressure on regional aquifers.',
        pageNumber: 24
      }
    ]
  }
];

export const INITIAL_CHAT_HISTORY: ChatMessage[] = [
  {
    id: 'msg-1',
    role: 'user',
    content: 'How does InvenioAI combine dense and sparse document retrieval?',
    timestamp: '10:02'
  },
  {
    id: 'msg-2',
    role: 'assistant',
    content: 'InvenioAI merges document search by implementing a **hybrid dense-sparse retriever architecture** [1]. \n\nThe system works through the following pipeline:\n1. **Dense Retrieval**: Utilizes the *BGE-M3* model to capture semantic similarities within a 1024-dimensional vector space.\n2. **Sparse Retrieval**: Employs the *BM25* algorithm for keyword-based lexical matching to capture exact product names, serial codes, or domain-specific terminology.\n3. **Fusion & Reranking**: Candidate documents retrieved from both paths are unified using the *Reciprocal Rank Fusion (RRF)* method and then sorted again via a *Cross-Encoder (Cohere Rerank v3)* to maximize precision before being fed into the LLM generator.',
    timestamp: '10:02',
    citations: [
      {
        id: '1',
        documentId: 'doc-2',
        documentName: 'InvenioAI_Architecture_Guide.pdf',
        pageNumber: 1,
        relevanceScore: 0.942,
        snippet: 'InvenioAI implements a hybrid dense-sparse retriever. Dense embeddings are generated using BGE-M3 (yielding a dense vector of 1024 dimensions), while sparse lexical representations are obtained via BM25. This hybrid candidates set is then merged via Reciprocal Rank Fusion (RRF) and reranked using a Cross-Encoder (Cohere Rerank v3) to maximize precision.'
      }
    ],
    metrics: {
      totalTimeMs: 420,
      retrievalTimeMs: 145,
      generationTimeMs: 275,
      chunksRetrieved: 3,
      cacheHit: false
    }
  }
];

export const INITIAL_QUERY_LOGS: QueryLog[] = [
  {
    id: 'log-1',
    query: 'How does InvenioAI combine dense and sparse document retrieval?',
    timestamp: '2026-07-10 10:02:11',
    latencyMs: 420,
    sourceCount: 1,
    topScore: 0.94
  },
  {
    id: 'log-2',
    query: 'What is the self-attention mechanism?',
    timestamp: '2026-07-10 09:41:03',
    latencyMs: 512,
    sourceCount: 2,
    topScore: 0.89
  },
  {
    id: 'log-3',
    query: 'What are Acme Corp carbon emissions in FY2026?',
    timestamp: '2026-07-10 09:18:45',
    latencyMs: 388,
    sourceCount: 1,
    topScore: 0.91
  },
  {
    id: 'log-4',
    query: 'How does semantic caching operate at the edge node?',
    timestamp: '2026-07-10 08:33:12',
    latencyMs: 12, // Cache Hit!
    sourceCount: 1,
    topScore: 0.98
  }
];

// Helper to simulate a streaming answer based on keyword mapping
export function getSmartMockResponse(query: string, currentDocs: DocumentFile[]): {
  content: string;
  citations: Citation[];
  metrics: QueryMetrics;
} {
  const lowercaseQuery = query.toLowerCase();
  const citations: Citation[] = [];
  let content = '';
  
  // Latency simulation variables
  let retrievalTimeMs = 120 + Math.floor(Math.random() * 60);
  let generationTimeMs = 280 + Math.floor(Math.random() * 120);
  let cacheHit = false;

  // Let's check for semantic cache first to simulate edge caches
  if (lowercaseQuery.includes('cache') || lowercaseQuery.includes('edge') || lowercaseQuery.includes('redis')) {
    const doc = currentDocs.find(d => d.id === 'doc-2');
    const chunk = doc?.chunks.find(c => c.id === 'chunk-2-2');
    if (chunk) {
      cacheHit = Math.random() > 0.4; // 60% chance of cache hit if repeating
      if (cacheHit) {
        retrievalTimeMs = 11;
        generationTimeMs = 3;
      }
      
      citations.push({
        id: '1',
        documentId: doc!.id,
        documentName: doc!.name,
        pageNumber: chunk.pageNumber,
        relevanceScore: 0.982,
        snippet: chunk.text
      });

      content = `InvenioAI deploys a **dual-layer semantic caching system** directly at the edge node [1]. \n\nThis architecture is optimized for low-latency operations:\n- **Exact Match Check**: Any incoming query matching a previous request exactly is immediately served via an in-memory Redis key-value store in less than 15 milliseconds.\n- **Semantic Match Evaluation**: If the query is slightly altered but retains identical semantic meaning, the system calculates the *Cosine similarity* of the candidate query vector embedding. If the alignment exceeds a threshold of **0.95**, the cached response is served instantly. This avoids redundant LLM invocation fees and yields ultra-low response latencies.`;
    }
  }
  
  // Transformer / Attention query
  else if (lowercaseQuery.includes('attention') || lowercaseQuery.includes('transformer') || lowercaseQuery.includes('multi-head') || lowercaseQuery.includes('self-attention')) {
    const doc = currentDocs.find(d => d.id === 'doc-1');
    if (doc) {
      const chunk1 = doc.chunks.find(c => c.id === 'chunk-1-1');
      const chunk2 = doc.chunks.find(c => c.id === 'chunk-1-2');
      const chunk3 = doc.chunks.find(c => c.id === 'chunk-1-3');

      if (lowercaseQuery.includes('multi-head') && chunk3) {
        citations.push({
          id: '1',
          documentId: doc.id,
          documentName: doc.name,
          pageNumber: chunk3.pageNumber,
          relevanceScore: 0.925,
          snippet: chunk3.text
        });
        content = `According to the source documentation, **Multi-head attention** allows the model to jointly attend to information from different representation subspaces at different positions [1].\n\n- With a single attention head, averaging inhibits this rich subspace focus.\n- Multi-head attention instead consists of **eight parallel attention layers (heads)**, allowing the model to perform highly efficient, simultaneous representation learning.`;
      } else {
        if (chunk1) {
          citations.push({
            id: '1',
            documentId: doc.id,
            documentName: doc.name,
            pageNumber: chunk1.pageNumber,
            relevanceScore: 0.911,
            snippet: chunk1.text
          });
        }
        if (chunk2) {
          citations.push({
            id: '2',
            documentId: doc.id,
            documentName: doc.name,
            pageNumber: chunk2.pageNumber,
            relevanceScore: 0.884,
            snippet: chunk2.text
          });
        }
        
        content = `According to the landmark research paper, the **Transformer** architecture is designed solely around attention mechanisms, completely dispensing with recurrence and convolutional structures [1].\n\nA core module within this architecture is **Self-Attention** (sometimes termed intra-attention) [2]. Self-attention connects different relative positions of a single sequence to construct an expressive representation of the entire sequence. It has been successfully deployed across reading comprehension, abstractive summarization, textual entailment, and learning task-independent representations.`;
      }
    }
  }
  
  // Sustainability / Acme Corp query
  else if (lowercaseQuery.includes('acme') || lowercaseQuery.includes('sustainability') || lowercaseQuery.includes('emission') || lowercaseQuery.includes('emisi') || lowercaseQuery.includes('carbon') || lowercaseQuery.includes('water') || lowercaseQuery.includes('aquifer') || lowercaseQuery.includes('reclamation')) {
    const doc = currentDocs.find(d => d.id === 'doc-3');
    if (doc) {
      const chunk1 = doc.chunks.find(c => c.id === 'chunk-3-1');
      const chunk2 = doc.chunks.find(c => c.id === 'chunk-3-2');

      if ((lowercaseQuery.includes('water') || lowercaseQuery.includes('aquifer') || lowercaseQuery.includes('reclamation')) && chunk2) {
        citations.push({
          id: '1',
          documentId: doc.id,
          documentName: doc.name,
          pageNumber: chunk2.pageNumber,
          relevanceScore: 0.941,
          snippet: chunk2.text
        });
        content = `Based on the Acme Corp Sustainability Report, the company operates dedicated water reclamation facilities at their primary manufacturing sites in **Oregon and Selangor** [1].\n\nThese plants successfully recycled **over 1.2 million cubic meters of industrial water** in FY2026. This equates to an overall water reuse rate of **82%**, substantially reducing withdrawal strain on critical regional aquifers.`;
      } else {
        if (chunk1) {
          citations.push({
            id: '1',
            documentId: doc.id,
            documentName: doc.name,
            pageNumber: chunk1.pageNumber,
            relevanceScore: 0.954,
            snippet: chunk1.text
          });
        }
        content = `Acme Corp reported a **42% reduction in greenhouse gas emissions** across Level 1 and Level 2 operations as of Q2 2026 [1].\n\nThis achievement positions them ahead of their scheduled 2030 net-zero roadmap milestones. The significant carbon drop was primarily driven by:\n1. **Electrification of the logistics fleet** globally.\n2. **Deployment of solar arrays** across their factory rooftops.`;
      }
    }
  }
  
  // Fallback for custom uploaded documents or general query
  else {
    // If we have custom documents, we try to fetch chunks from them
    const customDocs = currentDocs.filter(d => !['doc-1', 'doc-2', 'doc-3'].includes(d.id));
    if (customDocs.length > 0 && customDocs[0].chunks.length > 0) {
      const activeDoc = customDocs[0];
      const randomChunk = activeDoc.chunks[0];
      
      citations.push({
        id: '1',
        documentId: activeDoc.id,
        documentName: activeDoc.name,
        pageNumber: randomChunk.pageNumber,
        relevanceScore: 0.876 + (Math.random() * 0.08),
        snippet: randomChunk.text
      });

      content = `Based on a semantic search within your uploaded document (**${activeDoc.name}**), the retriever extracted the following relevant excerpt [1]:\n\n> "${randomChunk.text}"\n\nThis specific text passage was identified as the best match in the document. Would you like me to analyze other aspects or elaborate on this section of the document?`;
    } else {
      // General response fallback citing preloaded docs or explaining system
      const doc = currentDocs[0] || PRELOADED_DOCUMENTS[1];
      const chunk = doc.chunks[0];
      
      citations.push({
        id: '1',
        documentId: doc.id,
        documentName: doc.name,
        pageNumber: chunk.pageNumber,
        relevanceScore: 0.724,
        snippet: chunk.text
      });

      content = `I couldn't find an exact answer to your question in the uploaded documents.\n\nHowever, the most relevant passage found was in **${doc.name}** (Page ${chunk.pageNumber}) [1].\n\nTo ensure accuracy, I restrict my answers to the factual content in your uploaded PDFs. You can query me on:\n- **Search pipelines and semantic caching** (Architecture Guide)\n- **Transformers, self-attention, and multi-head attention** (Attention Is All You Need Paper)\n- **Carbon emissions reductions or water recycling rates** (Sustainability Report)\n- Or simply drag-and-drop your own PDF in the left sidebar!`;
    }
  }

  return {
    content,
    citations,
    metrics: {
      totalTimeMs: retrievalTimeMs + generationTimeMs,
      retrievalTimeMs,
      generationTimeMs,
      chunksRetrieved: citations.length + Math.floor(Math.random() * 2),
      cacheHit
    }
  };
}
