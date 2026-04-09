"""RAG pipeline orchestration.

The end-to-end flow is:

rewrite query → retrieve → rerank → generate answer → return metrics.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from .config import GEMINI_API_KEY, LLM_MODEL, RETRIEVAL_K
from .reranker import rerank
from .retriever import build_retriever
from .utils import format_docs


logger = logging.getLogger(__name__)


QUERY_REWRITE_PROMPT = """
Rewrite the question into a standalone question.

Chat History:
{history}

Question:
{question}

Standalone Question:
"""


RAG_PROMPT = """
Answer the question using ONLY the context.

Context:
{context}

Question:
{question}

Answer in the same language.

Sources:
{sources}
"""


@lru_cache(maxsize=1)
def _get_llm() -> ChatGoogleGenerativeAI:
    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY belum di-set. Isi di .env (lihat .env.example) sebelum menjalankan query."
        )

    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0,
    )


def format_history(history: Any) -> str:
    """Format chat history into a prompt-friendly string."""
    if not history:
        return ""

    if isinstance(history, str):
        return history

    return "\n".join(str(item) for item in history)


def rewrite_query(question: str, history: Any) -> str:
    """Rewrite a question into a standalone query."""
    prompt = QUERY_REWRITE_PROMPT.format(
        history=format_history(history),
        question=question,
    )
    return _get_llm().invoke(prompt).content


def rag_pipeline(question: str, history: Any) -> dict[str, Any]:
    """Run the RAG pipeline.

    Returns a dict containing the answer, a sources string, and timing metrics.
    """

    total_start = time.monotonic()
    retriever, vectorstore, _ = build_retriever()

    try:
        standalone_query = rewrite_query(question, history)

        retrieval_start = time.monotonic()
        retrieved_docs = retriever.invoke(standalone_query)

        reranked_docs = rerank(standalone_query, retrieved_docs)
        retrieval_time = time.monotonic() - retrieval_start

        # Per-document similarity scores for the dashboard.
        try:
            scored = vectorstore.similarity_search_with_relevance_scores(
                standalone_query, k=RETRIEVAL_K
            )
            retrieval_scores = [round(float(score), 4) for _, score in scored]
        except Exception:
            logger.debug("Failed to fetch relevance scores", exc_info=True)
            retrieval_scores = []

        context, sources = format_docs(reranked_docs)

        prompt = RAG_PROMPT.format(context=context, question=question, sources=sources)

        generation_start = time.monotonic()
        answer_msg = _get_llm().invoke(prompt)
        generation_time = time.monotonic() - generation_start

        total_time = time.monotonic() - total_start

        return {
            "answer": answer_msg.content,
            "sources": sources,
            "metrics": {
                "total_time": round(total_time, 2),
                "retrieval_time": round(retrieval_time, 2),
                "generation_time": round(generation_time, 2),
                "docs_retrieved": len(retrieved_docs),
                "chunks_processed": len(reranked_docs),
                "retrieval_scores": retrieval_scores,
            },
        }
    finally:
        # Qdrant client is managed as a process-wide singleton and closed on
        # application shutdown.
        pass