"""Retriever construction.

All retrieval wiring lives here (embeddings, Qdrant vector store, retriever
strategy). The RAG pipeline can then focus on orchestration and metrics.
"""

from __future__ import annotations

import logging
from typing import Tuple

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from .embeddings import get_embeddings
from .config import GEMINI_API_KEY, LLM_MODEL, QDRANT_COLLECTION, RETRIEVAL_K
from .qdrant_conn import get_qdrant_client


logger = logging.getLogger(__name__)

# Default to quieter logs; let the application configure logging if needed.
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.WARNING)


def build_retriever() -> Tuple[MultiQueryRetriever, QdrantVectorStore, QdrantClient]:
    """Build and validate the retriever stack.

    Returns a tuple of (retriever, vectorstore, client). Raises a ValueError if
    the API key is missing or the expected Qdrant collection does not exist.
    """
    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY belum di-set. Isi di .env (lihat .env.example) sebelum menjalankan query."
        )

    # Embedding model (cached)
    embeddings = get_embeddings()

    # Shared Qdrant client (avoids local storage lock issues)
    client = get_qdrant_client()

    # Guard: ensure the collection exists before serving queries.
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        raise ValueError(
            "Belum ada dokumen yang diindex. Upload dan index PDF terlebih dahulu."
        )

    # Vector store wrapper
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embedding=embeddings,
    )

    # Base retriever with MMR for diversity
    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": RETRIEVAL_K,
            "fetch_k": max(RETRIEVAL_K * 4, RETRIEVAL_K),
            "lambda_mult": 0.5
        }
    )

    # Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0
    )

    # MultiQuery Retriever
    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )

    logger.debug("Retriever ready (collection=%s, k=%s)", QDRANT_COLLECTION, RETRIEVAL_K)
    return retriever, vectorstore, client