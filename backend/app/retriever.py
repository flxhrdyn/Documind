"""Retriever construction.

All retrieval wiring lives here (embeddings, Qdrant vector store, retriever
strategy). The RAG pipeline can then focus on orchestration and metrics.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient

from .cache_manager import CacheManager
from .embeddings import get_embeddings, get_sparse_embeddings
from .config import (
    GROQ_API_KEY,
    QDRANT_COLLECTION,
    RETRIEVAL_K,
    USE_HYBRID_SEARCH,
)
from .qdrant_conn import get_qdrant_client


logger = logging.getLogger(__name__)

_RetrieverStack = Tuple[Any, QdrantVectorStore, QdrantClient]
_retriever_cache: Optional[_RetrieverStack] = None
_retriever_cache_version: Optional[float] = None
_retriever_cache_lock = threading.Lock()

# Shared version stamp so multi-worker deployments invalidate their
# process-local retriever cache together: a worker rebuilds the stack
# whenever this stamp (bumped on upload/delete/clear) moves past what it
# last built from, instead of only reacting to its own in-process calls.
_RETRIEVER_VERSION_KEY = "retriever_cache_version"
_cache_manager = CacheManager()


def invalidate_retriever_cache() -> None:
    """Drop the cached retriever stack.

    Call this whenever the indexed knowledge base changes (upload/delete/clear)
    so the next query re-validates that the collection still exists. Also
    bumps a shared version stamp so other worker processes drop their own
    cached stack on their next call.
    """
    global _retriever_cache
    with _retriever_cache_lock:
        _retriever_cache = None
    _cache_manager.set(_RETRIEVER_VERSION_KEY, time.time(), ttl=30 * 24 * 60 * 60)


def build_retriever() -> _RetrieverStack:
    """Build and validate the retriever stack.

    Returns a tuple of (retriever, vectorstore, client). Raises a ValueError if
    the API key is missing or the expected Qdrant collection does not exist.

    The stack is cached process-wide: rebuilding a `ChatGroq` client and
    re-validating collection existence on every query added avoidable latency.
    The cache is invalidated by `invalidate_retriever_cache()` (document
    changes) and automatically when the underlying Qdrant client instance is
    swapped out (e.g. after a "client closed" recovery).
    """
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY belum di-set. Isi di .env (lihat .env.example) sebelum menjalankan query."
        )

    client = get_qdrant_client()
    current_version = _cache_manager.get(_RETRIEVER_VERSION_KEY)

    global _retriever_cache, _retriever_cache_version
    with _retriever_cache_lock:
        if (
            _retriever_cache is not None
            and _retriever_cache[2] is client
            and _retriever_cache_version == current_version
        ):
            return _retriever_cache

    # Embedding models (cached)
    embeddings = get_embeddings()
    sparse_embeddings = get_sparse_embeddings()

    # Guard: ensure the collection exists before serving queries.
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        raise ValueError(
            "Belum ada dokumen yang diindex. Upload dan index PDF terlebih dahulu."
        )

    # Vector store wrapper with Native Hybrid Search support
    if USE_HYBRID_SEARCH:
        logger.info("Initializing QdrantVectorStore in HYBRID mode (Dense + Sparse/BM42)")
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=QDRANT_COLLECTION,
            embedding=embeddings,
            sparse_embedding=sparse_embeddings,
            vector_name="",
            sparse_vector_name="sparse",
            retrieval_mode=RetrievalMode.HYBRID,
        )
    else:
        logger.info("Initializing QdrantVectorStore in DENSE mode")
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=QDRANT_COLLECTION,
            embedding=embeddings,
            vector_name="",
        )

    # Base retriever with MMR for diversity
    # Note: search_kwargs for hybrid might differ depending on langchain-qdrant version
    # but usually they are passed through to the search method.
    search_kwargs: Dict[str, Any] = {
        "k": RETRIEVAL_K,
    }

    base_retriever = vectorstore.as_retriever(
        search_type="mmr" if not USE_HYBRID_SEARCH else "similarity", # MMR might not be fully compatible with hybrid yet in all versions
        search_kwargs=search_kwargs
    )

    retriever = base_retriever

    logger.debug("Retriever ready (collection=%s, k=%s, mode=%s)",
                 QDRANT_COLLECTION, RETRIEVAL_K, "hybrid" if USE_HYBRID_SEARCH else "dense")

    stack = (retriever, vectorstore, client)
    with _retriever_cache_lock:
        _retriever_cache = stack
        _retriever_cache_version = current_version
    return stack


def _dedup_by_content(doc_lists: List[List[Document]]) -> List[Document]:
    """Flatten and dedupe retrieved docs by page_content, preserving first-seen
    order (matches MultiQueryRetriever's default unique-union behavior)."""
    seen: set[str] = set()
    result: List[Document] = []
    for docs in doc_lists:
        for doc in docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                result.append(doc)
    return result


def retrieve_documents(
    query: str,
    *,
    dense_retriever: Any, # Keeping name for compatibility, but it handles hybrid
    client: QdrantClient,
    fusion_queries: List[str] = (),
) -> Tuple[List[Document], Dict[str, Any]]:
    """Retrieve documents for the standalone query plus its fusion variations
    (if any), deduped by content, using the native hybrid strategy (if enabled)."""

    doc_lists = [dense_retriever.invoke(query)]
    for fusion_query in fusion_queries:
        doc_lists.append(dense_retriever.invoke(fusion_query))
    docs = _dedup_by_content(doc_lists)

    metadata: Dict[str, Any] = {
        "mode": "hybrid-native" if USE_HYBRID_SEARCH else "dense",
        "count": len(docs),
    }

    return docs, metadata


async def retrieve_documents_async(
    query: str,
    dense_retriever: Any,
    client: QdrantClient,
    fusion_queries: List[str] = (),
) -> Tuple[List[Document], Dict[str, Any]]:
    """Async counterpart of `retrieve_documents`."""

    doc_lists = [await dense_retriever.ainvoke(query)]
    for fusion_query in fusion_queries:
        doc_lists.append(await dense_retriever.ainvoke(fusion_query))
    docs = _dedup_by_content(doc_lists)

    metadata: Dict[str, Any] = {
        "mode": "hybrid-native" if USE_HYBRID_SEARCH else "dense",
        "count": len(docs),
    }

    return docs, metadata