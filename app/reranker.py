"""Cross-encoder reranking.

Reranking is best-effort: if the reranker cannot be loaded (missing wheels,
network restrictions, HF rate limits), the app falls back to the base retriever
order instead of failing the whole request.
"""

from __future__ import annotations

import logging
from typing import Any, List

# Import config FIRST so Hugging Face Hub env defaults (timeouts/offline) are set
# before importing the HF stack.
from . import config as _config  # noqa: F401
from .config import RERANKER_MODEL, RERANK_TOP_K

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_reranker_model: CrossEncoder | None = None


def _get_reranker_model() -> CrossEncoder:
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder(RERANKER_MODEL)
    return _reranker_model


def rerank(query: str, docs: List[Any]) -> List[Any]:
    """Return documents ordered by cross-encoder relevance.

    Args:
        query: User query (or rewritten query).
        docs: Retrieved documents (LangChain `Document`-like objects).

    Returns:
        Up to `RERANK_TOP_K` documents. If reranking fails, returns the first
        `RERANK_TOP_K` documents in their original order.
    """

    pairs = [(query, doc.page_content) for doc in docs]

    try:
        scores = _get_reranker_model().predict(pairs)
    except Exception as exc:
        # Reranking is an optimization; treat failures as non-fatal.
        logger.warning("Reranker unavailable; skipping rerank (%s)", exc)
        return docs[:RERANK_TOP_K]

    ranked = sorted(zip(scores, docs), reverse=True, key=lambda x: x[0])
    return [doc for _, doc in ranked[:RERANK_TOP_K]]