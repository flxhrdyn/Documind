"""Local metrics store used by the Streamlit dashboard.

Metrics are stored in a small JSON file (`metrics.json`) so the dashboard can
render without an external database.
"""
import logging
import json
import math
import os
import tempfile
from datetime import datetime
import threading
from typing import Any, Dict, List, Optional

import portalocker

from .config import IR_RELEVANCE_THRESHOLD, METRICS_FILE


logger = logging.getLogger(__name__)
_metrics_lock = threading.Lock()
_METRICS_LOCKFILE = f"{METRICS_FILE}.lock"


def load_metrics() -> Dict[str, Any]:
    """Load metrics from disk.

    Returns default values when the file is missing or unreadable.
    """
    default_metrics = {
        "total_queries": 0,
        "total_documents_indexed": 0,
        "total_response_time": 0,
        "total_retrieval_time": 0,
        "total_generation_time": 0,
        "total_docs_retrieved": 0,
        "total_chunks_processed": 0,
        "query_history": []
    }
    
    if not os.path.exists(METRICS_FILE):
        return default_metrics
    
    try:
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            metrics = json.load(f)
            # Merge with default to ensure all keys exist (backward compatibility)
            for key in default_metrics:
                if key not in metrics:
                    metrics[key] = default_metrics[key]
            return metrics
    except Exception:
        logger.exception("Failed to load metrics file: %s", METRICS_FILE)
        return default_metrics


def save_metrics(metrics: Dict[str, Any]) -> None:
    """Save metrics to disk."""
    try:
        parent_dir = os.path.dirname(METRICS_FILE)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        # Atomic write to avoid corrupting the JSON file on crashes or
        # concurrent writes.
        fd, tmp_path = tempfile.mkstemp(prefix="metrics_", suffix=".json", dir=parent_dir or None)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            os.replace(tmp_path, METRICS_FILE)
        finally:
            # If os.replace fails, best-effort cleanup.
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
    except Exception:
        logger.exception("Failed to save metrics file: %s", METRICS_FILE)


def log_query(
    question: str, 
    response_time: float, 
    answer_length: int = 0,
    retrieval_time: float = 0,
    generation_time: float = 0,
    docs_retrieved: int = 0,
    chunks_processed: int = 0,
    retrieval_scores: Optional[List[float]] = None,
    initial_ranks: Optional[List[int]] = None,
    thoughts: str = "",
    answer: str = "",
    standalone_query: str = "",
) -> None:
    """Log a query with RAG metrics."""
    # threading.Lock guards same-process concurrency cheaply; portalocker's
    # file lock additionally guards the read-modify-write across multiple
    # worker processes, which the threading.Lock alone cannot do.
    with _metrics_lock:
        with portalocker.Lock(_METRICS_LOCKFILE, timeout=10):
            _log_query_locked(
                question, response_time, answer_length, retrieval_time,
                generation_time, docs_retrieved, chunks_processed,
                retrieval_scores, initial_ranks, thoughts, answer, standalone_query,
            )


def _log_query_locked(
    question: str,
    response_time: float,
    answer_length: int,
    retrieval_time: float,
    generation_time: float,
    docs_retrieved: int,
    chunks_processed: int,
    retrieval_scores: Optional[List[float]],
    initial_ranks: Optional[List[int]],
    thoughts: str,
    answer: str,
    standalone_query: str,
) -> None:
        metrics = load_metrics()
        
        metrics["total_queries"] += 1
        metrics["total_response_time"] += response_time
        metrics["total_retrieval_time"] += retrieval_time
        metrics["total_generation_time"] += generation_time
        metrics["total_docs_retrieved"] += docs_retrieved
        metrics["total_chunks_processed"] += chunks_processed
        
        # Keep only last 100 queries to prevent file from growing too large
        if len(metrics["query_history"]) >= 100:
            metrics["query_history"].pop(0)
        
        # Calculate answer length if not provided
        final_answer_length = answer_length or len(answer)
        
        metrics["query_history"].append({
            "timestamp": datetime.now().isoformat(),
            "question": question[:100],  # Truncate long questions
            "response_time": round(response_time, 2),
            "retrieval_time": round(retrieval_time, 2),
            "generation_time": round(generation_time, 2),
            "answer_length": final_answer_length,
            "docs_retrieved": docs_retrieved,
            "chunks_processed": chunks_processed,
            "retrieval_scores": retrieval_scores or [],
            "initial_ranks": initial_ranks or [],
            "thoughts": thoughts,
            "standalone_query": standalone_query,
        })
        
        save_metrics(metrics)


def log_document_indexed() -> None:
    """Record that a document was indexed."""
    with _metrics_lock:
        with portalocker.Lock(_METRICS_LOCKFILE, timeout=10):
            metrics = load_metrics()
            metrics["total_documents_indexed"] += 1
            save_metrics(metrics)


def sync_indexed_docs_count(count: int) -> None:
    """Manually sync the total documents indexed count (e.g. from vector store)."""
    with _metrics_lock:
        with portalocker.Lock(_METRICS_LOCKFILE, timeout=10):
            metrics = load_metrics()
            metrics["total_documents_indexed"] = count
            save_metrics(metrics)


def _avg_per_query(field: str, ndigits: int = 2) -> float:
    """Average a cumulative metrics field over total_queries."""
    metrics = load_metrics()
    if metrics["total_queries"] == 0:
        return 0.0
    return round(metrics[field] / metrics["total_queries"], ndigits)


def _pct_of_total_response_time(field: str) -> float:
    """A cumulative metrics field as a percentage of total_response_time."""
    metrics = load_metrics()
    if metrics["total_response_time"] == 0:
        return 0.0
    return round((metrics[field] / metrics["total_response_time"]) * 100, 1)


def get_avg_response_time() -> float:
    """Return average total response time."""
    return _avg_per_query("total_response_time")


def get_avg_retrieval_time() -> float:
    """Return average retrieval time."""
    return _avg_per_query("total_retrieval_time")


def get_avg_generation_time() -> float:
    """Return average generation time."""
    return _avg_per_query("total_generation_time")


def get_avg_docs_retrieved() -> float:
    """Return average number of retrieved documents per query."""
    return _avg_per_query("total_docs_retrieved", ndigits=1)


def get_avg_chunks_processed() -> float:
    """Return average number of chunks processed per query."""
    return _avg_per_query("total_chunks_processed", ndigits=1)


def get_retrieval_efficiency() -> float:
    """Return retrieval time as a percentage of total time."""
    return _pct_of_total_response_time("total_retrieval_time")


def get_generation_efficiency() -> float:
    """Return generation time as a percentage of total time."""
    return _pct_of_total_response_time("total_generation_time")


def reset_metrics() -> None:
    """Reset all stored metrics."""
    with _metrics_lock:
        with portalocker.Lock(_METRICS_LOCKFILE, timeout=10):
            metrics = {
                "total_queries": 0,
                "total_documents_indexed": 0,
                "total_response_time": 0,
                "total_retrieval_time": 0,
                "total_generation_time": 0,
                "total_docs_retrieved": 0,
                "total_chunks_processed": 0,
                "query_history": []
            }
            save_metrics(metrics)


# ── IR Metric Computation ─────────────────────────────────────────────────────

def _binary_relevance(scores: List[float], threshold: float) -> List[int]:
    """Return binary relevance list: 1 if score >= threshold, else 0."""
    return [1 if s >= threshold else 0 for s in scores]


def precision_at_k(scores: List[float], k: int, threshold: float = IR_RELEVANCE_THRESHOLD) -> float:
    """Precision@k: fraction of top-k retrieved docs that are relevant."""
    if not scores or k == 0:
        return 0.0
    rel = _binary_relevance(scores[:k], threshold)
    return sum(rel) / k


def recall_at_k(scores: List[float], k: int, threshold: float = IR_RELEVANCE_THRESHOLD) -> float:
    """Recall@k: fraction of relevant docs found in top-k.

    Since total relevant in the collection is unknown, we approximate using the
    total relevant among all retrieved scores as a lower bound.
    """
    if not scores:
        return 0.0
    total_relevant = sum(_binary_relevance(scores, threshold))
    if total_relevant == 0:
        return 0.0
    relevant_in_k = sum(_binary_relevance(scores[:k], threshold))
    return relevant_in_k / total_relevant


def mrr(scores: List[float], threshold: float = IR_RELEVANCE_THRESHOLD) -> float:
    """Mean Reciprocal Rank: 1/rank of the first relevant document."""
    for i, s in enumerate(scores):
        if s >= threshold:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(scores: List[float], k: int, initial_ranks: Optional[List[int]] = None) -> float:
    """nDCG@k: post-rerank scores as graded relevance.

    `scores` is already sorted descending by rerank score (FlashRank's output
    order), so comparing it to itself sorted descending is a no-op and nDCG
    is trivially always 1.0. `initial_ranks[i]` is the position doc `i` (in
    `scores` order) held in the pre-rerank retrieval order - a serving order
    genuinely independent of the relevance grade. When available, DCG is
    computed over that pre-rerank order instead, benchmarked against the
    ideal (score-sorted) ordering for IDCG. Without it, falls back to the
    degenerate 1.0 (no independent ordering to score).
    """
    if not scores:
        return 0.0
    ideal = sorted(scores, reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    if idcg == 0:
        return 0.0

    if initial_ranks and len(initial_ranks) == len(scores):
        pre_rerank_order = sorted(range(len(scores)), key=lambda i: initial_ranks[i])
        actual = [scores[i] for i in pre_rerank_order][:k]
    else:
        actual = scores[:k]

    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(actual))
    return dcg / idcg


def hit_rate_at_k(scores: List[float], k: int, threshold: float = IR_RELEVANCE_THRESHOLD) -> float:
    """HitRate@k: 1 if at least one relevant doc in top-k, else 0."""
    if not scores:
        return 0.0
    return 1.0 if any(s >= threshold for s in scores[:k]) else 0.0


def compute_ir_metrics(
    query_history: List[Dict[str, Any]],
    k: int = 5,
    threshold: float = IR_RELEVANCE_THRESHOLD,
) -> Dict[str, Any]:
    """Compute aggregate IR metrics from stored query history."""
    entries_with_scores = [
        q for q in query_history if q.get("retrieval_scores")
    ]
    if not entries_with_scores:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "mrr": 0.0,
            "ndcg": 0.0,
            "hit_rate": 0.0,
            "evaluated_queries": 0,
        }

    p_vals, r_vals, mrr_vals, ndcg_vals, hr_vals = [], [], [], [], []
    for q in entries_with_scores:
        s = q["retrieval_scores"]
        p_vals.append(precision_at_k(s, k, threshold))
        r_vals.append(recall_at_k(s, k, threshold))
        mrr_vals.append(mrr(s, threshold))
        ndcg_vals.append(ndcg_at_k(s, k, initial_ranks=q.get("initial_ranks")))
        hr_vals.append(hit_rate_at_k(s, k, threshold))

    n = len(entries_with_scores)
    return {
        "precision": round(sum(p_vals) / n, 4),
        "recall": round(sum(r_vals) / n, 4),
        "mrr": round(sum(mrr_vals) / n, 4),
        "ndcg": round(sum(ndcg_vals) / n, 4),
        "hit_rate": round(sum(hr_vals) / n, 4),
        "evaluated_queries": n,
    }


def per_query_ir_metrics(
    query_history: List[Dict[str, Any]],
    k: int = 5,
    threshold: float = IR_RELEVANCE_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Return per-query IR metrics for charting."""
    rows = []
    for q in query_history:
        s = q.get("retrieval_scores", [])
        rows.append({
            "timestamp": q.get("timestamp", ""),
            "question": q.get("question", "")[:50],
            "response_time": q.get("response_time", 0),
            "retrieval_time": q.get("retrieval_time", 0),
            "generation_time": q.get("generation_time", 0),
            "docs_retrieved": q.get("docs_retrieved", 0),
            "precision": round(precision_at_k(s, k, threshold), 4) if s else None,
            "recall": round(recall_at_k(s, k, threshold), 4) if s else None,
            "mrr": round(mrr(s, threshold), 4) if s else None,
            "ndcg": round(ndcg_at_k(s, k, initial_ranks=q.get("initial_ranks")), 4) if s else None,
            "hit_rate": round(hit_rate_at_k(s, k, threshold), 4) if s else None,
        })
    return rows
