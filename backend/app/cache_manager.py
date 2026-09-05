import json
import logging
import threading
from typing import Any, Optional

import diskcache
import redis

from .config import BASE_DIR, CACHE_TYPE, REDIS_URL

logger = logging.getLogger(__name__)

_SEMANTIC_REGISTRY_KEY = "semantic_registry"


class CacheManager:
    def __init__(self, cache_type: str = CACHE_TYPE, redis_url: str = REDIS_URL):
        self.cache_type = cache_type
        self.redis_client = None
        self.disk_cache = None
        # Guards the semantic registry read-modify-write cycle so concurrent
        # requests don't clobber each other's additions.
        self._semantic_lock = threading.Lock()

        if self.cache_type == "redis":
            try:
                self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
                # Force connection check immediately
                self.redis_client.ping()
                logger.info(f"Initialized Redis cache at {redis_url}")
            except Exception as e:
                logger.error(f"Failed to initialize Redis cache: {e}. Falling back to diskcache.")
                self.cache_type = "diskcache"
                self.redis_client = None
                
        if self.cache_type == "diskcache":
            cache_dir = BASE_DIR / ".cache" / "invenio"
            self.disk_cache = diskcache.Cache(cache_dir)
            logger.info(f"Initialized diskcache at {cache_dir}")

    def get(self, key: str) -> Optional[Any]:
        try:
            if self.cache_type == "redis" and self.redis_client:
                val = self.redis_client.get(key)
                if val:
                    return json.loads(val)
                return None
            elif self.cache_type == "diskcache" and self.disk_cache is not None:
                return self.disk_cache.get(key)
        except Exception as e:
            logger.warning(f"Cache get failed for key {key}: {e}")
        return None

    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        try:
            if self.cache_type == "redis" and self.redis_client:
                # We store objects as JSON strings in Redis
                self.redis_client.setex(key, ttl, json.dumps(value))
            elif self.cache_type == "diskcache" and self.disk_cache is not None:
                self.disk_cache.set(key, value, expire=ttl)
        except Exception as e:
            logger.warning(f"Cache set failed for key {key}: {e}")

    def _load_semantic_registry(self) -> list[dict[str, Any]]:
        if self.cache_type == "redis" and self.redis_client:
            raw = self.redis_client.get(_SEMANTIC_REGISTRY_KEY)
            return json.loads(raw) if raw else []
        elif self.cache_type == "diskcache" and self.disk_cache is not None:
            return self.disk_cache.get(_SEMANTIC_REGISTRY_KEY, [])
        return []

    def _save_semantic_registry(self, registry: list[dict[str, Any]]) -> None:
        if self.cache_type == "redis" and self.redis_client:
            self.redis_client.set(_SEMANTIC_REGISTRY_KEY, json.dumps(registry))
        elif self.cache_type == "diskcache" and self.disk_cache is not None:
            self.disk_cache.set(_SEMANTIC_REGISTRY_KEY, registry)

    def get_semantic(
        self,
        query_embedding: list[float],
        threshold: float = 0.90,
        query_text: Optional[str] = None,
    ) -> Optional[str]:
        """Find a similar query in the semantic registry and return its cache key.

        Works for both diskcache and Redis backends: the registry is a JSON list
        of {vector, key, query} entries stored under one key and scanned linearly.

        Includes a number/year guard: if the incoming query and the cached query
        differ in ANY numeric token (e.g. '2022' vs '2023', '15,433' vs '16,136'),
        the cache hit is rejected even when cosine similarity exceeds the threshold.
        This prevents year-shifted queries from returning stale cached answers.
        """
        try:
            with self._semantic_lock:
                registry = self._load_semantic_registry()
        except Exception as e:
            logger.warning(f"Failed to load semantic registry: {e}")
            return None
        if not registry:
            return None

        import re
        import numpy as np

        query_vec = np.array(query_embedding).flatten()
        norm_a = np.linalg.norm(query_vec)
        if norm_a == 0:
            return None

        # Vectorized cosine similarity: one matrix-vector product against all
        # registry entries instead of a per-entry Python loop, so the O(n)
        # scan stays cheap as the registry grows toward its 1000-entry cap.
        matrix = np.array([entry["vector"] for entry in registry])
        norms = np.linalg.norm(matrix, axis=1)
        valid = norms > 0
        scores = np.full(len(registry), -1.0)
        scores[valid] = (matrix[valid] @ query_vec) / (norms[valid] * norm_a)

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        best_key = registry[best_idx]["key"]
        best_entry_text = registry[best_idx].get("query")

        logger.debug(f"Semantic Cache Search: best_score={best_score:.4f}, threshold={threshold}")

        if best_score >= threshold:
            # Number/year guard: reject hits where numeric tokens differ.
            # Queries like 'total X in 2022' and 'total X in 2023' are nearly
            # identical in embedding space but have completely different answers.
            if query_text and best_entry_text:
                incoming_nums = set(re.findall(r'\b[\d][\d,\.]*[\d]\b|\b\d\b', query_text.lower()))
                cached_nums = set(re.findall(r'\b[\d][\d,\.]*[\d]\b|\b\d\b', best_entry_text.lower()))
                if incoming_nums != cached_nums:
                    logger.info(
                        f"Semantic Cache REJECTED (number mismatch): "
                        f"query_nums={incoming_nums} vs cached_nums={cached_nums}"
                    )
                    return None

            logger.info(f"Semantic Cache HIT: score={best_score:.4f}")
            return best_key

        return None

    def add_semantic(
        self, query_embedding: list[float], cache_key: str, query_text: Optional[str] = None
    ) -> None:
        """Add a new query embedding to the semantic registry.
        
        Stores the original query text alongside the vector so the number/year
        guard in get_semantic() can reject false-positive cache hits.

        Works for both diskcache and Redis backends. The read-modify-write is
        guarded by a lock to avoid concurrent requests overwriting each other's
        additions.
        """
        try:
            with self._semantic_lock:
                registry = self._load_semantic_registry()
                registry.append({
                    "vector": query_embedding,
                    "key": cache_key,
                    "query": query_text or "",
                })
                # Keep registry size manageable for linear scan (last 1000 items)
                if len(registry) > 1000:
                    registry = registry[-1000:]
                self._save_semantic_registry(registry)
        except Exception as e:
            logger.warning(f"Failed to update semantic registry: {e}")

    def clear(self) -> None:
        """Wipe this app's cache entries and semantic registry.

        Scoped to our own key prefixes (rag_exact:, rewrite_cache:,
        semantic_registry) instead of Redis flushdb(), which would wipe the
        entire database and any other app sharing that Redis instance."""
        try:
            if self.cache_type == "redis" and self.redis_client:
                deleted = 0
                for pattern in ("rag_exact:*", "rewrite_cache:*"):
                    keys = list(self.redis_client.scan_iter(match=pattern, count=500))
                    if keys:
                        deleted += self.redis_client.delete(*keys)
                deleted += self.redis_client.delete(_SEMANTIC_REGISTRY_KEY)
                logger.info(f"Redis cache cleared ({deleted} keys removed).")
            elif self.cache_type == "diskcache" and self.disk_cache is not None:
                self.disk_cache.clear()
                logger.info("Diskcache cleared.")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
