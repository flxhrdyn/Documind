"""Application configuration.

Importing `app.config` loads `.env` and sets a few environment defaults used by
the Hugging Face stack. Those defaults need to be in place before heavy ML
libraries are imported.
"""

from pathlib import Path
import os

from dotenv import load_dotenv


load_dotenv(override=False)

BASE_DIR = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: str = "0") -> bool:
    return (os.getenv(name, default) or "").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, min_value: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return max(default, min_value)
    try:
        value = int(raw)
    except ValueError:
        return max(default, min_value)
    return max(value, min_value)


def _env_float(name: str, default: float, *, min_value: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return max(default, min_value)
    try:
        value = float(raw)
    except ValueError:
        return max(default, min_value)
    return max(value, min_value)


def _env_str(name: str, default: str) -> str:
    raw = (os.getenv(name) or "").strip()
    return raw if raw else default


def _configure_huggingface_hub_defaults() -> None:
    """Set safe HF Hub defaults (timeouts, offline mode, telemetry).

    Values are applied via `os.environ.setdefault(...)` so user-provided env
    vars always take precedence.
    """

    # Avoid noisy telemetry by default.
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    # Hugging Face Hub HTTP timeouts. The default read timeout (10s) is often
    # too aggressive on slow networks.
    os.environ.setdefault("HF_HUB_CONNECT_TIMEOUT", os.getenv("HF_HUB_CONNECT_TIMEOUT", "30"))
    os.environ.setdefault("HF_HUB_READ_TIMEOUT", os.getenv("HF_HUB_READ_TIMEOUT", "60"))

    # Optional offline mode (if you have pre-downloaded models into the cache).
    if os.getenv("HF_HUB_OFFLINE") == "1":
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


_configure_huggingface_hub_defaults()

# Qdrant
QDRANT_PATH = str(BASE_DIR / "qdrant_storage")
QDRANT_COLLECTION = "invenioai_collection"

# Optional: use Qdrant server/cloud instead of local storage.
_qdrant_url = (os.getenv("QDRANT_URL") or "").strip()
QDRANT_URL = _qdrant_url or None  # e.g. http://localhost:6333

_qdrant_api_key = (os.getenv("QDRANT_API_KEY") or "").strip()
QDRANT_API_KEY = _qdrant_api_key or None
UPLOAD_DIR = str(BASE_DIR / "uploaded_docs")
METRICS_FILE = str(BASE_DIR / "metrics.json")

# When enabled, uploaded PDFs are deleted from local disk after indexing.
# Useful for deployments with ephemeral disks.
DELETE_UPLOADED_PDFS = (
    (os.getenv("INVENIOAI_DELETE_UPLOADED_PDFS") or os.getenv("DELETE_UPLOADED_PDFS") or "0").strip() == "1"
)

# Chunking
CHUNK_SIZE = _env_int("INVENIOAI_CHUNK_SIZE", default=1500, min_value=128)
CHUNK_OVERLAP = _env_int("INVENIOAI_CHUNK_OVERLAP", default=100, min_value=0)
if CHUNK_OVERLAP >= CHUNK_SIZE:
    CHUNK_OVERLAP = max(0, CHUNK_SIZE // 4)

# Smaller default batch is safer for constrained runtimes (HF Spaces free tier,
# small containers) and often avoids long stalls from memory pressure.
INDEXING_BATCH_SIZE = _env_int("INVENIOAI_INDEXING_BATCH_SIZE", default=16, min_value=8)

# Startup preload can make the first request faster, but on constrained
# deployments it may increase cold-start time and memory pressure.
PRELOAD_EMBEDDINGS_ON_STARTUP = _env_bool("INVENIOAI_PRELOAD_EMBEDDINGS", default="1")

# Retrieval
RETRIEVAL_K = 20

# Hybrid retrieval (dense + sparse/lexical) settings.
# Hybrid uses Qdrant's native server-side RRF fusion (dense + sparse/BM42).
# Set INVENIOAI_ENABLE_HYBRID_SEARCH=0 to force dense-only.
# Note: langchain-qdrant's QdrantVectorStore doesn't expose fusion weight
# knobs, so there is no dense/sparse weight setting here - RRF fusion order
# is the only lever available through this library.
USE_HYBRID_SEARCH = _env_bool("INVENIOAI_ENABLE_HYBRID_SEARCH", default="1")
SPARSE_MODEL_NAME = _env_str("INVENIOAI_SPARSE_MODEL_NAME", "Qdrant/bm42-all-minilm-l6-v2-attentions")

# Reranking
RERANK_TOP_K = 7

# IR quality metrics (precision/recall/MRR/hit-rate in the analytics dashboard)
# treat a reranker score >= this threshold as "relevant". This assumes the
# active RERANKER_MODEL's scores are roughly normalized to [0, 1] - if you
# swap reranker models, verify their score distribution and recalibrate this.
IR_RELEVANCE_THRESHOLD = _env_float("INVENIOAI_IR_RELEVANCE_THRESHOLD", default=0.7, min_value=0.0)

# Models
LLM_MODEL = _env_str("INVENIOAI_LLM_MODEL", "llama-3.1-8b-instant")
EMBEDDING_MODEL = _env_str("INVENIOAI_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
RERANKER_MODEL = _env_str("INVENIOAI_RERANKER_MODEL", "ms-marco-MultiBERT-L-12")

# API Keys
_groq_api_key = (os.getenv("GROQ_API_KEY") or "").strip()
GROQ_API_KEY = _groq_api_key or None

# Optional API key. When set, protected endpoints (upload, query, document
# management, metrics) require a matching `X-API-Key` header. Leave unset for
# local/demo use (auth disabled) - this preserves today's default behavior.
_api_key = (os.getenv("INVENIOAI_API_KEY") or "").strip()
API_KEY = _api_key or None

# Max accepted upload size in megabytes (server-side; independent of any
# client-side limit like Streamlit's own uploader cap).
MAX_UPLOAD_SIZE_MB = _env_int("INVENIOAI_MAX_UPLOAD_SIZE_MB", default=100, min_value=1)

# CORS: comma-separated list of allowed origins for browser clients (e.g. a
# future React frontend). Defaults to "*" since the bundled Streamlit
# frontend talks to this API server-side (no browser CORS involved).
_allowed_origins_raw = (os.getenv("INVENIOAI_ALLOWED_ORIGINS") or "*").strip()
ALLOWED_ORIGINS = [o.strip() for o in _allowed_origins_raw.split(",") if o.strip()]

# Caching
CACHE_TYPE = _env_str("CACHE_TYPE", "diskcache") # 'redis' or 'diskcache'
REDIS_URL = _env_str("REDIS_URL", "redis://localhost:6379/0")

# RAG Fusion
NUM_FUSION_QUERIES = _env_int("INVENIOAI_NUM_FUSION_QUERIES", default=3, min_value=1)