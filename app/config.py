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
QDRANT_COLLECTION = "documind_collection"

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
    (os.getenv("DOCUMIND_DELETE_UPLOADED_PDFS") or os.getenv("DELETE_UPLOADED_PDFS") or "0").strip() == "1"
)

# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Retrieval
RETRIEVAL_K = 10

# Hybrid retrieval (dense + lexical) settings.
# Hybrid uses weighted reciprocal-rank fusion (RRF) before reranking and is
# enabled by default. Set DOCUMIND_ENABLE_HYBRID_SEARCH=0 to force dense-only.
USE_HYBRID_SEARCH = _env_bool("DOCUMIND_ENABLE_HYBRID_SEARCH", default="1")
HYBRID_LEXICAL_K = _env_int("DOCUMIND_HYBRID_LEXICAL_K", default=10, min_value=1)
HYBRID_FUSION_LIMIT = _env_int("DOCUMIND_HYBRID_FUSION_LIMIT", default=20, min_value=1)
HYBRID_MAX_LEXICAL_DOCS = _env_int("DOCUMIND_HYBRID_MAX_LEXICAL_DOCS", default=3000, min_value=100)
HYBRID_RRF_K = _env_int("DOCUMIND_HYBRID_RRF_K", default=60, min_value=1)
HYBRID_DENSE_WEIGHT = _env_float("DOCUMIND_HYBRID_DENSE_WEIGHT", default=1.0, min_value=0.0)
HYBRID_LEXICAL_WEIGHT = _env_float("DOCUMIND_HYBRID_LEXICAL_WEIGHT", default=1.0, min_value=0.0)

# Reranking
RERANK_TOP_K = 5

# Models
LLM_MODEL = "gemini-3.1-flash-lite-preview"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = (os.getenv("RERANKER_MODEL") or "cross-encoder/ms-marco-MiniLM-L-6-v2").strip()

# API Keys
_gemini_api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
GEMINI_API_KEY = _gemini_api_key or None