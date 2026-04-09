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

# Reranking
RERANK_TOP_K = 5

# Models
LLM_MODEL = "gemini-3.1-flash-lite-preview"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = (os.getenv("RERANKER_MODEL") or "cross-encoder/ms-marco-MiniLM-L-6-v2").strip()

# API Keys
_gemini_api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
GEMINI_API_KEY = _gemini_api_key or None