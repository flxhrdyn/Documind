# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

InvenioAI - advanced RAG system for document Q&A over PDFs. FastAPI backend + Streamlit frontend, Qdrant
hybrid (dense + sparse BM42) retrieval, RAG Fusion multi-query, cross-encoder reranking, CoT reasoning via
Groq (Llama 3.3/3.1), dual-layer semantic caching.

## Commands

Setup:
```bash
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env            # set GROQ_API_KEY etc.
```

Run backend (from `backend/`, package root is `app`):
```bash
cd backend && uvicorn app.main:app --reload
```

Run frontend:
```bash
streamlit run frontend/streamlit_app.py
```

Run both together (Linux/Mac, mirrors HF Space startup):
```bash
./start.sh
```
Windows equivalent: `run_local.bat`.

Docker (single container serving both backend + frontend, used for HF Spaces/Azure):
```bash
docker build -t invenioai .
docker run -p 7860:7860 invenioai
```
Multi-container (backend/frontend/redis) via `docker-compose.yml`.

Tests (pytest config lives in `pyproject.toml`: `testpaths = ["backend/tests"]`, `pythonpath = ["backend"]`,
`asyncio_mode = "strict"`):
```bash
pytest                                   # run all tests
pytest backend/tests/test_hybrid_retrieval.py -k some_case   # single test
```

Init/reset the vector DB:
```bash
python backend/scripts/init_vector_db.py
```

## Architecture

Backend is a flat module layout under `backend/app/` (no sub-packages) — imports are relative (e.g.
`from .rag_pipeline import rag_pipeline`):

- `main.py` - FastAPI entrypoint. Exposes `POST /query` (sync) and `POST /query/jobs` (async, in-memory job
  polling — job state resets on process restart). `lifespan` preloads dense/sparse embedding models,
  reranker, and Qdrant client on startup (`PRELOAD_EMBEDDINGS_ON_STARTUP` in `config.py`).
- `index_api.py` / `index_data.py` - PDF ingestion endpoints and indexing pipeline (LlamaParse extraction,
  header/footer stripping, structure-aware chunking via `MarkdownHeaderTextSplitter` +
  `RecursiveCharacterTextSplitter`, cross-page heading propagation).
- `embeddings.py` - dense (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) and sparse
  (`Qdrant/bm42-all-minilm-l6-v2-attentions`) embedding singletons.
- `qdrant_conn.py` - Qdrant client lifecycle (local storage by default, or `QDRANT_URL` for
  server/cloud).
- `retriever.py` - hybrid retrieval (dense MMR + sparse BM42, native Qdrant fusion) and RAG Fusion
  multi-query expansion.
- `reranker.py` - FlashRank cross-encoder (`ms-marco-MiniLM-L-12-v2`) reranking of retrieved candidates.
- `rag_pipeline.py` - orchestrates query rewriting → semantic cache lookup → hybrid retrieval → rerank →
  CoT-structured (4-step: deconstruction, filtering, synthesis, strategy) Groq LLM generation.
  `rag_pipeline` is a module-level singleton used by `main.py`.
- `cache_manager.py` - dual-layer cache: exact-match + semantic similarity (cosine > 0.90) via
  DiskCache or Redis (`CACHE_TYPE` env var).
- `metrics.py` - tracks latency, retrieval/generation efficiency, and IR metrics (nDCG, HitRate);
  persisted to `metrics.json` and surfaced through the Streamlit analytics dashboard.
- `config.py` - all env-driven settings (`GROQ_API_KEY`, `QDRANT_URL`, `INVENIOAI_ENABLE_HYBRID_SEARCH`,
  `INVENIOAI_DELETE_UPLOADED_PDFS`, `INVENIOAI_LLM_MODEL`, `CACHE_TYPE`, `REDIS_URL`, etc.). See
  `.env.example` for the full list.

Frontend (`frontend/`) is a Streamlit app (`streamlit_app.py` + `pages/dashboard.py` for analytics,
`theme.py` for the custom CSS design system) that talks to the backend over REST
(`INVENIOAI_API_BASE_URL`, defaults to `http://backend:8000` in Docker Compose).

Data flow: PDF upload → `index_api` → LlamaParse + chunking → Qdrant (dense+sparse) → query comes in via
`main.py` → `rag_pipeline` checks semantic cache → on miss, hybrid retrieve from Qdrant → rerank → CoT
generation via Groq → response + metrics logged.

## Notes

- Python 3.12 only (`requires-python = ">=3.12,<3.13"`), dependency/lock managed with `uv` (`uv.lock`).
- Backend and frontend have separate `requirements.txt`/Dockerfiles for independent container builds, but
  the root `Dockerfile` bundles both into one image for the primary deployment target (HF Spaces).
