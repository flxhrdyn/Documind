# 🧠 DocuMind — Document Q&A (RAG) + Analytics

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-teal?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-6C63FF)
![Gemini](https://img.shields.io/badge/Google-Gemini-4285F4?logo=google)

**DocuMind** is an MVP-ready document question-answering system built with **FastAPI** (backend) and **Streamlit** (frontend). It runs a **RAG pipeline** (query rewriting → retrieval → reranking → answer generation) backed by **Qdrant**.

---

## 🌟 Features

- **📄 PDF indexing**: Upload and index multiple PDF documents
- **🔍 RAG pipeline**: Query rewriting + semantic retrieval + reranking
- **⚡ Vector search**: Qdrant vector store (local or server/cloud)
- **📊 Analytics**: Local metrics file + a Streamlit dashboard page
- **🔐 Config via env**: `.env` / `.env.example` for secrets and runtime flags

---

## 🔗 Live Demo

This repo does not ship public demo links by default. Local endpoints:

- Frontend (Streamlit): `http://localhost:8501`
- Backend (FastAPI): `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

For hosted deployments, this repo provides:

- Dockerfiles for backend and frontend
- A GitHub Actions workflow to deploy the **Streamlit frontend** to Hugging Face Spaces

---

## 🎯 Problem Statement

When you have a set of PDF documents, it’s tedious to manually search for answers. DocuMind turns PDFs into a searchable knowledge base and lets you ask questions with answers grounded in retrieved context, while tracking basic performance metrics.

---

## 🏗️ RAG Pipeline & Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│  Frontend   │─────▶│   FastAPI    │─────▶│     LLM     │
│ (Streamlit) │      │   Backend    │      │   (Gemini)  │
└─────────────┘      └──────────────┘      └─────────────┘
       │                    │
       │                    ▼
       │             ┌──────────────┐
       │             │  RAG Pipeline│
       │             └──────────────┘
       │                    │
       ▼                    ▼
┌─────────────┐      ┌──────────────┐
│  Metrics    │      │   Qdrant     │
│ (local JSON)│      │ Vector Store │
└─────────────┘      └──────────────┘
```

Pipeline (high-level):

1. Rewrite question into a standalone query (LLM)
2. Retrieve relevant chunks from Qdrant (MultiQueryRetriever)
3. Rerank top chunks (CrossEncoder)
4. Generate answer constrained to retrieved context (LLM)

---

## 🛠️ Tech Stack

| Layer | Tech |
|------:|------|
| Backend API | FastAPI + Uvicorn |
| Frontend UI | Streamlit |
| Vector DB | Qdrant (local or server/cloud) |
| RAG | LangChain + langchain-qdrant |
| LLM | Google Gemini (via `langchain-google-genai`) |
| Embeddings | Sentence Transformers |
| Reranker | CrossEncoder (sentence-transformers) |
| Testing | pytest |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Google Gemini API key

### Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set at minimum:

```bash
GEMINI_API_KEY=...
```

### Run backend

```bash
uvicorn app.main:app --reload
```

### Run frontend

```bash
streamlit run frontend/streamlit_app.py
```

---

## 📖 Usage

1. Upload a PDF from the sidebar
2. Click **Index Document**
3. Ask questions in the chat input
4. Open the **Dashboard** page to see metrics

---

## 🧪 Tests

```bash
pip install -r requirements-dev.txt
pytest
```

---

## 📊 Dashboard

The Streamlit dashboard page shows:

- Total queries
- Total documents indexed
- Average response time
- Per-query history (timings + retrieval scores when available)

Note: metrics are written to a local `metrics.json` file and are not shared across multiple instances.

---

## 🔧 Configuration

Common settings (see `.env.example`):

```bash
# Required
GEMINI_API_KEY=...

# Optional: Qdrant server/cloud (recommended for multi-process)
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_PREFER_GRPC=0

# Optional: Streamlit -> backend URL
DOCUMIND_API_BASE_URL=http://localhost:8000

# Optional: delete PDFs after indexing
DOCUMIND_DELETE_UPLOADED_PDFS=0

# Optional: Hugging Face Hub timeouts
HF_HUB_CONNECT_TIMEOUT=30
HF_HUB_READ_TIMEOUT=120
```

### Qdrant local storage lock (important)

By default the project can use **local Qdrant storage** at `qdrant_storage/`. Local mode can be accessed by only **one client instance per process**, and can fail under multi-process/concurrent usage.

If you see:

`RuntimeError: Storage folder ... is already accessed by another instance of Qdrant client`

use **Qdrant server/cloud mode** by setting `QDRANT_URL`.

---

## 🐳 Deployment

### Docker (backend)

```bash
docker build -t documind-api .
docker run -p 8000:8000 documind-api
```

### Docker (frontend)

```bash
docker build -t documind-ui -f Dockerfile.streamlit .
docker run -p 8501:8501 \
  -e DOCUMIND_API_BASE_URL=http://host.docker.internal:8000 \
  documind-ui
```

### Hugging Face Spaces (Docker) via CI/CD

This repo includes a GitHub Actions workflow that syncs a clean “snapshot” of the repo to a Hugging Face **Docker** Space on every push to `main`.

The Space runs a single container that starts:

- FastAPI backend (internal): `127.0.0.1:8000`
- Streamlit UI (public): `0.0.0.0:7860` (Hugging Face provides `PORT=7860`)

1) Create a **Docker** Space on Hugging Face.

2) In Space **Settings → Variables / Secrets**, set:

- **Secret**: `GEMINI_API_KEY`

Optional (advanced): `QDRANT_URL`, `QDRANT_API_KEY`, `DOCUMIND_DELETE_UPLOADED_PDFS=1`

3) In GitHub **Settings → Secrets and variables → Actions**, add:

- `HF_TOKEN`: Hugging Face access token with write access
- `HF_SPACE_ID`: `username/space-name`
- (Optional) `HF_SPACE_BRANCH`: defaults to `main`

Workflow file: `.github/workflows/sync-to-hf-space.yml`

---

## 📁 Project Structure

```
documind-test/
├── app/                     # FastAPI backend + RAG pipeline
├── frontend/                # Streamlit UI + dashboard page
├── tests/                   # Unit tests
├── uploaded_docs/           # Local PDF storage (optional)
├── qdrant_storage/          # Local Qdrant storage (optional)
├── .env.example
├── Dockerfile
├── Dockerfile.streamlit
├── requirements.txt
└── readme.md
```

---

## ⚠️ Limitations (current MVP)

- `/query/jobs` state is stored in memory (lost on restart)
- Dashboard metrics are stored in a local `metrics.json` file (not shared across instances)
- No authentication / rate limiting built-in
