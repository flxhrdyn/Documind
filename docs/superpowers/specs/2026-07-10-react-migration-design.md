# React Migration Design

**Date:** 2026-07-10
**Status:** Approved
**Topic:** Migrate InvenioAI frontend from Streamlit to React, redesign visual style, split deployment.

## Goal

Replace the Streamlit frontend with a React single-page app that talks to the existing FastAPI backend.
Keep the same structure and features, but redesign the visual style away from the generic Google-AI-Studio
Tailwind look toward a clean, modern "warm editorial" aesthetic (warm palette, display type for headings,
crafted rather than corporate, not over the top).

The existing `invenioai-ui-ref/` React app captures the desired structure and feature set but is
mock-data-only and styled in the generic look we are moving away from. It serves as a starting point for
structure and component boundaries, not for visual style.

## Non-Goals

- No changes to the backend RAG pipeline logic (retrieval, rerank, CoT generation).
- No rewrite of the Streamlit app. The existing HF Space stays live, untouched, as an archived demo.
- No user accounts / auth. Backend runs without `INVENIOAI_API_KEY` in the new deployment.

## Decisions

| Area | Decision |
|------|----------|
| Framework | React + Vite + TypeScript (reuse `invenioai-ui-ref/` as starting point) |
| Visual direction | Warm editorial: warm palette (cream/charcoal), display font for headings, generous whitespace, crafted feel |
| Routing | Route-based via React Router. `/` redirects to `/chat`; routes `/chat`, `/analytics` |
| Styling | Tailwind with custom warm-editorial theme tokens (palette + fonts) in `tailwind.config` |
| Data layer | TanStack Query for server state (documents, metrics); custom hook + `fetch`/`ReadableStream` for SSE |
| Chat persistence | `localStorage` (namespaced key), survives refresh |
| Auth / API key | Skipped. No secret env vars in the frontend build. Public backend URL only |
| Frontend deploy | Vercel static build. `VITE_API_BASE_URL` points to Azure backend |
| Backend deploy | New container (backend-only, no Streamlit) on Azure Container Apps |
| CORS | `INVENIOAI_ALLOWED_ORIGINS` restricted to the Vercel domain (middleware already exists in `main.py`) |
| Old Streamlit HF Space | Left idle as an archived demo, not maintained |
| Repo strategy | Single repo, feature branch, merge to `main` via PR |

## Architecture

- **Location**: new `frontend-react/` folder in the existing repo. Deploy target differs from backend, but
  both live in one repo.
- **Routing**: React Router. `/` -> redirect `/chat`. Routes: `/chat`, `/analytics`. Real URLs enable
  deep-linking and sharing (e.g. link straight to the analytics dashboard) - a portfolio win.
- **Data layer**: TanStack Query owns `documents` and `metrics` server state (caching, refetch,
  loading/error). SSE `/query/stream` is not native to Query, so a custom hook consumes the stream via
  `fetch` + `ReadableStream`; on completion it invalidates the relevant Query caches (documents, metrics).
- **Local state**: chat history in React state, synced to `localStorage` under a namespaced key.
- **Backend changes**: `CORSMiddleware` already exists in `main.py` reading `INVENIOAI_ALLOWED_ORIGINS`; no
  code change needed beyond setting that env at deploy. New container image variant serves FastAPI only (no
  Streamlit / `start.sh`).

## Pages & Components

- **Layout**: persistent sidebar (brand, `ThemeToggle`, `UploadPanel`, `KnowledgeBaseList`) + main content
  area that swaps per route.
- **`/chat`**:
  - `ChatPanel` - real SSE streaming from `/query/stream` (not the ui-ref word-by-word simulation). Renders
    CoT "Thought Process" as a collapsible section, not raw text.
  - `SourcesPanel` - grounded citations, click-to-scroll NotebookLM-style flow (as in ui-ref).
- **`/analytics`**:
  - `MetricsDashboard` fetches real `/metrics`. KPI cards (HitRate@k, nDCG@k, avg latency, indexed docs),
    response-time trend chart (recharts, already a ui-ref dependency), query history table, and an
    "Advanced Metrics" expander (Precision/Recall/MRR + definitions).
- **Upload flow**: `UploadPanel` calls real `/upload/jobs`, polls status with the same backoff as Streamlit
  (1s -> 3s -> 5s cap), renders progress states (pending/parsing/indexing/succeeded/failed).
- **KnowledgeBaseList**: fetches `/documents`; per-file delete and delete-all use a custom two-step confirm
  modal (not the native browser confirm) to stay on-style.

## Deployment

- **Repo**: one repo, feature branch, merged to `main` via PR when ready.
- **Backend**: reuse the `Dockerfile` pattern, backend-only (no Streamlit), deployed to Azure Container Apps
  (fits a FastAPI container that loads reranker/embedding models on startup; scale-to-zero optional for cost).
- **CORS**: `INVENIOAI_ALLOWED_ORIGINS` set to the Vercel domain.
- **Frontend**: Vercel, `VITE_API_BASE_URL` points to the Azure backend URL.
- **Secrets on Azure**: `GROQ_API_KEY`, `QDRANT_URL`, etc. stored as Azure Container Apps secrets, never in
  the repo.
- **Old HF Space (Streamlit)**: left idle, not touched, not maintained - archived demo.

## Security Note

The Vercel April 2026 supply-chain breach exposed customer environment variables stored on Vercel. Because
this frontend stores no secrets on Vercel (static build, public backend URL only), exposure is effectively
nil. If a secret is ever added later (e.g. a signed-URL token), adopt routine credential rotation.

## Backend API Surface (existing, consumed by React)

- `POST /query/stream` - SSE: steps `cached`, `rewriting`, `retrieving`, `reranking`, `generating`,
  `thinking`, `token`, `done`, `error`.
- `POST /upload/jobs` -> `job_id`; `GET /upload/jobs/{job_id}` -> status polling.
- `GET /documents` -> indexed filenames.
- `DELETE /documents/delete?filename=` - delete one; `DELETE /documents` - delete all.
- `GET /metrics`, `POST /metrics/sync` - analytics data.
