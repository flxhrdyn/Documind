#!/usr/bin/env bash
set -euo pipefail

# Hugging Face Spaces typically provides PORT=7860.
PORT="${PORT:-7860}"

# Resolve Streamlit security/proxy flags.
# - Local/dev default: secure (true/true)
# - HF Spaces default: proxy-friendly (false/false)
# - Explicit env vars always win.
ON_HF_SPACES="0"
if [[ -n "${SPACE_ID:-}" || -n "${SPACE_HOST:-}" ]]; then
  ON_HF_SPACES="1"
fi

if [[ -n "${STREAMLIT_ENABLE_CORS:-}" ]]; then
  STREAMLIT_CORS_VALUE="${STREAMLIT_ENABLE_CORS}"
elif [[ "${ON_HF_SPACES}" == "1" ]]; then
  STREAMLIT_CORS_VALUE="false"
else
  STREAMLIT_CORS_VALUE="true"
fi

if [[ -n "${STREAMLIT_ENABLE_XSRF_PROTECTION:-}" ]]; then
  STREAMLIT_XSRF_VALUE="${STREAMLIT_ENABLE_XSRF_PROTECTION}"
elif [[ "${ON_HF_SPACES}" == "1" ]]; then
  STREAMLIT_XSRF_VALUE="false"
else
  STREAMLIT_XSRF_VALUE="true"
fi

# Start FastAPI backend (internal-only).
uvicorn app.main:app --host 127.0.0.1 --port 8000 &
UVICORN_PID=$!

cleanup() {
  kill "${UVICORN_PID}" >/dev/null 2>&1 || true
  wait "${UVICORN_PID}" >/dev/null 2>&1 || true
}

trap cleanup EXIT

# Start Streamlit (public).
exec streamlit run frontend/streamlit_app.py \
  --server.address=0.0.0.0 \
  --server.port="${PORT}" \
  --server.headless=true \
  --server.enableCORS="${STREAMLIT_CORS_VALUE}" \
  --server.enableXsrfProtection="${STREAMLIT_XSRF_VALUE}"