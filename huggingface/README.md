---
title: DocuMind
emoji: "🧠"
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# DocuMind (Docker Space)

This Space runs DocuMind as a single Docker container:

- FastAPI backend (internal): `http://127.0.0.1:8000`
- Streamlit UI (public): `http://0.0.0.0:${PORT}` (Hugging Face usually sets `PORT=7860`)

## Required secrets / variables (Space Settings)

- `GEMINI_API_KEY` (Secret)

Recommended for Spaces:

- `QDRANT_URL` / `QDRANT_API_KEY` (Qdrant server/cloud; preferred over local storage on Spaces)
- `DOCUMIND_DELETE_UPLOADED_PDFS=1` (if you want to delete PDFs after indexing)

## Notes

- This Space is self-contained: Streamlit calls the backend at `http://127.0.0.1:8000` inside the same container.
- First build can take a while because ML dependencies may be large.
