FROM python:3.12

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Install dependencies first so changes in app code do not invalidate this layer.
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Copy runtime sources after dependency installation.
COPY app ./app
COPY frontend ./frontend
COPY start.sh ./start.sh

# Ensure expected writable directories exist (for local Qdrant + uploads).
RUN mkdir -p uploaded_docs qdrant_storage \
    && chmod +x start.sh

EXPOSE 7860

CMD ["bash", "start.sh"]