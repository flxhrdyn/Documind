# Perbaikan Teknis Backend (7 temuan terverifikasi) - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**File output:** Plan final disimpan ke `docs/superpowers/plans/2026-07-10-perbaikan-teknis-backend.md` (mengikuti pola `docs/audit-*.md` yang sudah ada di repo ini).

**Goal:** Menutup 7 temuan konkret dari audit lanjutan (semua sudah diverifikasi langsung terhadap kode saat ini, bukan asumsi) di backend InvenioAI: docstring stale, tanpa retry Groq, threshold cache hardcoded tersebar, race condition metrics lintas proses, tanpa validasi magic-byte upload, tanpa startup warning API key kosong, dan gap test coverage.

**Architecture:** Setiap task adalah perbaikan kecil dan independen (bug fix/hardening, bukan fitur baru) di modul backend yang sudah ada (`main.py`, `llm.py`, `config.py`, `cache_manager.py`, `rag_pipeline.py`, `metrics.py`, `index_api.py`) plus test baru/tambahan yang mengunci perilaku tersebut. Tidak ada perubahan arsitektur - pola yang dipakai (env-driven config lewat `_env_*` helper, `lru_cache` singleton, `threading.Lock`, pytest+httpx `AsyncClient`) semuanya sudah ada di codebase dan diikuti apa adanya.

**Tech Stack:** FastAPI, langchain-groq (`ChatGroq`), pytest + pytest-asyncio + httpx, `filelock` (dependency baru untuk lock lintas proses).

## Global Constraints

- Python 3.12, dependency dikelola `uv` (`pyproject.toml` + `uv.lock`).
- Ikuti pola env-config yang sudah ada di `backend/app/config.py` (`_env_bool`/`_env_int`/`_env_float`/`_env_str`) - jangan baca `os.getenv` langsung di modul lain.
- Semua field baru di `config.py` harus punya default aman yang tidak mengubah perilaku hari ini kalau env var tidak di-set.
- Test baru mengikuti pola yang sudah ada: `httpx.AsyncClient` + `ASGITransport` untuk endpoint test (lihat `test_main.py`, `test_metrics_api.py`), `unittest.mock.patch` untuk isolasi dependency eksternal (Groq, Qdrant).
- Jangan tambah abstraksi baru (no wrapper class, no feature flag) di luar yang diminta tiap task.

---

### Task 1: Perbaiki docstring stale di `main.py`

**Files:**
- Modify: `backend/app/main.py:1-9`

**Interfaces:** Tidak ada perubahan interface, murni dokumentasi.

- [ ] **Step 1: Ganti docstring modul**

Ganti isi baris 1-9 (docstring di puncak file) dari:

```python
"""FastAPI application entrypoint.

Wires the API router and exposes two ways to query the RAG pipeline:

- `POST /query` for a simple request/response flow.
- `POST /query/jobs` for background execution with polling.

Job state is stored in-memory, so it resets on process restart.
"""
```

menjadi:

```python
"""FastAPI application entrypoint.

Wires the API router and exposes two ways to query the RAG pipeline:

- `POST /query` for a simple synchronous request/response flow.
- `POST /query/stream` for an async, Server-Sent Events (SSE) streamed
  response.

Document upload/indexing has its own background-job + polling flow; see
`index_api.py` (`POST /upload/jobs`, `GET /upload/jobs/{job_id}`). That job
state is stored in-memory, so it resets on process restart.
"""
```

- [ ] **Step 2: Verifikasi tidak ada referensi lain ke `/query/jobs`**

Run: `grep -rn "query/jobs" backend/ frontend/ README.md`
Expected: tidak ada hasil (kosong) selain README/audit docs historis yang boleh dibiarkan.

- [ ] **Step 3: Commit**

```bash
git add backend/app/main.py
git commit -m "docs(backend): fix stale /query/jobs docstring in main.py"
```

---

### Task 2: Tambah timeout & retry ke Groq LLM client

**Files:**
- Modify: `backend/app/config.py` (tambah 2 env var baru, setelah blok `# API Keys`)
- Modify: `backend/app/llm.py`
- Test: `backend/tests/test_llm.py` (baru)
- Modify: `backend/tests/conftest.py` (tambah `get_llm.cache_clear()` ke fixture `_clear_singleton_caches`)

**Interfaces:**
- Produces: `config.GROQ_TIMEOUT_SECONDS: int`, `config.GROQ_MAX_RETRIES: int` - dipakai `llm.get_llm()`.

- [ ] **Step 1: Tambah env var di `config.py`**

Setelah blok berikut di `config.py` (sekitar baris 132-134):

```python
# API Keys
_groq_api_key = (os.getenv("GROQ_API_KEY") or "").strip()
GROQ_API_KEY = _groq_api_key or None
```

tambahkan:

```python
# Groq client resilience: transient errors (rate limits, network blips) are
# retried by the underlying Groq SDK client itself before raising.
GROQ_TIMEOUT_SECONDS = _env_int("INVENIOAI_GROQ_TIMEOUT_SECONDS", default=30, min_value=1)
GROQ_MAX_RETRIES = _env_int("INVENIOAI_GROQ_MAX_RETRIES", default=3, min_value=0)
```

- [ ] **Step 2: Write the failing test**

Create `backend/tests/test_llm.py`:

```python
from unittest.mock import patch, MagicMock

import pytest

from app import llm


@pytest.fixture(autouse=True)
def _reset_llm_cache():
    llm.get_llm.cache_clear()
    yield
    llm.get_llm.cache_clear()


def test_get_llm_raises_without_api_key():
    with patch.object(llm, "GROQ_API_KEY", None):
        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            llm.get_llm()


def test_get_llm_passes_timeout_and_retries():
    with patch.object(llm, "GROQ_API_KEY", "fake-key"), \
         patch.object(llm, "GROQ_TIMEOUT_SECONDS", 42), \
         patch.object(llm, "GROQ_MAX_RETRIES", 5), \
         patch("app.llm.ChatGroq") as mock_chat_groq:
        mock_chat_groq.return_value = MagicMock()

        llm.get_llm()

        mock_chat_groq.assert_called_once()
        _, kwargs = mock_chat_groq.call_args
        assert kwargs["timeout"] == 42
        assert kwargs["max_retries"] == 5


def test_get_llm_is_a_singleton():
    with patch.object(llm, "GROQ_API_KEY", "fake-key"), \
         patch("app.llm.ChatGroq") as mock_chat_groq:
        mock_chat_groq.return_value = MagicMock()

        first = llm.get_llm()
        second = llm.get_llm()

        assert first is second
        mock_chat_groq.assert_called_once()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest backend/tests/test_llm.py -v`
Expected: FAIL - `test_get_llm_passes_timeout_and_retries` fails with `KeyError: 'timeout'` (kwargs belum ada), yang lain mungkin sudah pass secara kebetulan.

- [ ] **Step 4: Update `llm.py`**

Ganti isi `backend/app/llm.py` (baris 14-28) dari:

```python
from .config import GROQ_API_KEY, LLM_MODEL


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY belum di-set. Isi di .env (lihat .env.example) sebelum menjalankan query."
        )

    return ChatGroq(
        model=LLM_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=0,
    )
```

menjadi:

```python
from .config import GROQ_API_KEY, GROQ_MAX_RETRIES, GROQ_TIMEOUT_SECONDS, LLM_MODEL


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY belum di-set. Isi di .env (lihat .env.example) sebelum menjalankan query."
        )

    return ChatGroq(
        model=LLM_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=0,
        timeout=GROQ_TIMEOUT_SECONDS,
        max_retries=GROQ_MAX_RETRIES,
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest backend/tests/test_llm.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Tambahkan `get_llm.cache_clear()` ke fixture bersama di `conftest.py`**

Di `backend/tests/conftest.py`, pada fixture `_clear_singleton_caches` (baris 36-60), tambahkan blok analog untuk `llm.get_llm` di dua tempat (sebelum dan sesudah `yield`):

```python
@pytest.fixture(autouse=True)
def _clear_singleton_caches():
    """Clear module-level caches between tests.

    Some app modules cache heavy objects (e.g. embedding model) for runtime
    performance. Unit tests often patch constructors and expect a fresh object
    per test.
    """

    try:
        from app.embeddings import get_embeddings

        get_embeddings.cache_clear()
    except Exception:
        # If imports fail for any reason, don't block unrelated tests.
        pass

    try:
        from app.llm import get_llm

        get_llm.cache_clear()
    except Exception:
        pass

    yield

    try:
        from app.embeddings import get_embeddings

        get_embeddings.cache_clear()
    except Exception:
        pass

    try:
        from app.llm import get_llm

        get_llm.cache_clear()
    except Exception:
        pass
```

- [ ] **Step 7: Jalankan full suite untuk pastikan tidak ada regresi**

Run: `pytest backend/tests -v`
Expected: semua test PASS.

- [ ] **Step 8: Commit**

```bash
git add backend/app/config.py backend/app/llm.py backend/tests/test_llm.py backend/tests/conftest.py
git commit -m "fix(backend): add timeout and retry config to Groq LLM client"
```

---

### Task 3: Satukan threshold semantic cache ke `config.py`

**Files:**
- Modify: `backend/app/config.py`
- Modify: `backend/app/cache_manager.py:78-83`
- Modify: `backend/app/rag_pipeline.py:228`, `backend/app/rag_pipeline.py:437`
- Test: `backend/tests/test_cache_manager_semantic.py` (tambah 1 test)

**Interfaces:**
- Produces: `config.SEMANTIC_CACHE_THRESHOLD: float` (default `0.995`, sama seperti nilai yang dipakai `rag_pipeline.py` saat ini - tidak mengubah perilaku produksi).
- Consumes: helper `_env_float` yang sudah ada di `config.py` (Task ini reuse, bukan bikin baru).

- [ ] **Step 1: Tambah env var di `config.py`**

Tambahkan di `config.py`, setelah blok `# Caching` yang sudah ada (baris 152-154):

```python
# Caching
CACHE_TYPE = _env_str("CACHE_TYPE", "diskcache") # 'redis' or 'diskcache'
REDIS_URL = _env_str("REDIS_URL", "redis://localhost:6379/0")

# Cosine similarity threshold for the semantic cache (see cache_manager.py).
# 0.995 was chosen empirically for near-duplicate query matches; too low a
# threshold risks serving stale answers for questions that are merely similar
# in embedding space but differ semantically.
SEMANTIC_CACHE_THRESHOLD = _env_float("INVENIOAI_SEMANTIC_CACHE_THRESHOLD", default=0.995, min_value=0.0)
```

- [ ] **Step 2: Write the failing test**

Tambahkan di akhir `backend/tests/test_cache_manager_semantic.py`:

```python
def test_get_semantic_default_threshold_matches_config():
    """get_semantic()'s default threshold should come from config, not be a
    second hardcoded literal that can drift out of sync."""
    from app.config import SEMANTIC_CACHE_THRESHOLD
    import inspect

    from app.cache_manager import CacheManager

    default = inspect.signature(CacheManager.get_semantic).parameters["threshold"].default
    assert default == SEMANTIC_CACHE_THRESHOLD
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest backend/tests/test_cache_manager_semantic.py::test_get_semantic_default_threshold_matches_config -v`
Expected: FAIL - `assert 0.9 == 0.995`

- [ ] **Step 4: Update `cache_manager.py`**

Ganti import di baris 9 dari:

```python
from .config import BASE_DIR, CACHE_TYPE, REDIS_URL
```

menjadi:

```python
from .config import BASE_DIR, CACHE_TYPE, REDIS_URL, SEMANTIC_CACHE_THRESHOLD
```

Ganti signature `get_semantic` (baris 78-83) dari:

```python
    def get_semantic(
        self,
        query_embedding: list[float],
        threshold: float = 0.90,
        query_text: Optional[str] = None,
    ) -> Optional[str]:
```

menjadi:

```python
    def get_semantic(
        self,
        query_embedding: list[float],
        threshold: float = SEMANTIC_CACHE_THRESHOLD,
        query_text: Optional[str] = None,
    ) -> Optional[str]:
```

- [ ] **Step 5: Update kedua call site di `rag_pipeline.py`**

Baris 228:

```python
            semantic_key = cache.get_semantic(query_embedding, threshold=0.995, query_text=standalone_query)
```

menjadi (hapus argumen `threshold` eksplisit supaya pakai default dari config, bukan literal kedua yang bisa drift):

```python
            semantic_key = cache.get_semantic(query_embedding, query_text=standalone_query)
```

Baris 437, perubahan yang sama:

```python
        semantic_key = cache.get_semantic(query_embedding, threshold=0.995, query_text=standalone_query)
```

menjadi:

```python
        semantic_key = cache.get_semantic(query_embedding, query_text=standalone_query)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest backend/tests/test_cache_manager_semantic.py -v`
Expected: PASS (semua test termasuk yang baru)

- [ ] **Step 7: Jalankan test yang menyentuh `rag_pipeline.py` untuk regresi**

Run: `pytest backend/tests/test_caching.py backend/tests/test_semantic_caching_integration.py -v`
Expected: semua PASS (perilaku numerik tidak berubah, threshold efektif tetap 0.995).

- [ ] **Step 8: Commit**

```bash
git add backend/app/config.py backend/app/cache_manager.py backend/app/rag_pipeline.py backend/tests/test_cache_manager_semantic.py
git commit -m "refactor(backend): move semantic cache threshold to config, single source of truth"
```

---

### Task 4: Lindungi `metrics.json` dari race condition lintas proses

**Files:**
- Modify: `pyproject.toml` (tambah dependency `filelock`)
- Modify: `backend/app/metrics.py`
- Test: `backend/tests/test_metrics.py` (tambah 1 test)

**Interfaces:**
- Produces: modul-level `_metrics_file_lock` (FileLock) di `metrics.py`, dipakai bersama `_metrics_lock` (threading.Lock) yang sudah ada - tidak ada API publik baru, `log_query`/`log_document_indexed`/`sync_indexed_docs_count`/`reset_metrics` tetap sama signature-nya.

- [ ] **Step 1: Tambah dependency**

Di `pyproject.toml`, tambahkan `"filelock>=3.16.0",` ke list `dependencies` (setelah `"diskcache>=5.6.3",`):

```toml
    "diskcache>=5.6.3",
    "filelock>=3.16.0",
    "redis>=5.2.1",
```

Run: `uv sync`
Expected: `filelock` terinstal, `uv.lock` ter-update.

- [ ] **Step 2: Write the failing test**

Tambahkan di `backend/tests/test_metrics.py` (baca dulu isinya untuk ikuti pola fixture yang ada; kalau memakai `monkeypatch`/`tmp_path` untuk `METRICS_FILE`, ikuti pola yang sama). Tambahkan:

```python
def test_metrics_uses_cross_process_file_lock():
    """log_query() must be guarded by a FileLock, not just a threading.Lock,
    so multi-worker deployments (uvicorn --workers N) don't lose updates."""
    from app import metrics
    from filelock import FileLock

    assert isinstance(metrics._metrics_file_lock, FileLock)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest backend/tests/test_metrics.py::test_metrics_uses_cross_process_file_lock -v`
Expected: FAIL - `AttributeError: module 'app.metrics' has no attribute '_metrics_file_lock'`

- [ ] **Step 4: Update `metrics.py`**

Ganti import block (baris 6-19) dari:

```python
import logging
import json
import math
import os
import tempfile
from datetime import datetime
import threading
from typing import Any, Dict, List, Optional

from .config import IR_RELEVANCE_THRESHOLD, METRICS_FILE


logger = logging.getLogger(__name__)
_metrics_lock = threading.Lock()
```

menjadi:

```python
import logging
import json
import math
import os
import tempfile
from datetime import datetime
import threading
from typing import Any, Dict, List, Optional

from filelock import FileLock

from .config import IR_RELEVANCE_THRESHOLD, METRICS_FILE


logger = logging.getLogger(__name__)
# threading.Lock guards concurrent access within one process; FileLock guards
# the same read-modify-write cycle across processes (e.g. uvicorn --workers N
# or multiple container replicas sharing the same metrics.json path).
_metrics_lock = threading.Lock()
_metrics_file_lock = FileLock(f"{METRICS_FILE}.lock", timeout=10)
```

Lalu ganti tiap `with _metrics_lock:` (fungsi `log_query`, `log_document_indexed`, `sync_indexed_docs_count`, `reset_metrics`) menjadi `with _metrics_lock, _metrics_file_lock:`. Contoh untuk `log_query` (baris 93):

```python
    with _metrics_lock, _metrics_file_lock:
        metrics = load_metrics()
        ...
```

Terapkan pola yang sama persis di `log_document_indexed` (baris 129), `sync_indexed_docs_count` (baris 137), dan `reset_metrics` (baris 201).

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest backend/tests/test_metrics.py -v`
Expected: PASS (semua test termasuk yang baru)

- [ ] **Step 6: Jalankan test lain yang menyentuh metrics untuk regresi**

Run: `pytest backend/tests/test_metrics_integration.py backend/tests/test_pipeline_metrics.py backend/tests/test_metrics_api.py -v`
Expected: semua PASS. Kalau ada test yang gagal karena file `.lock` residual tertinggal di direktori kerja test, pastikan fixture yang mengatur `METRICS_FILE` sementara (`tmp_path`) juga membersihkan file `.lock` di teardown - cek pola yang sudah dipakai test lain di file yang sama.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock backend/app/metrics.py backend/tests/test_metrics.py
git commit -m "fix(backend): guard metrics.json read-modify-write with cross-process file lock"
```

---

### Task 5: Validasi magic-byte PDF di upload

**Files:**
- Modify: `backend/app/index_api.py:57-92` (`_save_uploaded_pdf`)
- Test: `backend/tests/test_index_api.py`

**Interfaces:** Tidak ada perubahan signature - `_save_uploaded_pdf(file: UploadFile) -> Tuple[str, str]` tetap sama, hanya menolak lebih banyak input tidak valid (HTTP 400).

- [ ] **Step 1: Baca pola test upload yang sudah ada**

Baca `backend/tests/test_index_api.py` untuk melihat bagaimana `UploadFile`/upload endpoint di-test di file ini sebelum menambah test baru (pastikan gaya konsisten - mis. `io.BytesIO` + `UploadFile(file=..., filename=...)` atau lewat `httpx.AsyncClient` `files=...`).

- [ ] **Step 2: Write the failing test**

Tambahkan di `backend/tests/test_index_api.py`:

```python
import io

from fastapi import HTTPException, UploadFile
import pytest

from app.index_api import _save_uploaded_pdf


def test_save_uploaded_pdf_rejects_non_pdf_content(tmp_path, monkeypatch):
    """A file renamed to .pdf but whose content isn't a real PDF must be
    rejected before it reaches the parsing layer."""
    monkeypatch.setattr("app.index_api.UPLOAD_DIR", str(tmp_path))

    fake_pdf = UploadFile(
        filename="fake.pdf",
        file=io.BytesIO(b"this is not a pdf, just renamed"),
    )

    with pytest.raises(HTTPException) as exc_info:
        _save_uploaded_pdf(fake_pdf)

    assert exc_info.value.status_code == 400
    assert list(tmp_path.iterdir()) == []  # partial file cleaned up


def test_save_uploaded_pdf_accepts_real_pdf_header(tmp_path, monkeypatch):
    monkeypatch.setattr("app.index_api.UPLOAD_DIR", str(tmp_path))

    real_pdf = UploadFile(
        filename="real.pdf",
        file=io.BytesIO(b"%PDF-1.4\n%rest of a minimal pdf body..."),
    )

    file_path, content_hash = _save_uploaded_pdf(real_pdf)

    assert file_path.endswith("real.pdf")
    assert len(content_hash) == 64
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest backend/tests/test_index_api.py -k save_uploaded_pdf -v`
Expected: `test_save_uploaded_pdf_rejects_non_pdf_content` FAILS (no 400 raised today), `test_save_uploaded_pdf_accepts_real_pdf_header` PASSES already.

- [ ] **Step 4: Update `_save_uploaded_pdf` di `index_api.py`**

Ganti body (baris 74-92) dari:

```python
    total_bytes = 0
    hasher = hashlib.sha256()
    try:
        with open(file_path, "wb") as f:
            while chunk := file.file.read(_UPLOAD_READ_CHUNK):
                total_bytes += len(chunk)
                if total_bytes > _MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {MAX_UPLOAD_SIZE_MB}MB upload limit",
                    )
                hasher.update(chunk)
                f.write(chunk)
    except HTTPException:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise

    return file_path, hasher.hexdigest()
```

menjadi:

```python
    total_bytes = 0
    hasher = hashlib.sha256()
    is_first_chunk = True
    try:
        with open(file_path, "wb") as f:
            while chunk := file.file.read(_UPLOAD_READ_CHUNK):
                if is_first_chunk:
                    # Reject files renamed to .pdf whose content isn't
                    # actually a PDF, before handing them to the parser.
                    if not chunk.startswith(b"%PDF-"):
                        raise HTTPException(
                            status_code=400,
                            detail="File content is not a valid PDF (missing %PDF- header)",
                        )
                    is_first_chunk = False

                total_bytes += len(chunk)
                if total_bytes > _MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {MAX_UPLOAD_SIZE_MB}MB upload limit",
                    )
                hasher.update(chunk)
                f.write(chunk)
    except HTTPException:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise

    return file_path, hasher.hexdigest()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest backend/tests/test_index_api.py -k save_uploaded_pdf -v`
Expected: PASS (kedua test)

- [ ] **Step 6: Jalankan seluruh `test_index_api.py` untuk regresi**

Run: `pytest backend/tests/test_index_api.py -v`
Expected: semua PASS - pastikan test upload lain yang memakai konten dummy (bukan `%PDF-`) tidak ikut ke-reject; kalau ada, update fixture-nya untuk memakai konten berawalan `b"%PDF-"`.

- [ ] **Step 7: Commit**

```bash
git add backend/app/index_api.py backend/tests/test_index_api.py
git commit -m "fix(backend): validate PDF magic bytes on upload, not just filename extension"
```

---

### Task 6: Startup warning kalau `GROQ_API_KEY`/`INVENIOAI_API_KEY` kosong

**Files:**
- Modify: `backend/app/main.py:26,60-72`
- Test: `backend/tests/test_main.py`

**Interfaces:** Tidak ada perubahan signature publik - murni tambahan `logger.warning(...)` di awal `lifespan()`.

- [ ] **Step 1: Write the failing test**

Tambahkan di `backend/tests/test_main.py`:

```python
import logging

import pytest


@pytest.mark.asyncio
async def test_lifespan_warns_when_groq_api_key_missing(caplog):
    from app.main import lifespan, app

    with patch("app.main.GROQ_API_KEY", None), patch("app.main.API_KEY", None):
        with caplog.at_level(logging.WARNING):
            async with lifespan(app):
                pass

    assert any("GROQ_API_KEY" in record.message for record in caplog.records)
    assert any("INVENIOAI_API_KEY" in record.message or "without authentication" in record.message.lower()
               for record in caplog.records)
```

(Tambahkan `from unittest.mock import patch` ke import block paling atas file jika belum ada - cek dulu, `test_query_stream_endpoint` di file yang sama sudah mengimpornya.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_main.py::test_lifespan_warns_when_groq_api_key_missing -v`
Expected: FAIL - `assert any(...)` gagal karena belum ada log warning apapun soal `GROQ_API_KEY`.

- [ ] **Step 3: Update `main.py`**

Ganti import config (baris 26) dari:

```python
from .config import ALLOWED_ORIGINS, PRELOAD_EMBEDDINGS_ON_STARTUP
```

menjadi:

```python
from .config import ALLOWED_ORIGINS, API_KEY, GROQ_API_KEY, PRELOAD_EMBEDDINGS_ON_STARTUP
```

Di awal `lifespan()` (baris 60-63), ganti:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info(f"Lifespan starting. Preload setting: {PRELOAD_EMBEDDINGS_ON_STARTUP}")
```

menjadi:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    if not GROQ_API_KEY:
        logger.warning(
            "GROQ_API_KEY is not set - /query and /query/stream will fail with a "
            "500 error until it is configured in .env."
        )
    if not API_KEY:
        logger.warning(
            "INVENIOAI_API_KEY is not set - all endpoints are running without "
            "authentication. Set it in .env before exposing this server publicly."
        )

    logger.info(f"Lifespan starting. Preload setting: {PRELOAD_EMBEDDINGS_ON_STARTUP}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_main.py -v`
Expected: PASS (semua test di file ini)

- [ ] **Step 5: Jalankan full suite untuk regresi**

Run: `pytest backend/tests -v`
Expected: semua PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/app/main.py backend/tests/test_main.py
git commit -m "feat(backend): warn at startup when GROQ_API_KEY or INVENIOAI_API_KEY is unset"
```

---

### Task 7: Tutup gap test coverage - `qdrant_conn.py` dan endpoint `/query`

**Files:**
- Test: `backend/tests/test_qdrant_conn.py` (baru)
- Modify: `backend/tests/test_main.py` (tambah test untuk `POST /query`)

**Interfaces:** Tidak ada perubahan kode produksi di task ini - murni menambah test untuk kode yang sudah ada (`qdrant_conn.get_qdrant_client`/`recreate_qdrant_client`/`close_qdrant_client`, dan `main.query()`).

- [ ] **Step 1: Write `test_qdrant_conn.py`**

Create `backend/tests/test_qdrant_conn.py`:

```python
from unittest.mock import MagicMock, patch

import pytest

from app import qdrant_conn


@pytest.fixture(autouse=True)
def _reset_qdrant_singleton():
    qdrant_conn._client = None
    yield
    qdrant_conn._client = None


def test_get_qdrant_client_is_a_singleton():
    with patch.object(qdrant_conn, "_create_qdrant_client") as mock_create:
        mock_create.return_value = MagicMock()

        first = qdrant_conn.get_qdrant_client()
        second = qdrant_conn.get_qdrant_client()

        assert first is second
        mock_create.assert_called_once()


def test_recreate_qdrant_client_closes_old_and_creates_new():
    old_client = MagicMock()
    new_client = MagicMock()

    with patch.object(qdrant_conn, "_create_qdrant_client", side_effect=[old_client, new_client]):
        first = qdrant_conn.get_qdrant_client()
        assert first is old_client

        recreated = qdrant_conn.recreate_qdrant_client()

        old_client.close.assert_called_once()
        assert recreated is new_client
        assert qdrant_conn.get_qdrant_client() is new_client


def test_close_qdrant_client_resets_singleton():
    client = MagicMock()
    with patch.object(qdrant_conn, "_create_qdrant_client", return_value=client):
        qdrant_conn.get_qdrant_client()
        qdrant_conn.close_qdrant_client()

        client.close.assert_called_once()
        assert qdrant_conn._client is None


def test_is_qdrant_client_closed_error_detects_known_messages():
    assert qdrant_conn.is_qdrant_client_closed_error(Exception("Client has been closed"))
    assert qdrant_conn.is_qdrant_client_closed_error(Exception("client IS CLOSED"))
    assert not qdrant_conn.is_qdrant_client_closed_error(Exception("connection refused"))
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest backend/tests/test_qdrant_conn.py -v`
Expected: PASS (kode `qdrant_conn.py` sudah ada dan benar - test ini murni menutup gap coverage, bukan TDD untuk kode baru).

- [ ] **Step 3: Write test untuk `POST /query` di `test_main.py`**

Tambahkan di `backend/tests/test_main.py`:

```python
@pytest.mark.asyncio
@patch("app.main.rag_pipeline")
async def test_query_endpoint_success(mock_rag_pipeline):
    mock_rag_pipeline.return_value = {
        "answer": "This is the answer.",
        "sources": "doc.pdf",
        "metrics": {"docs_retrieved": 3},
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/query", json={"question": "test query", "history": []})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "This is the answer."
    assert data["sources"] == "doc.pdf"
    assert data["metrics"] == {"docs_retrieved": 3}


@pytest.mark.asyncio
@patch("app.main.rag_pipeline")
async def test_query_endpoint_value_error_returns_400(mock_rag_pipeline):
    mock_rag_pipeline.side_effect = ValueError("GROQ_API_KEY belum di-set.")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/query", json={"question": "test query", "history": []})

    assert response.status_code == 400
    assert "GROQ_API_KEY" in response.json()["detail"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_main.py -v`
Expected: PASS (semua test di file, termasuk 2 test baru).

- [ ] **Step 5: Jalankan full suite**

Run: `pytest backend/tests -v`
Expected: semua PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/tests/test_qdrant_conn.py backend/tests/test_main.py
git commit -m "test(backend): add coverage for qdrant_conn singleton and POST /query endpoint"
```

---

## Verifikasi akhir (setelah semua task)

1. `pytest backend/tests -v` - seluruh suite (termasuk semua test baru) harus PASS.
2. `uv sync` - pastikan `pyproject.toml`/`uv.lock` konsisten setelah penambahan `filelock`.
3. Jalankan backend lokal (`cd backend && uvicorn app.main:app --reload`) tanpa `GROQ_API_KEY`/`INVENIOAI_API_KEY` di `.env`, konfirmasi dua baris `WARNING` muncul di log startup (Task 6).
4. Grep manual: `grep -rn "threshold=0.995" backend/app/rag_pipeline.py` harus kosong (Task 3 sudah menghapus literal duplikat).
