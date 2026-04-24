"""FastAPI application entrypoint.

Wires the API router and exposes two ways to query the RAG pipeline:

- `POST /query` for a simple request/response flow.
- `POST /query/jobs` for background execution with polling.

Job state is stored in-memory, so it resets on process restart.
"""

import logging
import threading
import time
import uuid
from typing import Any, Dict, List, Literal, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field
from .embeddings import get_embeddings
from .index_api import router as index_router
from .config import PRELOAD_EMBEDDINGS_ON_STARTUP
from .rag_pipeline import rag_pipeline
from .qdrant_conn import close_qdrant_client


logger = logging.getLogger(__name__)


app = FastAPI(title="InvenioAI API")
app.include_router(index_router)


@app.on_event("startup")
def _startup() -> None:
    # Optional warm-up; disabled by default for constrained deployments.
    if not PRELOAD_EMBEDDINGS_ON_STARTUP:
        logger.info("Embedding preload skipped (INVENIOAI_PRELOAD_EMBEDDINGS=0)")
        return

    try:
        get_embeddings()
        logger.info("Embedding model preloaded")
    except Exception:
        logger.warning("Embedding preload failed; falling back to lazy init", exc_info=True)


@app.on_event("shutdown")
def _shutdown() -> None:
    close_qdrant_client()


class Query(BaseModel):
    question: str
    history: List[str] = Field(default_factory=list)


JobState = Literal["pending", "running", "succeeded", "failed"]


class QueryJob(BaseModel):
    job_id: str
    status: JobState
    created_at: float
    updated_at: float
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


_jobs_lock = threading.Lock()
_jobs: Dict[str, QueryJob] = {}


def _set_job(job: QueryJob) -> None:
    with _jobs_lock:
        _jobs[job.job_id] = job


def _get_job(job_id: str) -> Optional[QueryJob]:
    with _jobs_lock:
        return _jobs.get(job_id)


def _run_query_job(job_id: str, q: Query) -> None:
    job = _get_job(job_id)
    if not job:
        return

    now = time.time()
    job.status = "running"
    job.updated_at = now
    _set_job(job)

    try:
        result = rag_pipeline(q.question, q.history)
        payload = {
            "answer": result["answer"],
            "sources": result.get("sources", ""),
            "metrics": result.get("metrics", {}),
        }
        now = time.time()
        job.status = "succeeded"
        job.result = payload
        job.updated_at = now
        _set_job(job)
    except Exception as exc:
        logger.exception("Background query job failed (job_id=%s)", job_id)
        now = time.time()
        job.status = "failed"
        job.error = f"{type(exc).__name__}: {exc}"
        job.updated_at = now
        _set_job(job)


@app.post("/query")
def query(q: Query) -> Dict[str, Any]:
    try:
        result = rag_pipeline(q.question, q.history)
        return {
            "answer": result["answer"],
            "sources": result.get("sources", ""),
            "metrics": result.get("metrics", {}),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("/query failed")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.post("/query/jobs")
def create_query_job(q: Query, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    now = time.time()
    job = QueryJob(
        job_id=job_id,
        status="pending",
        created_at=now,
        updated_at=now,
        result=None,
        error=None,
    )
    _set_job(job)
    background_tasks.add_task(_run_query_job, job_id, q)
    return job


@app.get("/query/jobs/{job_id}")
def get_query_job(job_id: str):
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/")
def root():
    return {"status": "InvenioAI API running"}