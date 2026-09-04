"""Indexing and document-management routes.

Endpoints here cover PDF upload/indexing, listing indexed sources, and clearing
the vector store.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import threading
import time
import uuid
from typing import Any, Dict, Literal, Optional, Callable, Tuple
from qdrant_client import QdrantClient, models

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from .config import (
    DELETE_UPLOADED_PDFS,
    MAX_UPLOAD_SIZE_MB,
    QDRANT_COLLECTION,
    QDRANT_PATH,
    QDRANT_URL,
    UPLOAD_DIR,
)
from .index_data import index_documents
from .qdrant_conn import close_qdrant_client, get_qdrant_client

router = APIRouter()

logger = logging.getLogger(__name__)

os.makedirs(UPLOAD_DIR, exist_ok=True)

UploadJobState = Literal["pending", "running", "parsing", "indexing", "succeeded", "failed"]
_upload_jobs_lock = threading.Lock()
_upload_jobs: Dict[str, Dict[str, Any]] = {}


def _set_upload_job(job: Dict[str, Any]) -> None:
    with _upload_jobs_lock:
        _upload_jobs[job["job_id"]] = job


def _get_upload_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _upload_jobs_lock:
        return _upload_jobs.get(job_id)


_MAX_UPLOAD_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
_UPLOAD_READ_CHUNK = 1024 * 1024


def _save_uploaded_pdf(file: UploadFile) -> Tuple[str, str]:
    """Save the upload to disk and return `(file_path, content_hash)`."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    safe_name = os.path.basename(file.filename)
    if not safe_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported")

    file_path = os.path.join(UPLOAD_DIR, safe_name)
    if os.path.exists(file_path):
        stem, ext = os.path.splitext(safe_name)
        file_path = os.path.join(UPLOAD_DIR, f"{stem}_{uuid.uuid4().hex[:8]}{ext}")

    # Stream to disk in chunks and enforce a hard size cap, instead of
    # buffering the whole upload into memory with a single file.read(). Hash
    # incrementally so duplicate content can be detected before indexing.
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


def _find_duplicate_document(client, content_hash: str) -> Optional[str]:
    """Return the filename of an already-indexed document with identical content, if any."""
    try:
        existing = [c.name for c in client.get_collections().collections]
        if QDRANT_COLLECTION not in existing:
            return None

        try:
            client.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name="metadata.content_hash",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass

        points, _ = client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.content_hash",
                        match=models.MatchValue(value=content_hash),
                    )
                ]
            ),
            limit=1,
            with_payload=["metadata.source_file", "metadata.source"],
            with_vectors=False,
        )
        if not points:
            return None
        meta = (points[0].payload or {}).get("metadata") or {}
        return meta.get("source_file") or meta.get("source")
    except Exception:
        logger.warning("Duplicate-content check failed; proceeding with indexing", exc_info=True)
        return None


def _index_uploaded_pdf(
    file_path: str,
    content_hash: str,
    status_callback: Optional[Callable[[str], None]] = None
) -> Dict[str, str]:
    duplicate_of = _find_duplicate_document(get_qdrant_client(), content_hash)
    if duplicate_of:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=409,
            detail=f"Document with identical content is already indexed as '{duplicate_of}'",
        )

    # Index into Qdrant.
    try:
        index_documents(file_path, content_hash=content_hash, status_callback=status_callback)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Indexing failed")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {type(exc).__name__}: {exc}")

    # Optional: delete local PDF after indexing (useful for deployments where
    # disk is ephemeral or not shared across instances).
    if DELETE_UPLOADED_PDFS:
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass
        except Exception:
            # Best-effort cleanup; indexing already succeeded.
            pass

    _on_knowledge_base_changed()
    return {"status": "PDF indexed successfully", "filename": os.path.basename(file_path)}


def _on_knowledge_base_changed(doc_count: Optional[int] = None) -> None:
    """Refresh derived state after the indexed knowledge base changes.

    - Invalidates the cached retriever stack (`retriever.py`) so the next
      query re-validates the collection instead of reusing a stale one.
    - Syncs the dashboard's indexed-document count. Reuses `list_documents()`'s
      unique-source scan instead of duplicating the Qdrant scroll logic (also
      done once at startup in `main.py`'s lifespan), unless the caller already
      knows the count (e.g. 0 right after the collection was deleted).
    """
    try:
        from .retriever import invalidate_retriever_cache
        invalidate_retriever_cache()
    except Exception:
        logger.warning("Failed to invalidate retriever cache", exc_info=True)

    try:
        from .metrics import sync_indexed_docs_count
        count = doc_count if doc_count is not None else list_documents()["count"]
        sync_indexed_docs_count(count)
    except Exception:
        logger.warning("Failed to sync indexed document count", exc_info=True)


def _run_upload_job(job_id: str, file_path: str, content_hash: str) -> None:
    job = _get_upload_job(job_id)
    if not job:
        return

    now = time.time()
    job["status"] = "running"
    job["updated_at"] = now
    _set_upload_job(job)

    def status_callback(new_status: str):
        job["status"] = new_status
        job["updated_at"] = time.time()
        _set_upload_job(job)

    try:
        result = _index_uploaded_pdf(file_path, content_hash, status_callback=status_callback)
        now = time.time()
        job["status"] = "succeeded"
        job["result"] = result
        job["updated_at"] = now
        _set_upload_job(job)
    except HTTPException as exc:
        now = time.time()
        job["status"] = "failed"
        job["error"] = str(exc.detail)
        job["updated_at"] = now
        _set_upload_job(job)
    except Exception as exc:
        logger.exception("Background upload job failed (job_id=%s)", job_id)
        now = time.time()
        job["status"] = "failed"
        job["error"] = f"{type(exc).__name__}: {exc}"
        job["updated_at"] = now
        _set_upload_job(job)


@router.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF and index it into Qdrant.

    The file is written to `UPLOAD_DIR` first, then passed to the indexing
    pipeline. If `DELETE_UPLOADED_PDFS=1`, the local file is removed after a
    successful index.
    """

    file_path, content_hash = _save_uploaded_pdf(file)
    return _index_uploaded_pdf(file_path, content_hash)


@router.post("/upload/jobs")
def create_upload_job(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload a PDF and index it in the background.

    Returns a job object immediately so clients can poll status via
    `/upload/jobs/{job_id}`.
    """

    file_path, content_hash = _save_uploaded_pdf(file)

    job_id = str(uuid.uuid4())
    now = time.time()
    job: Dict[str, Any] = {
        "job_id": job_id,
        "status": "pending",
        "filename": os.path.basename(file_path),
        "created_at": now,
        "updated_at": now,
        "result": None,
        "error": None,
    }
    _set_upload_job(job)
    background_tasks.add_task(_run_upload_job, job_id, file_path, content_hash)
    return job


@router.get("/upload/jobs/{job_id}")
def get_upload_job(job_id: str):
    """Return upload/indexing job state."""

    job = _get_upload_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def get_indexed_document_names(client: QdrantClient) -> list[str]:
    """Scroll the whole collection and collect unique indexed document names.

    No artificial point cap - scrolls until Qdrant returns offset=None.
    """
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        return []

    documents: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=500,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for p in points:
            payload = getattr(p, "payload", None) or {}
            meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else payload
            source = meta.get("source_file") or meta.get("file") or meta.get("source") or meta.get("filename")
            if source and isinstance(source, str):
                documents.add(os.path.basename(source))
        if offset is None:
            break
    return sorted(documents)


@router.get("/documents")
def list_documents():
    """List indexed document names.

    This derives the list from Qdrant payload metadata (document `source`) so
    the UI can work even when uploaded PDFs are deleted after indexing.
    """

    client = get_qdrant_client()
    try:
        documents = get_indexed_document_names(client)
    except Exception as exc:
        logger.exception("Failed to list Qdrant collections")
        raise HTTPException(status_code=500, detail=f"Failed to query Qdrant: {type(exc).__name__}: {exc}")

    return {"documents": documents, "count": len(documents)}



@router.delete("/documents/delete")
def delete_document(filename: str, retry: bool = True):
    """Delete a specific document by its filename from the vector store."""
    from .qdrant_conn import is_qdrant_client_closed_error
    safe_name = os.path.basename(filename)
    client = get_qdrant_client()
    try:
        # Check if collection exists first
        try:
            collections = client.get_collections().collections
            existing = [c.name for c in collections]
            if QDRANT_COLLECTION not in existing:
                # Collection doesn't exist, so document isn't there
                return {"status": "success", "message": f"Document '{safe_name}' not found in index (collection missing)"}
        except Exception as e:
            logger.warning("Failed to check collection existence: %s", e)

        # Ensure payload indices exist for both fields we filter on
        try:
            client.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name="metadata.source_file",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass

        try:
            client.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name="metadata.source",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass

        # Qdrant delete call using a filter on both metadata fields
        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    should=[
                        models.FieldCondition(
                            key="metadata.source_file",
                            match=models.MatchValue(value=safe_name),
                        ),
                        models.FieldCondition(
                            key="metadata.source",
                            match=models.MatchValue(value=safe_name),
                        ),
                    ]
                )
            ),
        )
        
        # Also try to delete from local UPLOAD_DIR if it exists
        file_path = os.path.join(UPLOAD_DIR, safe_name)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                logger.warning("Could not delete physical file %s: %s", file_path, e)
            
    except Exception as exc:
        if retry and is_qdrant_client_closed_error(exc):
            logger.warning("Qdrant client closed during delete; retrying once...")
            from .qdrant_conn import recreate_qdrant_client
            recreate_qdrant_client()
            return delete_document(filename, retry=False)
        
        logger.error(f"Failed to delete document '{safe_name}': {type(exc).__name__}: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Delete failed for '{safe_name}': {type(exc).__name__}: {exc}",
        )

    # Cached answers may reference the deleted document; wipe cache so stale
    # answers aren't served after the knowledge base changes.
    try:
        from .rag_pipeline import get_cache_manager
        get_cache_manager().clear()
        logger.info("Cache cleared after deleting document '%s'.", safe_name)
    except Exception as e:
        logger.warning(f"Could not clear cache after deleting document: {e}")

    _on_knowledge_base_changed()
    return {"status": "success", "message": f"Document '{safe_name}' deleted successfully"}


@router.delete("/documents")
def clear_documents():
    """Delete all indexed documents and clear the semantic cache.

    - In Qdrant server/cloud mode: deletes the collection.
    - In local mode: removes the on-disk storage directory.
    - Always clears the semantic and deep cache.
    """

    if os.path.exists(UPLOAD_DIR):
        for filename in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    try:
        if QDRANT_URL:
            # Server mode: delete collection without closing shared client.
            client = get_qdrant_client()
            existing = [c.name for c in client.get_collections().collections]
            if QDRANT_COLLECTION in existing:
                client.delete_collection(collection_name=QDRANT_COLLECTION)
        else:
            # Local mode: remove the storage directory.
            # Close open client first to avoid Windows file lock issues.
            close_qdrant_client()
            if os.path.exists(QDRANT_PATH):
                shutil.rmtree(QDRANT_PATH)
    except Exception as exc:
        logger.exception("Failed to clear documents")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear vector store: {type(exc).__name__}: {exc}",
        )

    # Clear semantic and deep cache to ensure fresh answers for new documents
    try:
        from .rag_pipeline import get_cache_manager
        get_cache_manager().clear()
        logger.info("Semantic and deep cache cleared.")
    except Exception as e:
        logger.warning(f"Could not clear semantic cache: {e}")

    _on_knowledge_base_changed(doc_count=0)

    return {"status": "Documents, vector store, and cache cleared"}