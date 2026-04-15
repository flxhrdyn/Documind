"""PDF indexing pipeline.

Given a local PDF file path, it loads the pages, splits them into overlapping
chunks, embeds the chunks, and upserts them into the configured Qdrant
collection.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Distance, VectorParams

from .embeddings import get_embeddings
from .config import CHUNK_OVERLAP, CHUNK_SIZE, INDEXING_BATCH_SIZE, QDRANT_COLLECTION
from .metrics import log_document_indexed
from .qdrant_conn import get_qdrant_client, is_qdrant_client_closed_error, recreate_qdrant_client


logger = logging.getLogger(__name__)


def _collection_exists(client, collection_name: str) -> bool:
    if hasattr(client, "collection_exists"):
        return bool(client.collection_exists(collection_name=collection_name))

    collections = client.get_collections().collections
    return collection_name in [collection.name for collection in collections]


def index_documents(file_path: str) -> None:
    """Index a single PDF into Qdrant.

    Creates the Qdrant collection on-demand if it does not exist.
    """

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    total_start = time.monotonic()
    logger.info("Indexing PDF: %s", path.name)

    # Load PDF
    load_start = time.monotonic()
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    logger.info("Loaded %d pages (%.2fs)", len(documents), time.monotonic() - load_start)

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    split_start = time.monotonic()
    chunks = splitter.split_documents(documents)

    logger.info("Created %d chunks (%.2fs)", len(chunks), time.monotonic() - split_start)

    # Embedding model (cached)
    embeddings = get_embeddings()

    max_attempts = 2
    for attempt in range(max_attempts):
        client = get_qdrant_client() if attempt == 0 else recreate_qdrant_client()

        try:
            if not _collection_exists(client, QDRANT_COLLECTION):
                logger.info("Creating Qdrant collection: %s", QDRANT_COLLECTION)

                # Determine embedding vector size once from the configured model.
                sample_vector = embeddings.embed_query("sample")
                vector_size = len(sample_vector)

                client.create_collection(
                    collection_name=QDRANT_COLLECTION,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )

                logger.info("Qdrant collection created")
            else:
                logger.info("Using existing Qdrant collection: %s", QDRANT_COLLECTION)

            vectorstore = QdrantVectorStore(
                client=client,
                collection_name=QDRANT_COLLECTION,
                embedding=embeddings,
            )
            upsert_start = time.monotonic()
            vectorstore.add_documents(chunks, batch_size=INDEXING_BATCH_SIZE)
            logger.info(
                "Indexed %d chunks into Qdrant (batch=%d, %.2fs)",
                len(chunks),
                INDEXING_BATCH_SIZE,
                time.monotonic() - upsert_start,
            )
            break
        except Exception as exc:
            if attempt < (max_attempts - 1) and is_qdrant_client_closed_error(exc):
                logger.warning(
                    "Qdrant client was closed during indexing; recreating client and retrying once"
                )
                continue
            raise

    # Update local metrics for the dashboard.
    log_document_indexed()
    logger.info("Indexing complete (%.2fs total)", time.monotonic() - total_start)