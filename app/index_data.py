"""PDF indexing pipeline.

Given a local PDF file path, it loads the pages, splits them into overlapping
chunks, embeds the chunks, and upserts them into the configured Qdrant
collection.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Distance, VectorParams

from .embeddings import get_embeddings
from .config import CHUNK_OVERLAP, CHUNK_SIZE, QDRANT_COLLECTION
from .metrics import log_document_indexed
from .qdrant_conn import get_qdrant_client


logger = logging.getLogger(__name__)


def index_documents(file_path: str) -> None:
    """Index a single PDF into Qdrant.

    Creates the Qdrant collection on-demand if it does not exist.
    """

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info("Indexing PDF: %s", path.name)

    # Load PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    logger.info("Loaded %d pages", len(documents))

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(documents)

    logger.info("Created %d chunks", len(chunks))

    # Embedding model (cached)
    embeddings = get_embeddings()

    client = get_qdrant_client()

    # Check if collection exists
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]

    if QDRANT_COLLECTION not in collection_names:
        logger.info("Creating Qdrant collection: %s", QDRANT_COLLECTION)

        # Determine embedding vector size
        sample_vector = embeddings.embed_query("sample")
        vector_size = len(sample_vector)

        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

        logger.info("Qdrant collection created")
    else:
        logger.info("Using existing Qdrant collection: %s", QDRANT_COLLECTION)

    # Add documents using QdrantVectorStore (works for both new and existing)
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embedding=embeddings,
    )
    vectorstore.add_documents(chunks)
    logger.info("Indexed %d chunks into Qdrant", len(chunks))

    # Update local metrics for the dashboard.
    log_document_indexed()
    logger.info("Indexing complete")