"""Tests for the process-wide retriever cache in `app.retriever.build_retriever`."""
from contextlib import ExitStack

import pytest
from unittest.mock import MagicMock, patch

from app.retriever import build_retriever, invalidate_retriever_cache


@pytest.fixture(autouse=True)
def _reset_retriever_cache():
    """Isolate the module-level cache between tests (and from other test files)."""
    invalidate_retriever_cache()
    yield
    invalidate_retriever_cache()


@pytest.fixture
def mocked_retriever_deps():
    """Patch everything `build_retriever()` touches besides the Qdrant client,
    so tests can exercise the real caching logic without hitting real
    Qdrant/Groq/embedding models. Yields the mocked QdrantVectorStore class
    for call-count assertions."""
    with ExitStack() as stack:
        stack.enter_context(patch("app.retriever.GROQ_API_KEY", "fake-key"))
        stack.enter_context(patch("app.retriever.get_embeddings", return_value=MagicMock()))
        stack.enter_context(patch("app.retriever.get_sparse_embeddings", return_value=MagicMock()))
        mock_vectorstore_class = stack.enter_context(patch("app.retriever.QdrantVectorStore"))
        mock_vectorstore_class.return_value.as_retriever.return_value = MagicMock()
        yield mock_vectorstore_class


def _make_qdrant_client_with_collection():
    client = MagicMock()
    collection = MagicMock()
    collection.name = "invenioai_collection"
    client.get_collections.return_value = MagicMock(collections=[collection])
    return client


def test_build_retriever_reuses_cached_stack_for_same_client(mocked_retriever_deps):
    """Calling build_retriever() twice with the same underlying Qdrant client
    must not rebuild the vectorstore/retriever or re-check collection
    existence the second time - that's the whole point of the cache."""
    mock_vectorstore_class = mocked_retriever_deps
    client = _make_qdrant_client_with_collection()

    with patch("app.retriever.get_qdrant_client", return_value=client):
        first = build_retriever()
        second = build_retriever()

    assert first is second
    assert mock_vectorstore_class.call_count == 1
    assert mock_vectorstore_class.return_value.as_retriever.call_count == 1
    client.get_collections.assert_called_once()


def test_invalidate_retriever_cache_forces_rebuild(mocked_retriever_deps):
    """After invalidate_retriever_cache(), the next build_retriever() call
    must rebuild the stack (e.g. because the document set changed)."""
    mock_vectorstore_class = mocked_retriever_deps
    client = _make_qdrant_client_with_collection()

    with patch("app.retriever.get_qdrant_client", return_value=client):
        build_retriever()
        invalidate_retriever_cache()
        build_retriever()

    assert mock_vectorstore_class.call_count == 2
    assert mock_vectorstore_class.return_value.as_retriever.call_count == 2


def test_build_retriever_rebuilds_when_qdrant_client_instance_changes(mocked_retriever_deps):
    """If the underlying Qdrant client singleton gets swapped (e.g. after a
    'client closed' recovery in qdrant_conn.py), the cache must not keep
    serving a retriever bound to the old, now-defunct client."""
    mock_vectorstore_class = mocked_retriever_deps
    client_a = _make_qdrant_client_with_collection()
    client_b = _make_qdrant_client_with_collection()

    with patch("app.retriever.get_qdrant_client", side_effect=[client_a, client_b]):
        _, _, first_client = build_retriever()
        _, _, second_client = build_retriever()

    assert first_client is client_a
    assert second_client is client_b
    assert mock_vectorstore_class.call_count == 2


def test_build_retriever_raises_without_groq_api_key():
    with patch("app.retriever.GROQ_API_KEY", None):
        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            build_retriever()


def test_build_retriever_raises_when_collection_missing():
    client = MagicMock()
    client.get_collections.return_value = MagicMock(collections=[])

    with patch("app.retriever.get_qdrant_client", return_value=client), \
         patch("app.retriever.GROQ_API_KEY", "fake-key"), \
         patch("app.retriever.get_embeddings", return_value=MagicMock()), \
         patch("app.retriever.get_sparse_embeddings", return_value=MagicMock()):
        with pytest.raises(ValueError, match="Belum ada dokumen"):
            build_retriever()
