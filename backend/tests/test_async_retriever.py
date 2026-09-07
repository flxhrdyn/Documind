import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.documents import Document
from app.retriever import retrieve_documents, retrieve_documents_async

@pytest.mark.asyncio
async def test_retrieve_documents_async_returns_correct_types_and_metadata():
    # Setup Mocks
    mock_dense_retriever = AsyncMock()
    mock_dense_retriever.ainvoke.return_value = [Document(page_content="test doc")]

    mock_client = MagicMock()
    # Mocking QdrantClient's scroll method for BM25 fallback check
    mock_client.scroll.return_value = ([], None)

    # Execute
    query = "test query"
    docs, metadata = await retrieve_documents_async(query, mock_dense_retriever, mock_client)

    # Assertions
    assert isinstance(docs, list)
    assert len(docs) > 0
    assert isinstance(metadata, dict)

    assert "mode" in metadata
    assert "count" in metadata
    assert metadata["count"] == len(docs)


@pytest.mark.asyncio
async def test_retrieve_documents_async_merges_and_dedups_fusion_queries():
    """RAG Fusion: original query + fusion variations must be merged into one
    doc set, with duplicates (by page_content) collapsed and first-seen order
    kept - matches MultiQueryRetriever's default unique-union behavior."""
    mock_dense_retriever = AsyncMock()
    mock_dense_retriever.ainvoke.side_effect = [
        [Document(page_content="doc A"), Document(page_content="doc B")],
        [Document(page_content="doc B"), Document(page_content="doc C")],
        [Document(page_content="doc D")],
    ]
    mock_client = MagicMock()

    docs, metadata = await retrieve_documents_async(
        "original query",
        mock_dense_retriever,
        mock_client,
        fusion_queries=["fusion query 1", "fusion query 2"],
    )

    assert [d.page_content for d in docs] == ["doc A", "doc B", "doc C", "doc D"]
    assert metadata["count"] == 4
    assert mock_dense_retriever.ainvoke.call_count == 3
    mock_dense_retriever.ainvoke.assert_any_call("original query")
    mock_dense_retriever.ainvoke.assert_any_call("fusion query 1")
    mock_dense_retriever.ainvoke.assert_any_call("fusion query 2")


def test_retrieve_documents_sync_merges_and_dedups_fusion_queries():
    """Sync counterpart of the fusion merge/dedup test above."""
    mock_dense_retriever = MagicMock()
    mock_dense_retriever.invoke.side_effect = [
        [Document(page_content="doc A")],
        [Document(page_content="doc A"), Document(page_content="doc B")],
    ]
    mock_client = MagicMock()

    docs, metadata = retrieve_documents(
        "original query",
        dense_retriever=mock_dense_retriever,
        client=mock_client,
        fusion_queries=["fusion query 1"],
    )

    assert [d.page_content for d in docs] == ["doc A", "doc B"]
    assert metadata["count"] == 2
    assert mock_dense_retriever.invoke.call_count == 2


def test_retrieve_documents_sync_no_fusion_queries():
    mock_dense_retriever = MagicMock()
    mock_dense_retriever.invoke.return_value = [Document(page_content="only doc")]
    mock_client = MagicMock()

    docs, metadata = retrieve_documents(
        "solo query", dense_retriever=mock_dense_retriever, client=mock_client,
    )

    assert [d.page_content for d in docs] == ["only doc"]
    mock_dense_retriever.invoke.assert_called_once_with("solo query")
