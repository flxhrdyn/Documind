import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from app.main import app

client = TestClient(app)

@patch("app.index_api.get_qdrant_client")
def test_delete_document_success(mock_get_qdrant):
    # Setup mock
    mock_qdrant = MagicMock()
    mock_get_qdrant.return_value = mock_qdrant
    
    # Mock collection existence to proceed to delete
    from app.index_api import QDRANT_COLLECTION
    mock_coll = MagicMock()
    mock_coll.name = QDRANT_COLLECTION
    mock_qdrant.get_collections.return_value = MagicMock(collections=[mock_coll])
    
    # Mocking scroll to simulate the document exists (used in list_documents)
    # We'll just assume it exists for the DELETE call.
    # The actual implementation of DELETE /documents/{filename} will be added.
    
    filename = "test.pdf"
    
    # Execute - should use query param as per implementation
    response = client.delete("/documents/delete", params={"filename": filename})
    
    # Assert
    assert response.status_code == 200
    assert response.json() == {"status": "success", "message": f"Document '{filename}' deleted successfully"}
    
    # Verify Qdrant was called with correct filter
    # Expecting delete(collection_name=..., points_selector=...)
    mock_qdrant.delete.assert_called_once()
    args, kwargs = mock_qdrant.delete.call_args
    assert kwargs["collection_name"] == "invenioai_collection"
    
    # Verify filter
    points_selector = kwargs["points_selector"]
    from qdrant_client.http import models
    assert isinstance(points_selector, models.FilterSelector)
    
    # We expect a filter that matches metadata.source_file or metadata.source
    filter_obj = points_selector.filter
    assert filter_obj.should[0].key == "metadata.source_file"
    assert filter_obj.should[0].match.value == filename
    assert filter_obj.should[1].key == "metadata.source"


@patch("app.metrics.sync_indexed_docs_count")
@patch("app.retriever.invalidate_retriever_cache")
@patch("app.cache_manager.CacheManager")
@patch("app.index_api.get_qdrant_client")
def test_delete_document_clears_cache_and_invalidates_retriever(
    mock_get_qdrant, mock_cache_manager_class, mock_invalidate_retriever, mock_sync_count
):
    """Deleting a single document must not serve stale cached answers - it
    should clear the RAG cache and invalidate the cached retriever stack,
    same as clearing the whole knowledge base does."""
    mock_qdrant = MagicMock()
    mock_get_qdrant.return_value = mock_qdrant

    from app.index_api import QDRANT_COLLECTION
    mock_coll = MagicMock()
    mock_coll.name = QDRANT_COLLECTION
    mock_qdrant.get_collections.return_value = MagicMock(collections=[mock_coll])
    mock_qdrant.scroll.return_value = ([], None)

    mock_cache_instance = MagicMock()
    mock_cache_manager_class.return_value = mock_cache_instance

    response = client.delete("/documents/delete", params={"filename": "test.pdf"})

    assert response.status_code == 200
    mock_cache_instance.clear.assert_called_once()
    mock_invalidate_retriever.assert_called_once()
    mock_sync_count.assert_called_once()


class TestGetIndexedDocumentNames:
    """Tests for `get_indexed_document_names` (shared Qdrant scroll helper)."""

    def test_returns_empty_list_when_collection_missing(self):
        from app.index_api import get_indexed_document_names

        client_mock = MagicMock()
        client_mock.get_collections.return_value = MagicMock(collections=[])

        assert get_indexed_document_names(client_mock) == []
        client_mock.scroll.assert_not_called()

    def test_scrolls_all_pages_and_dedupes_by_basename(self):
        from app.index_api import get_indexed_document_names, QDRANT_COLLECTION

        client_mock = MagicMock()
        coll = MagicMock()
        coll.name = QDRANT_COLLECTION
        client_mock.get_collections.return_value = MagicMock(collections=[coll])

        page1_point = MagicMock()
        page1_point.payload = {"metadata": {"source_file": "uploaded_docs/report.pdf"}}
        page2_point = MagicMock()
        page2_point.payload = {"metadata": {"source_file": "report.pdf"}}

        client_mock.scroll.side_effect = [
            ([page1_point], "next-offset"),
            ([page2_point], None),
        ]

        docs = get_indexed_document_names(client_mock)
        assert docs == ["report.pdf"]
        assert client_mock.scroll.call_count == 2


class TestFindDuplicateDocument:
    """Tests for `_find_duplicate_document` (content-hash dedup check)."""

    def test_returns_none_when_no_collection_yet(self):
        from app.index_api import _find_duplicate_document, QDRANT_COLLECTION

        client_mock = MagicMock()
        client_mock.get_collections.return_value = MagicMock(collections=[])

        assert _find_duplicate_document(client_mock, "somehash") is None
        client_mock.scroll.assert_not_called()

    def test_returns_none_when_no_matching_hash(self):
        from app.index_api import _find_duplicate_document, QDRANT_COLLECTION

        client_mock = MagicMock()
        coll = MagicMock()
        coll.name = QDRANT_COLLECTION
        client_mock.get_collections.return_value = MagicMock(collections=[coll])
        client_mock.scroll.return_value = ([], None)

        assert _find_duplicate_document(client_mock, "somehash") is None

    def test_returns_existing_filename_when_hash_matches(self):
        from app.index_api import _find_duplicate_document, QDRANT_COLLECTION

        client_mock = MagicMock()
        coll = MagicMock()
        coll.name = QDRANT_COLLECTION
        client_mock.get_collections.return_value = MagicMock(collections=[coll])

        matching_point = MagicMock()
        matching_point.payload = {"metadata": {"source_file": "already_indexed.pdf"}}
        client_mock.scroll.return_value = ([matching_point], None)

        result = _find_duplicate_document(client_mock, "somehash")
        assert result == "already_indexed.pdf"


class TestIndexUploadedPdfDedup:
    """Tests for `_index_uploaded_pdf` rejecting duplicate content uploads."""

    @patch("app.index_api._find_duplicate_document")
    @patch("app.index_api.get_qdrant_client")
    @patch("os.path.exists", return_value=True)
    @patch("os.remove")
    def test_rejects_upload_with_duplicate_content_hash(
        self, mock_remove, mock_exists, mock_get_qdrant, mock_find_duplicate
    ):
        from app.index_api import _index_uploaded_pdf
        from fastapi import HTTPException

        mock_find_duplicate.return_value = "existing_report.pdf"

        with pytest.raises(HTTPException) as exc_info:
            _index_uploaded_pdf("/fake/uploaded_docs/new_report.pdf", "duplicate-hash-123")

        assert exc_info.value.status_code == 409
        assert "existing_report.pdf" in exc_info.value.detail
        # The freshly-saved duplicate file should be cleaned up, not left on disk.
        mock_remove.assert_called_once_with("/fake/uploaded_docs/new_report.pdf")
