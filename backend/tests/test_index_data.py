import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document as LangChainDocument
from app.index_data import (
    process_pdf_documents,
    index_documents,
    _check_or_record_embedding_model,
)

@patch("app.index_data.Path")
@patch("llama_parse.LlamaParse")
@patch("app.index_data.MarkdownHeaderTextSplitter")
@patch("app.index_data.RecursiveCharacterTextSplitter")
def test_process_pdf_documents_extracts_text_and_metadata(
    mock_recursive_splitter_class,
    mock_markdown_splitter_class,
    mock_llamaparse_class,
    mock_path_class
):
    # Setup Mocks
    mock_path_instance = mock_path_class.return_value
    mock_path_instance.exists.return_value = True
    mock_path_instance.name = "dummy.pdf"
    
    mock_parser = mock_llamaparse_class.return_value
    mock_llama_doc = MagicMock()
    mock_llama_doc.text = "test PDF"
    mock_llama_doc.metadata = {"page_number": "1"}
    mock_parser.load_data.return_value = [mock_llama_doc]
    
    mock_markdown_splitter = mock_markdown_splitter_class.return_value
    mock_markdown_splitter.split_text.return_value = [
        LangChainDocument(page_content="test PDF", metadata={})
    ]
    
    mock_recursive_splitter = mock_recursive_splitter_class.return_value
    mock_recursive_splitter.split_documents.return_value = [
        LangChainDocument(page_content="test PDF", metadata={
            "source_file": "dummy.pdf",
            "page_label": "1"
        })
    ]
    
    # Execute
    pdf_path = "dummy.pdf"
    docs = process_pdf_documents(pdf_path)
    
    # Assertions
    assert len(docs) > 0
    first_doc = docs[0]
    assert "test PDF" in first_doc.page_content
    assert first_doc.metadata["source_file"] == "dummy.pdf"
    assert first_doc.metadata["page_label"] == "1"


@patch("app.index_data.Path")
@patch("llama_parse.LlamaParse")
@patch("app.index_data.MarkdownHeaderTextSplitter")
@patch("app.index_data.RecursiveCharacterTextSplitter")
def test_recursive_splitting_preserves_structure(
    mock_recursive_splitter_class,
    mock_markdown_splitter_class,
    mock_llamaparse_class,
    mock_path_class
):
    # Setup Mocks
    mock_path_instance = mock_path_class.return_value
    mock_path_instance.exists.return_value = True
    mock_path_instance.name = "dummy.pdf"
    
    mock_parser = mock_llamaparse_class.return_value
    mock_llama_doc = MagicMock()
    mock_llama_doc.text = "sentence one. sentence two."
    mock_llama_doc.metadata = {"page_number": "1"}
    mock_parser.load_data.return_value = [mock_llama_doc]
    
    mock_markdown_splitter = mock_markdown_splitter_class.return_value
    mock_markdown_splitter.split_text.return_value = [
        LangChainDocument(page_content="sentence one. sentence two.", metadata={})
    ]
    
    mock_recursive_splitter = mock_recursive_splitter_class.return_value
    mock_recursive_splitter.split_documents.return_value = [
        LangChainDocument(page_content="sentence one.", metadata={"source_file": "dummy.pdf", "page_label": "1"}),
        LangChainDocument(page_content="sentence two.", metadata={"source_file": "dummy.pdf", "page_label": "1"})
    ]
    
    # Execute
    pdf_path = "dummy.pdf"
    docs = process_pdf_documents(pdf_path)
    
    # Assertions
    assert len(docs) == 2
    assert docs[0].metadata["page_label"] == "1"
    assert docs[1].metadata["page_label"] == "1"
    for doc in docs:
        assert doc.metadata["source_file"] == "dummy.pdf"


def test_strip_running_headers_footers_removes_repetitive_boundaries():
    """Verify that repetitive top/bottom boundary lines across pages are stripped."""
    from app.index_data import strip_running_headers_footers
    class MockDoc:
        def __init__(self, text):
            self.text = text

    # Buat 5 mock pages dengan running header dan footer yang sama di top/bottom,
    # namun dengan isi body yang berbeda
    pages = [
        MockDoc("Running Header Title\n\nBody content of page 1.\nImportant Info 1\n\nPage Footer 2023"),
        MockDoc("Running Header Title\n\nBody content of page 2.\nImportant Info 2\n\nPage Footer 2023"),
        MockDoc("Running Header Title\n\nBody content of page 3.\nImportant Info 3\n\nPage Footer 2023"),
        MockDoc("Running Header Title\n\nBody content of page 4.\nImportant Info 4\n\nPage Footer 2023"),
        MockDoc("Running Header Title\n\nBody content of page 5.\nImportant Info 5\n\nPage Footer 2023"),
    ]

    cleaned = strip_running_headers_footers(pages)
    
    # Assert running headers and footers are fully removed
    for page in cleaned:
        assert "Running Header Title" not in page.text
        assert "Page Footer 2023" not in page.text
        # Assert body content is untouched
        assert "Body content of page" in page.text
        assert "Important Info" in page.text


def test_strip_running_headers_footers_ignores_markdown_headers_and_unique_content():
    """Verify that actual structural markdown headings and unique body content are never removed."""
    from app.index_data import strip_running_headers_footers
    class MockDoc:
        def __init__(self, text):
            self.text = text

    # Halaman yang menduplikasi heading asli atau isi body yang kebetulan berulang
    pages = [
        MockDoc("# 7. Income taxes\n\nUnique text on page 1.\n\nFooter"),
        MockDoc("# 7. Income taxes\n\nUnique text on page 2.\n\nFooter"),
        MockDoc("# 7. Income taxes\n\nUnique text on page 3.\n\nFooter"),
    ]

    cleaned = strip_running_headers_footers(pages)
    
    # Assert structural headings starting with # are NOT removed
    for page in cleaned:
        assert "# 7. Income taxes" in page.text


def test_strip_running_headers_footers_robust_substring_match():
    """Verify that running headers and footers with minor variations (e.g. appended text/numbers) are stripped robustly."""
    from app.index_data import strip_running_headers_footers
    class MockDoc:
        def __init__(self, text):
            self.text = text

    # We create 5 pages where the header matches exactly on 4 pages,
    # but the 5th page has a minor suffix variation ("Financial Report 2023").
    # The running header should be detected from the first 4 pages (crossing the 15% threshold for 5 pages),
    # and the robust substring match should strip it on ALL pages, including the 5th page!
    pages = [
        MockDoc("Notes to the Syngenta AG Group\n\nPage 1 Content\n\nPage Footer"),
        MockDoc("Notes to the Syngenta AG Group\n\nPage 2 Content\n\nPage Footer"),
        MockDoc("Notes to the Syngenta AG Group\n\nPage 3 Content\n\nPage Footer"),
        MockDoc("Notes to the Syngenta AG Group\n\nPage 4 Content\n\nPage Footer"),
        MockDoc("Notes to the Syngenta AG Group Financial Report 2023\n\nPage 5 Content\n\nPage Footer 123"),
    ]

    cleaned = strip_running_headers_footers(pages)

    for page in cleaned:
        # Header should be fully removed on all pages
        assert "Notes to the Syngenta AG Group" not in page.text
        assert "Financial Report 2023" not in page.text
        
        # Footer (including dynamic numbers/suffixes) should be fully removed on all pages
        assert "Page Footer" not in page.text
        assert "123" not in page.text

        # Body should remain
        assert "Content" in page.text


@patch("app.index_data.Path")
@patch("llama_parse.LlamaParse")
@patch("app.index_data.MarkdownHeaderTextSplitter")
@patch("app.index_data.RecursiveCharacterTextSplitter")
def test_header_inheritance_propagates_headers_across_pages(
    mock_recursive_splitter_class,
    mock_markdown_splitter_class,
    mock_llamaparse_class,
    mock_path_class
):
    """Verify that headers are propagated to consecutive pages without headings."""
    # Setup Mocks
    mock_path_instance = mock_path_class.return_value
    mock_path_instance.exists.return_value = True
    mock_path_instance.name = "dummy.pdf"
    
    mock_parser = mock_llamaparse_class.return_value
    
    # Page 1 has Header 1
    # Page 2 has no new header
    # Page 3 has a new Header 1
    mock_llama_doc_1 = MagicMock()
    mock_llama_doc_1.text = "# 7. Income taxes\n\nPage 1 body"
    mock_llama_doc_1.metadata = {"page_number": "1"}
    
    mock_llama_doc_2 = MagicMock()
    mock_llama_doc_2.text = "Page 2 body (continuation)"
    mock_llama_doc_2.metadata = {"page_number": "2"}
    
    mock_llama_doc_3 = MagicMock()
    mock_llama_doc_3.text = "# 8. Financial instruments\n\nPage 3 body"
    mock_llama_doc_3.metadata = {"page_number": "3"}
    
    mock_parser.load_data.return_value = [mock_llama_doc_1, mock_llama_doc_2, mock_llama_doc_3]
    
    # Configure Markdown splitter mock
    mock_markdown_splitter = mock_markdown_splitter_class.return_value
    
    # Page 1 split: has Header 1
    doc_1_split = LangChainDocument(page_content="Page 1 body", metadata={"Header 1": "7. Income taxes"})
    # Page 2 split: has empty metadata (no headers on page 2)
    doc_2_split = LangChainDocument(page_content="Page 2 body (continuation)", metadata={})
    # Page 3 split: has new Header 1
    doc_3_split = LangChainDocument(page_content="Page 3 body", metadata={"Header 1": "8. Financial instruments"})
    
    # split_text is called per page
    mock_markdown_splitter.split_text.side_effect = [
        [doc_1_split],
        [doc_2_split],
        [doc_3_split]
    ]
    
    # Configure Recursive splitter mock to just return whatever it is given
    mock_recursive_splitter = mock_recursive_splitter_class.return_value
    def mock_split_documents(docs):
        return docs
    mock_recursive_splitter.split_documents.side_effect = mock_split_documents
    
    # Execute
    docs = process_pdf_documents("dummy.pdf")
    
    # Assertions
    assert len(docs) == 3
    # Page 1 split should have Header 1
    assert docs[0].metadata["Header 1"] == "7. Income taxes"
    assert docs[0].metadata["page_label"] == "1"
    
    # Page 2 split should have inherited Header 1 from page 1
    assert docs[1].metadata["Header 1"] == "7. Income taxes"
    assert docs[1].metadata["page_label"] == "2"
    
    # Page 3 split should have the new Header 1 and not inherit the old one
    assert docs[2].metadata["Header 1"] == "8. Financial instruments"
    assert docs[2].metadata["page_label"] == "3"


@patch("app.index_data.Path")
@patch("llama_parse.LlamaParse")
def test_table_header_carried_to_continuation_chunks(mock_llamaparse_class, mock_path_class):
    """A markdown table larger than CHUNK_SIZE must not leave later chunks
    with data rows but no column header - this exercises the real
    MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter (not mocked),
    matching what actually happens at runtime."""
    mock_path_instance = mock_path_class.return_value
    mock_path_instance.exists.return_value = True
    mock_path_instance.name = "table.pdf"

    rows = "".join(f"| RowLabelLong{i} | value_{i * 10} | value_{i * 20} |\n" for i in range(120))
    table_text = "# Financial Table\n| Metric | 2022 | 2023 |\n|---|---|---|\n" + rows

    mock_doc = MagicMock()
    mock_doc.text = table_text
    mock_doc.metadata = {"page_number": 1}

    mock_parser = mock_llamaparse_class.return_value
    mock_parser.load_data.return_value = [mock_doc]

    chunks = process_pdf_documents("/fake/path/table.pdf")

    assert len(chunks) > 1, "table should be split into multiple chunks for this to be a meaningful test"
    for chunk in chunks:
        assert chunk.page_content.strip().startswith("| Metric | 2022 | 2023 |"), (
            "every chunk of a split table must start with the column header row"
        )
        # The internal bookkeeping key must never leak into stored metadata.
        assert "_table_header" not in chunk.metadata


class TestEmbeddingModelGuard:
    """Tests for `_check_or_record_embedding_model` (embedding-space mismatch guard)."""

    def test_records_marker_on_first_index(self):
        """No marker point yet -> one is written with the active embedding model."""
        client = MagicMock()
        client.retrieve.return_value = []
        embeddings = MagicMock()
        embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        sparse_embeddings = MagicMock()
        sparse_vec = MagicMock()
        sparse_vec.indices = [0, 1]
        sparse_vec.values = [0.5, 0.5]
        sparse_embeddings.embed_query.return_value = sparse_vec

        _check_or_record_embedding_model(client, embeddings, sparse_embeddings)

        client.upsert.assert_called_once()
        _, kwargs = client.upsert.call_args
        point = kwargs["points"][0]
        assert point.payload["embedding_model"]

    def test_passes_when_stored_model_matches(self):
        """Marker already matches the active model -> no error, no re-write."""
        from app.config import EMBEDDING_MODEL

        client = MagicMock()
        marker_point = MagicMock()
        marker_point.payload = {"embedding_model": EMBEDDING_MODEL}
        client.retrieve.return_value = [marker_point]

        _check_or_record_embedding_model(client, MagicMock(), MagicMock())

        client.upsert.assert_not_called()

    def test_raises_when_stored_model_differs(self):
        """Marker recorded a different model -> reject with a clear error."""
        client = MagicMock()
        marker_point = MagicMock()
        marker_point.payload = {"embedding_model": "some-other-embedding-model"}
        client.retrieve.return_value = [marker_point]

        with pytest.raises(ValueError, match="berbeda dari model"):
            _check_or_record_embedding_model(client, MagicMock(), MagicMock())

        client.upsert.assert_not_called()


class TestIndexDocumentsRollback:
    """Tests for the batch-indexing rollback in `index_documents`."""

    def _make_fake_sparse_vector(self):
        v = MagicMock()
        v.indices = [0, 1]
        v.values = [0.1, 0.2]
        return v

    @patch("app.index_data.process_pdf_documents")
    @patch("app.index_data.get_sparse_embeddings")
    @patch("app.index_data.get_embeddings")
    @patch("app.index_data.get_qdrant_client")
    @patch("app.index_data.Path")
    @patch("app.index_data.INDEXING_BATCH_SIZE", 1)
    def test_rolls_back_points_when_a_batch_fails(
        self,
        mock_path_class,
        mock_get_client,
        mock_get_embeddings,
        mock_get_sparse_embeddings,
        mock_process_pdf,
    ):
        mock_path_instance = mock_path_class.return_value
        mock_path_instance.exists.return_value = True
        mock_path_instance.name = "doc.pdf"

        client = MagicMock()
        mock_get_client.return_value = client

        existing_collection = MagicMock()
        existing_collection.name = "invenioai_collection"
        client.get_collections.return_value = MagicMock(collections=[existing_collection])
        # No embedding-model marker yet, and no marker write should block the test.
        client.retrieve.return_value = []

        embeddings = MagicMock()
        embeddings.embed_query.return_value = [0.1, 0.2]
        embeddings.embed_documents.side_effect = lambda texts: [[0.1, 0.2]] * len(texts)
        mock_get_embeddings.return_value = embeddings

        sparse_embeddings = MagicMock()
        sparse_embeddings.embed_query.return_value = self._make_fake_sparse_vector()
        sparse_embeddings.embed_documents.side_effect = (
            lambda texts: [self._make_fake_sparse_vector() for _ in texts]
        )
        mock_get_sparse_embeddings.return_value = sparse_embeddings

        # 3 chunks, INDEXING_BATCH_SIZE=1 -> 3 separate upsert calls: marker
        # upsert (from the embedding-model guard) + batch 1 + batch 2 (fails).
        mock_process_pdf.return_value = [
            LangChainDocument(page_content=f"chunk {i}", metadata={"source_file": "doc.pdf", "source": "/fake/doc.pdf"})
            for i in range(3)
        ]
        client.upsert.side_effect = [None, None, Exception("Qdrant boom")]

        with pytest.raises(Exception, match="Qdrant boom"):
            index_documents("/fake/doc.pdf")

        # Rollback delete must target this document's points.
        client.delete.assert_called_once()
        _, kwargs = client.delete.call_args
        assert kwargs["collection_name"] == "invenioai_collection"
        filter_obj = kwargs["points_selector"].filter
        assert filter_obj.should[0].match.value == "doc.pdf"

