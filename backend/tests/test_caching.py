import pytest
from unittest.mock import MagicMock, patch
from app.rag_pipeline import rag_pipeline
from app.cache_manager import CacheManager

@pytest.fixture
def mock_cache():
    with patch("app.rag_pipeline.get_cache_manager") as mock:
        cache_inst = MagicMock(spec=CacheManager)
        cache_inst.get_semantic.return_value = None
        mock.return_value = cache_inst
        yield cache_inst

def test_rag_pipeline_uses_cache(mock_cache):
    """Test that rag_pipeline checks and uses the L1 exact cache."""
    # Setup cache hit
    cached_result = {"answer": "Cached answer", "sources": []}
    mock_cache.get.return_value = cached_result

    question = "What is AI?"
    history = []

    with patch("app.rag_pipeline.rewrite_query", return_value=("what is ai?", [])):
        # Execute
        result = rag_pipeline(question, history)

    # Assert
    assert result == cached_result
    mock_cache.get.assert_called_once()

def test_rag_pipeline_saves_to_cache_on_miss(mock_cache):
    """Test that rag_pipeline saves to the L1 exact cache when it's a miss."""
    # Setup cache miss
    mock_cache.get.return_value = None

    with patch("app.rag_pipeline._run_rag_pipeline_with_query") as mock_run, \
         patch("app.rag_pipeline.rewrite_query", return_value=("rewritten", [])):
        real_result = {"answer": "Real answer", "sources": [], "metrics": {"docs_retrieved": 3}}
        mock_run.return_value = real_result

        question = "New question"
        history = []

        # Execute
        result = rag_pipeline(question, history)

        # Assert
        assert result == real_result
        assert mock_cache.get.call_count == 1
        assert mock_cache.set.call_count == 1

@pytest.mark.asyncio
async def test_rag_pipeline_stream_async_uses_cache(mock_cache):
    """Test that rag_pipeline_stream_async checks and uses the cache."""
    from app.rag_pipeline import rag_pipeline_stream_async
    import json
    
    # Setup cache hit
    cached_result = {"answer": "Cached async answer", "sources": []}
    mock_cache.get.return_value = cached_result

    query = "Stream question"
    history = []

    # Execute
    with patch("app.rag_pipeline.rewrite_query_async", return_value=("stream question", [])):
        chunks = []
        async for chunk in rag_pipeline_stream_async(query, history):
            chunks.append(json.loads(chunk.strip()))

    # Assert
    assert any(c["step"] == "cached" for c in chunks)
    assert any(c["step"] == "done" and c["answer"] == "Cached async answer" for c in chunks)
    mock_cache.get.assert_called_once()
