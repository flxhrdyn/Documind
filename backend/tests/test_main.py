import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from unittest.mock import patch, AsyncMock

@pytest.mark.asyncio
@patch("app.rag_pipeline.rag_pipeline_stream_async")
async def test_query_stream_endpoint(mock_rag_pipeline_stream_async):
    # Setup mock to yield chunks
    async def mock_stream(*args, **kwargs):
        yield "chunk1"
        yield "chunk2"
    
    mock_rag_pipeline_stream_async.return_value = mock_stream()
    
    # Execute
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/query/stream", json={"question": "test query", "history": []})
    
    # Assertions
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    
    content = response.content.decode()
    assert "data: chunk1" in content
    assert "data: chunk2" in content


@pytest.mark.asyncio
async def test_healthz_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/healthz")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "InvenioAI"


@pytest.mark.asyncio
async def test_readyz_endpoint_success():
    with patch("app.qdrant_conn.get_qdrant_client") as mock_get_client:
        mock_client = mock_get_client.return_value
        mock_collection = type("Collection", (), {"name": "invenioai_collection"})()
        mock_client.get_collections.return_value = type("Collections", (), {"collections": [mock_collection]})()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/readyz")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["database"]["qdrant"] == "connected"
        assert data["database"]["collection_exists"] is True


@pytest.mark.asyncio
async def test_readyz_endpoint_failure():
    with patch("app.qdrant_conn.get_qdrant_client", side_effect=RuntimeError("Qdrant unreachable")):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/readyz")

        assert response.status_code == 503
        data = response.json()
        assert "Service Unavailable" in data["detail"]

