"""Tests for the optional API key auth (`app.auth.require_api_key`)."""
import pytest
from unittest.mock import patch
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.auth import require_api_key
from app.main import app

client = TestClient(app)


@pytest.mark.asyncio
async def test_disabled_by_default_allows_missing_header():
    """API_KEY unset (default/local/demo) -> no header required."""
    with patch("app.auth.API_KEY", None):
        # Should not raise.
        await require_api_key(x_api_key=None)


@pytest.mark.asyncio
async def test_rejects_missing_key_when_enabled():
    with patch("app.auth.API_KEY", "secret123"):
        with pytest.raises(HTTPException) as exc_info:
            await require_api_key(x_api_key=None)
        assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_rejects_wrong_key_when_enabled():
    with patch("app.auth.API_KEY", "secret123"):
        with pytest.raises(HTTPException) as exc_info:
            await require_api_key(x_api_key="wrong-key")
        assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_accepts_correct_key_when_enabled():
    with patch("app.auth.API_KEY", "secret123"):
        # Should not raise.
        await require_api_key(x_api_key="secret123")


def test_metrics_endpoint_rejects_missing_key_when_enabled():
    """End-to-end: a protected route 401s without X-API-Key once auth is on."""
    with patch("app.auth.API_KEY", "secret123"):
        response = client.get("/metrics")
    assert response.status_code == 401


def test_metrics_endpoint_accepts_correct_key():
    with patch("app.auth.API_KEY", "secret123"):
        response = client.get("/metrics", headers={"X-API-Key": "secret123"})
    assert response.status_code == 200


def test_root_health_check_never_requires_auth():
    """The `/` health check endpoint must stay open even with auth enabled -
    orchestrators/start scripts poll it before any API key is available."""
    with patch("app.auth.API_KEY", "secret123"):
        response = client.get("/")
    assert response.status_code == 200
