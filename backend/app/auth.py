"""Optional API key auth.

Disabled by default (local/demo use). Set `INVENIOAI_API_KEY` to require a
matching `X-API-Key` header on protected endpoints.
"""

from __future__ import annotations

from fastapi import Header, HTTPException, status

from .config import API_KEY


async def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    if API_KEY is None:
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
