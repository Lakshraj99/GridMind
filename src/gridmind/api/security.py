"""Optional constant-time local API-key authentication."""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Header, HTTPException, Request, status


def require_api_key(
    request: Request, x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None
) -> None:
    settings = request.app.state.settings
    if not settings.api_key_enabled:
        return
    configured = settings.gridmind_api_key or ""
    supplied = x_api_key or ""
    if not supplied or not secrets.compare_digest(supplied, configured):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A valid X-API-Key is required.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
