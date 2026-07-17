"""Console entry point for the Uvicorn API process."""

from __future__ import annotations

import uvicorn

from gridmind.config import Settings


def main() -> None:
    settings = Settings()
    if not settings.api_enabled:
        raise SystemExit("GridMind API is disabled by API_ENABLED=false.")
    uvicorn.run(
        "gridmind.api.app:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
    )
