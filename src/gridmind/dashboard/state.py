"""Dashboard client construction and rerun-scoped application state."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from gridmind.config import Settings
from gridmind.dashboard.api_client import DashboardAPIError, GridMindAPIClient


@dataclass(frozen=True)
class DashboardContext:
    region: str
    refreshed_at: datetime
    live: bool
    ready: bool
    readiness: dict[str, Any] = field(default_factory=dict)
    forecast_summary: dict[str, Any] = field(default_factory=dict)


def create_client(settings: Settings) -> GridMindAPIClient:
    return GridMindAPIClient(
        settings.dashboard_api_base_url,
        api_key=settings.gridmind_api_key,
        timeout=settings.dashboard_request_timeout_seconds,
    )


def safe_get(
    client: GridMindAPIClient, path: str, **params: object
) -> tuple[dict[str, Any] | None, str | None]:
    """Return display-safe API results for independently optional dashboard panels."""
    try:
        return client.get(path, **params), None
    except DashboardAPIError as exc:
        return None, str(exc)


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def default_context() -> DashboardContext:
    """Provide a compatible context for direct page rendering in tests and extensions."""
    return DashboardContext(
        region="PJM",
        refreshed_at=utc_now(),
        live=False,
        ready=False,
    )
