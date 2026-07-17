"""Dashboard client construction from validated environment settings."""

from gridmind.config import Settings
from gridmind.dashboard.api_client import GridMindAPIClient


def create_client(settings: Settings) -> GridMindAPIClient:
    return GridMindAPIClient(
        settings.dashboard_api_base_url,
        api_key=settings.gridmind_api_key,
        timeout=settings.dashboard_request_timeout_seconds,
    )
