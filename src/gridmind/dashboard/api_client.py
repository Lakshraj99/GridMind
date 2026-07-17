"""Typed, retry-aware HTTP client used exclusively by dashboard views."""

from __future__ import annotations

import time
from typing import Any

import httpx


class DashboardAPIError(RuntimeError):
    """Base error suitable for safe display in Streamlit."""


class DashboardAuthenticationError(DashboardAPIError):
    """The configured dashboard API key was rejected."""


class DashboardConnectionError(DashboardAPIError):
    """The API could not be reached before the timeout."""


class GridMindAPIClient:
    """Small injectable client; only safe GET calls receive automatic retries."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float = 15,
        get_retries: int = 2,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        headers = {"X-API-Key": api_key} if api_key else {}
        self.client = httpx.Client(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout, transport=transport
        )
        self.get_retries = get_retries

    def close(self) -> None:
        self.client.close()

    def get(self, path: str, **params: object) -> dict[str, Any]:
        for attempt in range(self.get_retries + 1):
            try:
                response = self.client.get(
                    path, params={k: str(v) for k, v in params.items() if v is not None}
                )
                if response.status_code >= 500 and attempt < self.get_retries:
                    continue
                return self._decode(response)
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt == self.get_retries:
                    raise DashboardConnectionError("GridMind API is unavailable.") from exc
                time.sleep(0)
        raise DashboardConnectionError("GridMind API is unavailable.")

    def patch(self, path: str, payload: dict[str, object]) -> dict[str, Any]:
        try:
            return self._decode(self.client.patch(path, json=payload))
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise DashboardConnectionError("GridMind API is unavailable.") from exc

    @staticmethod
    def _decode(response: httpx.Response) -> dict[str, Any]:
        if response.status_code == 401:
            raise DashboardAuthenticationError("Dashboard authentication failed.")
        if response.is_error:
            try:
                message = response.json()["error"]["message"]
            except (KeyError, TypeError, ValueError):
                message = f"GridMind API returned HTTP {response.status_code}."
            raise DashboardAPIError(str(message))
        data = response.json()
        if not isinstance(data, dict):
            raise DashboardAPIError("GridMind API returned an invalid response.")
        return data
