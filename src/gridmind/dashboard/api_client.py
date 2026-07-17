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
        self._get_cache: dict[tuple[str, tuple[tuple[str, str], ...]], dict[str, Any]] = {}

    def close(self) -> None:
        self.client.close()

    def get(self, path: str, **params: object) -> dict[str, Any]:
        serialized = tuple(
            sorted((key, str(value)) for key, value in params.items() if value is not None)
        )
        cache_key = (path, serialized)
        cached = self._get_cache.get(cache_key)
        if cached is not None:
            return cached
        for attempt in range(self.get_retries + 1):
            try:
                response = self.client.get(
                    path, params={k: str(v) for k, v in params.items() if v is not None}
                )
                if response.status_code >= 500 and attempt < self.get_retries:
                    continue
                result = self._decode(response)
                self._get_cache[cache_key] = result
                return result
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt == self.get_retries:
                    raise DashboardConnectionError("GridMind API is unavailable.") from exc
                time.sleep(0)
        raise DashboardConnectionError("GridMind API is unavailable.")

    def patch(self, path: str, payload: dict[str, object]) -> dict[str, Any]:
        try:
            result = self._decode(self.client.patch(path, json=payload))
            self.clear_cache()
            return result
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise DashboardConnectionError("GridMind API is unavailable.") from exc

    @staticmethod
    def _decode(response: httpx.Response) -> dict[str, Any]:
        if response.status_code in {401, 403}:
            raise DashboardAuthenticationError(
                "The API rejected the dashboard credentials. Check local configuration."
            )
        if response.is_error:
            try:
                message = response.json()["error"]["message"]
            except (KeyError, TypeError, ValueError):
                message = f"GridMind API returned HTTP {response.status_code}."
            raise DashboardAPIError(str(message))
        try:
            data = response.json()
        except ValueError as exc:
            raise DashboardAPIError("GridMind API returned an invalid response.") from exc
        if not isinstance(data, dict):
            raise DashboardAPIError("GridMind API returned an invalid response.")
        return data

    def clear_cache(self) -> None:
        """Invalidate safe GET results after a manual refresh or lifecycle write."""
        self._get_cache.clear()
