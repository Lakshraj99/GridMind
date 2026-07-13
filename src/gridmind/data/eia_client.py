"""Typed client for the EIA v2 hourly balancing-authority API."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import httpx

from gridmind.exceptions import (
    EIAAuthenticationError,
    EIAMalformedResponseError,
    EIANetworkError,
    EIARateLimitError,
)


@dataclass(frozen=True)
class EIAFetchResult:
    """Records and unmodified response pages returned by an EIA fetch."""

    records: list[dict[str, Any]]
    pages: list[dict[str, Any]]


class EIAClient:
    """Communicate with EIA without performing data modelling or persistence."""

    route = "/electricity/rto/region-data/data/"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.eia.gov/v2",
        *,
        timeout: float = 30.0,
        page_size: int = 5000,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Create a client; an injected HTTP client makes all network behavior testable."""
        if not api_key:
            raise EIAAuthenticationError("An EIA API key is required to create an EIA client.")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._page_size = page_size
        self._max_retries = max_retries
        self._backoff_factor = backoff_factor
        self._client = client or httpx.Client()
        self._owns_client = client is None
        self._sleep = sleep

    @property
    def data_url(self) -> str:
        """Return the isolated hourly region-data endpoint URL."""
        return f"{self._base_url}{self.route}"

    @property
    def redaction_secrets(self) -> tuple[str, ...]:
        """Return credentials exclusively for sanitizing response artifacts."""
        return (self._api_key,)

    def build_params(
        self,
        region: str,
        start_date: str | date | datetime,
        end_date: str | date | datetime,
        *,
        offset: int = 0,
    ) -> dict[str, str | int]:
        """Build EIA query parameters without exposing the key through logs."""
        start = _format_date(start_date)
        end = _format_date(end_date)
        return {
            "api_key": self._api_key,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": region,
            "start": start,
            "end": end,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "sort[1][column]": "type",
            "sort[1][direction]": "asc",
            "offset": offset,
            "length": self._page_size,
        }

    def fetch_hourly_data(
        self,
        region: str,
        start_date: str | date | datetime,
        end_date: str | date | datetime,
    ) -> EIAFetchResult:
        """Fetch every page of hourly balancing-authority measurements."""
        records: list[dict[str, Any]] = []
        pages: list[dict[str, Any]] = []
        offset = 0
        while True:
            payload = self._request(self.build_params(region, start_date, end_date, offset=offset))
            response = payload.get("response")
            if not isinstance(response, Mapping) or not isinstance(response.get("data"), list):
                raise EIAMalformedResponseError("EIA response must contain a 'response.data' list.")
            page_records = response["data"]
            if not all(isinstance(item, dict) for item in page_records):
                raise EIAMalformedResponseError("EIA response data contains a non-object record.")
            typed_records = [dict(item) for item in page_records]
            pages.append(payload)
            records.extend(typed_records)

            total_raw = response.get("total", len(records))
            try:
                total = int(total_raw)
            except (TypeError, ValueError) as exc:
                raise EIAMalformedResponseError("EIA response 'total' must be an integer.") from exc
            if not typed_records:
                break
            offset += len(typed_records)
            # A full page warrants one more request even when an older EIA API
            # version under-reports ``total``. A short page is complete only
            # when it also reaches the advertised total.
            if len(typed_records) < self._page_size and offset >= total:
                break
        return EIAFetchResult(records=records, pages=pages)

    def close(self) -> None:
        """Close an internally-created HTTP client."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> EIAClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _request(self, params: dict[str, str | int]) -> dict[str, Any]:
        last_status: int | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.get(self.data_url, params=params, timeout=self._timeout)
            except httpx.RequestError:
                if attempt == self._max_retries:
                    raise EIANetworkError(
                        "EIA request failed after retries; check network connectivity and retry."
                    ) from None
                self._sleep(self._backoff_factor * (2**attempt))
                continue

            last_status = response.status_code
            if response.status_code in (401, 403):
                raise EIAAuthenticationError(
                    "EIA authentication failed; verify that EIA_API_KEY is valid."
                )
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < self._max_retries:
                    self._sleep(self._backoff_factor * (2**attempt))
                    continue
                if response.status_code == 429:
                    raise EIARateLimitError("EIA rate limit exceeded after retries.")
                raise EIANetworkError(
                    f"EIA server returned HTTP {response.status_code} after retries."
                )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError:
                raise EIANetworkError(
                    f"EIA request returned HTTP {response.status_code}."
                ) from None
            try:
                payload = response.json()
            except ValueError as exc:
                raise EIAMalformedResponseError("EIA returned invalid JSON.") from exc
            if not isinstance(payload, dict):
                raise EIAMalformedResponseError("EIA response root must be a JSON object.")
            return payload
        raise EIANetworkError(f"EIA request failed with HTTP {last_status}.")  # pragma: no cover


def _format_date(value: str | date | datetime) -> str:
    """Format accepted date values for the EIA API."""
    return value.isoformat() if isinstance(value, (date, datetime)) else value
