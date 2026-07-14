"""Typed, cached, retrying Open-Meteo historical and forecast client."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

import httpx

from gridmind.exceptions import WeatherClientError
from gridmind.weather.locations import WeatherLocation

OPEN_METEO_FIELDS = (
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
)


@dataclass(frozen=True)
class WeatherFetchResult:
    """Provider pages and cache accounting for one location/range."""

    pages: tuple[dict[str, Any], ...]
    cache_hits: int


class WeatherClient:
    """Fetch Open-Meteo hourly UTC weather in bounded, cacheable date chunks."""

    def __init__(
        self,
        *,
        historical_url: str,
        forecast_url: str,
        cache_dir: Path,
        timeout: float = 30.0,
        max_retries: int = 3,
        chunk_days: int = 31,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if timeout <= 0 or max_retries < 0 or chunk_days <= 0:
            raise ValueError(
                "Weather timeout/chunk size must be positive and retries non-negative."
            )
        self.historical_url = historical_url
        self.forecast_url = forecast_url
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.max_retries = max_retries
        self.chunk_days = chunk_days
        self._client = client or httpx.Client()
        self._owns_client = client is None
        self._sleep = sleep

    def build_params(
        self, location: WeatherLocation, start: date, end: date
    ) -> dict[str, str | float]:
        """Build a credential-free UTC hourly provider request."""
        return {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "hourly": ",".join(OPEN_METEO_FIELDS),
            "timezone": "UTC",
        }

    def fetch(
        self,
        location: WeatherLocation,
        start_date: str | date,
        end_date: str | date,
        *,
        data_type: Literal["historical", "forecast"] = "historical",
    ) -> WeatherFetchResult:
        """Fetch every date chunk, reusing identical successful cached responses."""
        start = date.fromisoformat(start_date) if isinstance(start_date, str) else start_date
        end = date.fromisoformat(end_date) if isinstance(end_date, str) else end_date
        if end < start:
            raise ValueError("Weather end date must not precede start date.")
        url = self.historical_url if data_type == "historical" else self.forecast_url
        pages: list[dict[str, Any]] = []
        hits = 0
        cursor = start
        while cursor <= end:
            chunk_end = min(cursor + timedelta(days=self.chunk_days - 1), end)
            params = self.build_params(location, cursor, chunk_end)
            cache_path = self._cache_path(url, params)
            if cache_path.exists():
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                hits += 1
            else:
                payload = self._request(url, params)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(payload), encoding="utf-8")
            self._validate_payload(payload)
            pages.append(payload)
            cursor = chunk_end + timedelta(days=1)
        return WeatherFetchResult(tuple(pages), hits)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _request(self, url: str, params: dict[str, str | float]) -> dict[str, Any]:
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.get(url, params=params, timeout=self.timeout)
            except httpx.RequestError:
                if attempt == self.max_retries:
                    raise WeatherClientError(
                        "Weather request failed after retries; check connectivity."
                    ) from None
                self._sleep(0.5 * (2**attempt))
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < self.max_retries:
                    self._sleep(0.5 * (2**attempt))
                    continue
                raise WeatherClientError(
                    f"Weather provider returned HTTP {response.status_code} after retries."
                )
            try:
                response.raise_for_status()
                payload = response.json()
            except (httpx.HTTPStatusError, ValueError) as exc:
                raise WeatherClientError("Weather provider returned an invalid response.") from exc
            if not isinstance(payload, dict):
                raise WeatherClientError("Weather response root must be an object.")
            return payload
        raise WeatherClientError("Weather request failed.")  # pragma: no cover

    def _cache_path(self, url: str, params: dict[str, str | float]) -> Path:
        identity = json.dumps([url, sorted(params.items())], separators=(",", ":"))
        digest = hashlib.sha256(identity.encode()).hexdigest()
        return self.cache_dir / "raw" / f"{digest}.json"

    @staticmethod
    def _validate_payload(payload: dict[str, Any]) -> None:
        hourly = payload.get("hourly")
        if not isinstance(hourly, dict) or not isinstance(hourly.get("time"), list):
            raise WeatherClientError("Weather response must contain hourly.time.")
        size = len(hourly["time"])
        for field in OPEN_METEO_FIELDS:
            if not isinstance(hourly.get(field), list) or len(hourly[field]) != size:
                raise WeatherClientError(
                    f"Weather hourly field '{field}' is missing or misaligned."
                )
