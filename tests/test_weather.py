"""Offline weather client, mapping, processing, aggregation, and storage tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest

from gridmind.exceptions import DataValidationError, WeatherClientError, WeatherLocationError
from gridmind.weather.client import OPEN_METEO_FIELDS, WeatherClient
from gridmind.weather.locations import RegionLocationMapping, WeatherLocation, load_region_locations
from gridmind.weather.processing import (
    aggregate_region_weather,
    normalize_weather_pages,
    weather_quality_report,
)
from gridmind.weather.storage import WeatherStorage, write_weather_parquet


def _payload(*, direction: float = 10.0, start: str = "2024-01-01T00:00") -> dict[str, Any]:
    times = [start, (pd.Timestamp(start) + pd.Timedelta(hours=1)).isoformat(timespec="minutes")]
    hourly: dict[str, Any] = {"time": times}
    for field in OPEN_METEO_FIELDS:
        hourly[field] = [direction, direction] if field == "wind_direction_10m" else [10.0, 11.0]
    return {"latitude": 40.0, "longitude": -75.0, "hourly": hourly}


def _mapping() -> RegionLocationMapping:
    return RegionLocationMapping(
        region="PJM",
        version="test",
        source="fixture",
        rationale="test",
        locations=(
            WeatherLocation(name="East", latitude=40, longitude=-75, weight=2),
            WeatherLocation(name="West", latitude=41, longitude=-80, weight=1),
        ),
    )


def test_location_yaml_validation_and_weight_normalization(tmp_path: Path) -> None:
    path = tmp_path / "locations.yaml"
    path.write_text(
        """version: v1
source: fixture
regions:
  PJM:
    rationale: fixture
    aggregation: weighted_mean
    locations:
      - {name: A, latitude: 40, longitude: -75, weight: 2}
      - {name: B, latitude: 41, longitude: -80, weight: 1}
""",
        encoding="utf-8",
    )
    mapping = load_region_locations(path, "PJM")
    assert sum(item.weight for item in mapping.locations) == pytest.approx(1.0)
    assert mapping.locations[0].weight == pytest.approx(2 / 3)
    with pytest.raises(WeatherLocationError):
        load_region_locations(path, "MISO")
    with pytest.raises(ValueError):
        WeatherLocation(name="bad", latitude=100, longitude=0, weight=1)


def test_weather_client_chunking_cache_request_and_retry(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.url.params["timezone"] == "UTC"
        assert "temperature_2m" in request.url.params["hourly"]
        if calls == 1:
            return httpx.Response(500, request=request)
        return httpx.Response(200, json=_payload(), request=request)

    client = WeatherClient(
        historical_url="https://weather.test/archive",
        forecast_url="https://weather.test/forecast",
        cache_dir=tmp_path,
        max_retries=1,
        chunk_days=1,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda _seconds: None,
    )
    location = WeatherLocation(name="A", latitude=40, longitude=-75, weight=1)
    first = client.fetch(location, "2024-01-01", "2024-01-02")
    assert len(first.pages) == 2
    assert calls == 3
    cached = client.fetch(location, "2024-01-01", "2024-01-02")
    assert cached.cache_hits == 2
    assert calls == 3
    assert len(list((tmp_path / "raw").glob("*.json"))) == 2


def test_weather_client_malformed_and_network_failures(tmp_path: Path) -> None:
    malformed = httpx.MockTransport(lambda request: httpx.Response(200, json={}, request=request))
    client = WeatherClient(
        historical_url="https://weather.test",
        forecast_url="https://weather.test",
        cache_dir=tmp_path / "bad",
        client=httpx.Client(transport=malformed),
    )
    location = WeatherLocation(name="A", latitude=40, longitude=-75, weight=1)
    with pytest.raises(WeatherClientError, match=r"hourly\.time"):
        client.fetch(location, "2024-01-01", "2024-01-01")

    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    unavailable = WeatherClient(
        historical_url="https://weather.test",
        forecast_url="https://weather.test",
        cache_dir=tmp_path / "offline",
        max_retries=1,
        client=httpx.Client(transport=httpx.MockTransport(fail)),
        sleep=lambda _seconds: None,
    )
    with pytest.raises(WeatherClientError, match="after retries"):
        unavailable.fetch(location, "2024-01-01", "2024-01-01")


def test_weather_schema_weighted_and_circular_aggregation(tmp_path: Path) -> None:
    mapping = _mapping()
    east = normalize_weather_pages(
        (_payload(direction=350),),
        region="PJM",
        location=mapping.locations[0],
        data_type="historical",
        ingestion_timestamp=pd.Timestamp("2024-02-01", tz="UTC"),
    )
    west = normalize_weather_pages(
        (_payload(direction=10),),
        region="PJM",
        location=mapping.locations[1],
        data_type="historical",
        ingestion_timestamp=pd.Timestamp("2024-02-01", tz="UTC"),
    )
    location_data = pd.concat([east, west], ignore_index=True)
    regional = aggregate_region_weather(location_data, mapping)
    assert len(regional) == 2
    assert regional["wind_direction_10m_deg"].iloc[0] > 350
    assert regional["temperature_c"].iloc[0] == pytest.approx(10.0)
    assert weather_quality_report(location_data, regional)["duplicate_rows"] == 0
    with pytest.raises(DataValidationError, match="expected"):
        aggregate_region_weather(east, mapping)
    assert aggregate_region_weather(east, mapping, require_all_locations=False).empty

    database = tmp_path / "weather.duckdb"
    storage = WeatherStorage(database)
    assert storage.upsert_locations(location_data) == 4
    assert storage.upsert_locations(location_data) == 4
    assert storage.upsert_regions(regional) == 2
    assert storage.upsert_regions(regional) == 2
    assert len(storage.read_regions("PJM", data_type="historical")) == 2
    path = write_weather_parquet(regional, tmp_path / "regional", aggregated=True)
    assert path.suffix == ".parquet"
    assert not list(path.parent.glob("*.json"))


def test_weather_schema_rejects_invalid_ranges() -> None:
    mapping = _mapping()
    page = _payload()
    page["hourly"]["relative_humidity_2m"] = [101.0, 50.0]
    with pytest.raises(DataValidationError):
        normalize_weather_pages(
            (page,), region="PJM", location=mapping.locations[0], data_type="forecast"
        )
