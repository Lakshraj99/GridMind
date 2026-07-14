"""Historical/forecast weather ingestion orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import duckdb
import pandas as pd

from gridmind.config import Settings
from gridmind.data.storage import DuckDBStorage, write_json_report
from gridmind.weather.client import WeatherClient
from gridmind.weather.locations import load_region_locations
from gridmind.weather.processing import (
    aggregate_region_weather,
    normalize_weather_pages,
    weather_quality_report,
)
from gridmind.weather.storage import WeatherStorage, write_weather_parquet


@dataclass(frozen=True)
class WeatherIngestionResult:
    location_rows: int
    regional_rows: int
    cache_hits: int
    processed_path: Path
    report_path: Path
    duckdb_rows: int


def run_weather_ingestion(
    settings: Settings,
    *,
    region: str,
    start_date: str,
    end_date: str,
    data_type: Literal["historical", "forecast"] = "historical",
    client: WeatherClient | None = None,
) -> WeatherIngestionResult:
    mapping = load_region_locations(settings.grid_location_config, region)
    active = client or WeatherClient(
        historical_url=settings.weather_base_url,
        forecast_url=settings.weather_forecast_base_url,
        cache_dir=settings.weather_cache_dir,
        timeout=settings.weather_request_timeout_seconds,
        max_retries=settings.weather_max_retries,
    )
    owns = client is None
    frames: list[pd.DataFrame] = []
    hits = 0
    try:
        for location in mapping.locations:
            fetched = active.fetch(location, start_date, end_date, data_type=data_type)
            hits += fetched.cache_hits
            frames.append(
                normalize_weather_pages(
                    fetched.pages,
                    region=region,
                    location=location,
                    data_type=data_type,
                )
            )
    finally:
        if owns:
            active.close()
    location_data = pd.concat(frames, ignore_index=True)
    regional = aggregate_region_weather(location_data, mapping)
    processed = write_weather_parquet(
        location_data, settings.weather_cache_dir / "processed" / "locations"
    )
    write_weather_parquet(
        regional, settings.weather_cache_dir / "processed" / "regions", aggregated=True
    )
    storage = WeatherStorage(settings.duckdb_path)
    storage.upsert_locations(location_data)
    duckdb_rows = storage.upsert_regions(regional)
    report = weather_quality_report(location_data, regional)
    report.update(
        {
            "region": region,
            "mapping_version": mapping.version,
            "mapping_source": mapping.source,
            "cache_hits": hits,
        }
    )
    start_utc = pd.Timestamp(start_date, tz="UTC").to_pydatetime()
    end_utc = (
        pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    ).to_pydatetime()
    try:
        grid = DuckDBStorage(settings.duckdb_path).read_data(
            regions=[region], start_date=start_utc, end_date=end_utc
        )
    except (duckdb.Error, OSError):
        grid = pd.DataFrame(columns=["region", "timestamp_utc"])
    overlap = grid[["region", "timestamp_utc"]].merge(
        regional[["region", "timestamp_utc"]],
        on=["region", "timestamp_utc"],
        how="inner",
    )
    report["grid_weather_overlap"] = {
        "grid_rows": len(grid),
        "regional_weather_rows": len(regional),
        "overlap_rows": len(overlap),
        "grid_rows_without_weather": len(grid) - len(overlap),
        "weather_rows_without_grid": len(regional) - len(overlap),
    }
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    report_path = write_json_report(
        report, settings.data_quality_dir / f"weather_coverage_{stamp}.json"
    )
    return WeatherIngestionResult(
        len(location_data), len(regional), hits, processed, report_path, duckdb_rows
    )
