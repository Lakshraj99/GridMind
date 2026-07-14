"""Normalize Open-Meteo payloads and aggregate representative locations."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

from gridmind.exceptions import DataValidationError
from gridmind.weather.client import OPEN_METEO_FIELDS
from gridmind.weather.locations import RegionLocationMapping, WeatherLocation
from gridmind.weather.schemas import REGION_WEATHER_COLUMNS, WEATHER_COLUMNS, validate_weather_data

FIELD_MAP = {
    "temperature_2m": "temperature_c",
    "apparent_temperature": "apparent_temperature_c",
    "relative_humidity_2m": "relative_humidity_pct",
    "precipitation": "precipitation_mm",
    "cloud_cover": "cloud_cover_pct",
    "wind_speed_10m": "wind_speed_10m_kph",
    "wind_direction_10m": "wind_direction_10m_deg",
    "shortwave_radiation": "shortwave_radiation_wm2",
    "direct_radiation": "direct_radiation_wm2",
    "diffuse_radiation": "diffuse_radiation_wm2",
}


def normalize_weather_pages(
    pages: tuple[dict[str, Any], ...],
    *,
    region: str,
    location: WeatherLocation,
    data_type: Literal["historical", "forecast"],
    ingestion_timestamp: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Convert provider arrays to strict location-level rows."""
    rows: list[dict[str, Any]] = []
    stamp = ingestion_timestamp or pd.Timestamp(datetime.now(UTC))
    for page in pages:
        hourly = page["hourly"]
        for index, timestamp in enumerate(hourly["time"]):
            row: dict[str, Any] = {
                "timestamp_utc": pd.Timestamp(timestamp, tz="UTC"),
                "region": region,
                "location_name": location.name,
                "latitude": float(page.get("latitude", location.latitude)),
                "longitude": float(page.get("longitude", location.longitude)),
                "location_weight": location.weight,
                "ingestion_timestamp_utc": stamp,
                "data_source": "open_meteo",
                "weather_data_type": data_type,
            }
            for source in OPEN_METEO_FIELDS:
                row[FIELD_MAP[source]] = hourly[source][index]
            rows.append(row)
    frame = pd.DataFrame(rows, columns=WEATHER_COLUMNS)
    for column in FIELD_MAP.values():
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    return validate_weather_data(frame)


def aggregate_region_weather(
    frame: pd.DataFrame, mapping: RegionLocationMapping, *, require_all_locations: bool = True
) -> pd.DataFrame:
    """Create weighted regional fields and circularly aggregate wind direction."""
    validated = validate_weather_data(frame)
    expected = {location.name for location in mapping.locations}
    rows: list[dict[str, Any]] = []
    keys = ["region", "timestamp_utc", "weather_data_type"]
    weighted = [
        "temperature_c",
        "apparent_temperature_c",
        "relative_humidity_pct",
        "precipitation_mm",
        "cloud_cover_pct",
        "wind_speed_10m_kph",
        "shortwave_radiation_wm2",
        "direct_radiation_wm2",
        "diffuse_radiation_wm2",
    ]
    for key, group in validated.groupby(keys, sort=True, observed=True):
        present = set(group["location_name"].astype(str))
        if present != expected:
            if require_all_locations:
                raise DataValidationError(
                    f"Regional weather at {key[1]} has locations {sorted(present)}; "
                    f"expected {sorted(expected)}."
                )
            continue
        weights = group["location_weight"].to_numpy(dtype=float)
        weights = weights / weights.sum()
        direction = np.deg2rad(group["wind_direction_10m_deg"].to_numpy(dtype=float))
        sin_value = float(np.sum(np.sin(direction) * weights))
        cos_value = float(np.sum(np.cos(direction) * weights))
        row: dict[str, Any] = dict(zip(keys, key, strict=True))
        for column in weighted:
            row[column] = float(np.sum(group[column].to_numpy(dtype=float) * weights))
        row.update(
            {
                "wind_direction_sin": sin_value,
                "wind_direction_cos": cos_value,
                "wind_direction_10m_deg": float(np.degrees(np.arctan2(sin_value, cos_value)) % 360),
                "temperature_min_c": float(group["temperature_c"].min()),
                "temperature_max_c": float(group["temperature_c"].max()),
                "temperature_spread_c": float(
                    group["temperature_c"].max() - group["temperature_c"].min()
                ),
                "wind_speed_spread_kph": float(
                    group["wind_speed_10m_kph"].max() - group["wind_speed_10m_kph"].min()
                ),
                "ingestion_timestamp_utc": group["ingestion_timestamp_utc"].max(),
                "data_source": "open_meteo",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=REGION_WEATHER_COLUMNS).sort_values(
        ["region", "timestamp_utc", "weather_data_type"], ignore_index=True
    )


def weather_quality_report(location_data: pd.DataFrame, regional: pd.DataFrame) -> dict[str, Any]:
    """Report coverage, duplicates, gaps, missing variables, and aggregation retention."""
    gaps = 0
    if not regional.empty:
        diffs = regional.groupby(["region", "weather_data_type"], observed=True)[
            "timestamp_utc"
        ].diff()
        gaps = int(sum(max(int(value / pd.Timedelta(hours=1)) - 1, 0) for value in diffs.dropna()))
    return {
        "location_rows": len(location_data),
        "regional_rows": len(regional),
        "duplicate_rows": int(
            location_data.duplicated(
                ["region", "location_name", "timestamp_utc", "weather_data_type"]
            ).sum()
        ),
        "timestamp_gap_count": gaps,
        "missing_by_variable": {
            key: int(value) for key, value in location_data.isna().sum().items()
        },
        "weather_data_types": sorted(
            str(value) for value in location_data["weather_data_type"].unique()
        ),
    }
