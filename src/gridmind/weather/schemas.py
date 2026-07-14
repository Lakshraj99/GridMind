"""Strict Pandera contracts for location-level and aggregated weather."""

from __future__ import annotations

from datetime import UTC

import pandas as pd
import pandera.pandas as pa
from pandera.engines.pandas_engine import DateTime
from pandera.errors import SchemaError, SchemaErrors

from gridmind.exceptions import DataValidationError

WEATHER_COLUMNS = [
    "timestamp_utc",
    "region",
    "location_name",
    "latitude",
    "longitude",
    "location_weight",
    "temperature_c",
    "apparent_temperature_c",
    "relative_humidity_pct",
    "precipitation_mm",
    "cloud_cover_pct",
    "wind_speed_10m_kph",
    "wind_direction_10m_deg",
    "shortwave_radiation_wm2",
    "direct_radiation_wm2",
    "diffuse_radiation_wm2",
    "ingestion_timestamp_utc",
    "data_source",
    "weather_data_type",
]

REGION_WEATHER_COLUMNS = [
    "timestamp_utc",
    "region",
    "weather_data_type",
    "temperature_c",
    "apparent_temperature_c",
    "relative_humidity_pct",
    "precipitation_mm",
    "cloud_cover_pct",
    "wind_speed_10m_kph",
    "wind_direction_10m_deg",
    "wind_direction_sin",
    "wind_direction_cos",
    "shortwave_radiation_wm2",
    "direct_radiation_wm2",
    "diffuse_radiation_wm2",
    "temperature_min_c",
    "temperature_max_c",
    "temperature_spread_c",
    "wind_speed_spread_kph",
    "ingestion_timestamp_utc",
    "data_source",
]

_utc = pa.Column(DateTime(tz=UTC), nullable=False, coerce=True)  # type: ignore[call-arg]
WEATHER_SCHEMA = pa.DataFrameSchema(
    {
        "timestamp_utc": _utc,
        "region": pa.Column(pa.String, nullable=False, coerce=True),
        "location_name": pa.Column(pa.String, nullable=False, coerce=True),
        "latitude": pa.Column(float, pa.Check.in_range(-90, 90), coerce=True),
        "longitude": pa.Column(float, pa.Check.in_range(-180, 180), coerce=True),
        "location_weight": pa.Column(float, pa.Check.gt(0), coerce=True),
        "temperature_c": pa.Column(float, nullable=True, coerce=True),
        "apparent_temperature_c": pa.Column(float, nullable=True, coerce=True),
        "relative_humidity_pct": pa.Column(
            float, pa.Check.in_range(0, 100), nullable=True, coerce=True
        ),
        "precipitation_mm": pa.Column(float, pa.Check.ge(0), nullable=True, coerce=True),
        "cloud_cover_pct": pa.Column(float, pa.Check.in_range(0, 100), nullable=True, coerce=True),
        "wind_speed_10m_kph": pa.Column(float, pa.Check.ge(0), nullable=True, coerce=True),
        "wind_direction_10m_deg": pa.Column(
            float, pa.Check.in_range(0, 360), nullable=True, coerce=True
        ),
        "shortwave_radiation_wm2": pa.Column(float, pa.Check.ge(0), nullable=True, coerce=True),
        "direct_radiation_wm2": pa.Column(float, pa.Check.ge(0), nullable=True, coerce=True),
        "diffuse_radiation_wm2": pa.Column(float, pa.Check.ge(0), nullable=True, coerce=True),
        "ingestion_timestamp_utc": _utc,
        "data_source": pa.Column(pa.String, nullable=False, coerce=True),
        "weather_data_type": pa.Column(
            pa.String, pa.Check.isin(["historical", "forecast"]), coerce=True
        ),
    },
    strict=True,
    checks=[
        pa.Check(
            lambda x: (
                ~x.duplicated(
                    ["region", "location_name", "timestamp_utc", "weather_data_type"]
                ).any()
            )
        )
    ],
)


def validate_weather_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate location weather and stable UTC/order semantics."""
    try:
        result = WEATHER_SCHEMA.validate(frame[WEATHER_COLUMNS], lazy=True)
    except (SchemaError, SchemaErrors) as exc:
        raise DataValidationError(f"Weather data schema validation failed: {exc}") from exc
    if str(result["timestamp_utc"].dt.tz) != "UTC":
        raise DataValidationError("Weather timestamps must use UTC.")
    return result.sort_values(
        ["region", "location_name", "timestamp_utc", "weather_data_type"], ignore_index=True
    )
