"""Pandera-backed raw and canonical grid-data contracts."""

from __future__ import annotations

from datetime import UTC

import pandas as pd
import pandera.pandas as pa
from pandera.engines.pandas_engine import DateTime
from pandera.errors import SchemaError, SchemaErrors

from gridmind.exceptions import DataValidationError

RAW_SCHEMA = pa.DataFrameSchema(
    {
        "timestamp": pa.Column(object, nullable=False),
        "region": pa.Column(object, nullable=False),
        "measurement_type": pa.Column(object, nullable=False),
        "value": pa.Column(object, nullable=True),
    },
    strict=True,
    coerce=False,
    name="raw_eia_data",
)

PROCESSED_SCHEMA = pa.DataFrameSchema(
    {
        "timestamp_utc": pa.Column(
            DateTime(tz=UTC),  # type: ignore[call-arg]
            nullable=False,
            coerce=True,
        ),
        "region": pa.Column(pa.String, nullable=False, coerce=True),
        "demand_mw": pa.Column(float, pa.Check.ge(0), nullable=False, coerce=True),
        "forecast_demand_mw": pa.Column(float, nullable=True, coerce=True),
        "net_generation_mw": pa.Column(float, nullable=True, coerce=True),
        "total_interchange_mw": pa.Column(float, nullable=True, coerce=True),
        "ingestion_timestamp_utc": pa.Column(
            DateTime(tz=UTC),  # type: ignore[call-arg]
            nullable=False,
            coerce=True,
        ),
    },
    strict=True,
    coerce=False,
    checks=[
        pa.Check(
            lambda frame: ~frame.duplicated(["region", "timestamp_utc"]).any(),
            error="region and timestamp_utc must be unique",
        )
    ],
    name="processed_grid_data",
)


def validate_raw_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate normalized EIA records and provide contextual failures."""
    try:
        return RAW_SCHEMA.validate(frame, lazy=True)
    except (SchemaError, SchemaErrors) as exc:
        raise DataValidationError(f"Raw data schema validation failed: {exc}") from exc


def validate_processed_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate canonical grid data, including UTC timestamps and ordering."""
    try:
        validated = PROCESSED_SCHEMA.validate(frame, lazy=True)
    except (SchemaError, SchemaErrors) as exc:
        raise DataValidationError(f"Processed data schema validation failed: {exc}") from exc
    for column in ("timestamp_utc", "ingestion_timestamp_utc"):
        if not isinstance(validated[column].dtype, pd.DatetimeTZDtype):
            raise DataValidationError(f"Column '{column}' must be timezone-aware UTC datetime.")
        if str(validated[column].dt.tz) != "UTC":
            raise DataValidationError(f"Column '{column}' must use UTC timezone.")
    expected_order = validated.sort_values(["region", "timestamp_utc"]).index
    if not validated.index.equals(expected_order):
        raise DataValidationError("Processed data must be sorted by region and timestamp_utc.")
    return validated
