"""Strict renewable-generation and net-load contracts."""

from __future__ import annotations

from datetime import UTC

import pandas as pd
import pandera.pandas as pa
from pandera.engines.pandas_engine import DateTime
from pandera.errors import SchemaError, SchemaErrors

from gridmind.exceptions import RenewableDataError

RENEWABLE_COLUMNS = [
    "timestamp_utc",
    "region",
    "solar_generation_mw",
    "wind_generation_mw",
    "total_renewable_generation_mw",
    "ingestion_timestamp_utc",
]

RENEWABLE_SCHEMA = pa.DataFrameSchema(
    {
        "timestamp_utc": pa.Column(DateTime(tz=UTC), nullable=False, coerce=True),  # type: ignore[call-arg]
        "region": pa.Column(pa.String, nullable=False, coerce=True),
        "solar_generation_mw": pa.Column(float, pa.Check.ge(0), nullable=True, coerce=True),
        "wind_generation_mw": pa.Column(float, pa.Check.ge(0), nullable=True, coerce=True),
        "total_renewable_generation_mw": pa.Column(
            float, pa.Check.ge(0), nullable=True, coerce=True
        ),
        "ingestion_timestamp_utc": pa.Column(DateTime(tz=UTC), nullable=False, coerce=True),  # type: ignore[call-arg]
    },
    strict=True,
    checks=[pa.Check(lambda x: ~x.duplicated(["region", "timestamp_utc"]).any())],
)


def validate_renewable_data(frame: pd.DataFrame) -> pd.DataFrame:
    try:
        result = RENEWABLE_SCHEMA.validate(frame[RENEWABLE_COLUMNS], lazy=True)
    except (SchemaError, SchemaErrors) as exc:
        raise RenewableDataError(f"Renewable data schema validation failed: {exc}") from exc
    return result.sort_values(["region", "timestamp_utc"], ignore_index=True)
