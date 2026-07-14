"""Shared Milestone 3 forecast contract and idempotent DuckDB persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gridmind.data.duckdb_connection import connect_duckdb
from gridmind.exceptions import TargetForecastError
from gridmind.models.target_factory import TARGET_FORECAST_COLUMNS
from gridmind.renewables.targets import SUPPORTED_TARGETS


def validate_target_forecasts(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(TARGET_FORECAST_COLUMNS).difference(frame.columns)
    if missing:
        raise TargetForecastError(f"Target forecasts are missing columns: {sorted(missing)}")
    result = frame[TARGET_FORECAST_COLUMNS].copy()
    if not set(result["target"]).issubset(SUPPORTED_TARGETS):
        raise TargetForecastError("Target forecast contains an unsupported target.")
    for column in ("forecast_origin", "timestamp_utc", "created_at_utc"):
        result[column] = pd.to_datetime(result[column], utc=True, errors="raise")
    if not np.isfinite(result["predicted_value"].to_numpy(dtype=float)).all():
        raise TargetForecastError("Target forecasts contain non-finite values.")
    key = [
        "region",
        "target",
        "forecast_origin",
        "timestamp_utc",
        "model_name",
        "model_version",
        "weather_mode",
    ]
    if result.duplicated(key).any():
        raise TargetForecastError("Target forecasts contain duplicate contract keys.")
    return result.sort_values(["target", "region", "timestamp_utc"], ignore_index=True)


class TargetForecastStorage:
    table_name = "target_forecasts"

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def upsert(self, frame: pd.DataFrame) -> int:
        valid = validate_target_forecasts(frame)
        keys = [
            "region",
            "target",
            "forecast_origin",
            "timestamp_utc",
            "model_name",
            "model_version",
            "weather_mode",
        ]
        predicate = " AND ".join(f"source.{key} = target.{key}" for key in keys)
        with connect_duckdb(self.path) as connection:
            connection.register("incoming_target_forecasts", valid)
            connection.execute(
                f"CREATE TABLE IF NOT EXISTS {self.table_name} AS "
                "SELECT * FROM incoming_target_forecasts WHERE FALSE"
            )
            connection.execute(
                f"DELETE FROM {self.table_name} AS target WHERE EXISTS "
                f"(SELECT 1 FROM incoming_target_forecasts AS source WHERE {predicate})"
            )
            connection.execute(
                f"INSERT INTO {self.table_name} SELECT * FROM incoming_target_forecasts"
            )
            row = connection.execute(f"SELECT COUNT(*) FROM {self.table_name}").fetchone()
        return int(row[0]) if row else 0

    def read(self, *, target: str | None = None) -> pd.DataFrame:
        with connect_duckdb(self.path, read_only=True) as connection:
            if target is None:
                frame = connection.execute(
                    f"SELECT * FROM {self.table_name} ORDER BY target, region, timestamp_utc"
                ).fetchdf()
            else:
                frame = connection.execute(
                    f"SELECT * FROM {self.table_name} WHERE target = ? "
                    "ORDER BY region, timestamp_utc",
                    [target],
                ).fetchdf()
        return validate_target_forecasts(frame) if not frame.empty else frame
