"""Idempotent Parquet and DuckDB weather persistence."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gridmind.data.duckdb_connection import connect_duckdb
from gridmind.weather.schemas import REGION_WEATHER_COLUMNS, WEATHER_COLUMNS


def write_weather_parquet(
    frame: pd.DataFrame, directory: Path, *, aggregated: bool = False
) -> Path:
    """Persist only weather Parquet records under their own dataset directory."""
    columns = REGION_WEATHER_COLUMNS if aggregated else WEATHER_COLUMNS
    directory.mkdir(parents=True, exist_ok=True)
    working = (
        frame[columns]
        .copy()
        .drop_duplicates(
            ["region", "timestamp_utc", "weather_data_type"]
            if aggregated
            else ["region", "location_name", "timestamp_utc", "weather_data_type"],
            keep="last",
        )
    )
    path = directory / "weather.parquet"
    if path.exists():
        prior = pd.read_parquet(path)
        working = pd.concat([prior, working], ignore_index=True).drop_duplicates(
            ["region", "timestamp_utc", "weather_data_type"]
            if aggregated
            else ["region", "location_name", "timestamp_utc", "weather_data_type"],
            keep="last",
        )
    working.to_parquet(path, index=False)
    return path


class WeatherStorage:
    """DuckDB storage for location and regional hourly weather."""

    location_table = "hourly_location_weather"
    region_table = "hourly_region_weather"

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def upsert_locations(self, frame: pd.DataFrame) -> int:
        return self._upsert(
            frame[WEATHER_COLUMNS],
            self.location_table,
            ["region", "location_name", "timestamp_utc", "weather_data_type"],
        )

    def upsert_regions(self, frame: pd.DataFrame) -> int:
        return self._upsert(
            frame[REGION_WEATHER_COLUMNS],
            self.region_table,
            ["region", "timestamp_utc", "weather_data_type"],
        )

    def read_regions(self, region: str, *, data_type: str | None = None) -> pd.DataFrame:
        with connect_duckdb(self.path, read_only=True) as connection:
            clause = "region = ?"
            params: list[str] = [region]
            if data_type is not None:
                clause += " AND weather_data_type = ?"
                params.append(data_type)
            result = connection.execute(
                f"SELECT * FROM {self.region_table} WHERE {clause} ORDER BY timestamp_utc",
                params,
            ).fetchdf()
        for column in ("timestamp_utc", "ingestion_timestamp_utc"):
            if column in result:
                result[column] = pd.to_datetime(result[column], utc=True)
        return result

    def _upsert(self, frame: pd.DataFrame, table: str, keys: list[str]) -> int:
        predicate = " AND ".join(f"source.{key} = target.{key}" for key in keys)
        incoming = frame.copy()
        for column in ("timestamp_utc", "ingestion_timestamp_utc"):
            if column in incoming:
                incoming[column] = pd.to_datetime(incoming[column], utc=True, errors="raise")
        with connect_duckdb(self.path) as connection:
            connection.execute("BEGIN TRANSACTION")
            try:
                connection.register("incoming_weather", incoming)
                connection.execute(
                    f"CREATE TABLE IF NOT EXISTS {table} AS "
                    "SELECT * FROM incoming_weather WHERE FALSE"
                )
                connection.execute(
                    f"DELETE FROM {table} AS target WHERE EXISTS "
                    f"(SELECT 1 FROM incoming_weather AS source WHERE {predicate})"
                )
                connection.execute(f"INSERT INTO {table} SELECT * FROM incoming_weather")
                row = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        return int(row[0]) if row else 0
