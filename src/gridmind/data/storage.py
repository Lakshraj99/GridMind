"""Filesystem and DuckDB persistence for GridMind data."""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote
from uuid import uuid4

import pandas as pd
import pyarrow as pa

from gridmind.data.duckdb_connection import connect_duckdb
from gridmind.data.processing import CANONICAL_COLUMNS
from gridmind.exceptions import StorageError
from gridmind.logging_config import redact_sensitive_value
from gridmind.time_utils import format_utc_timestamp

FORECAST_COLUMNS = [
    "region",
    "forecast_origin",
    "timestamp_utc",
    "forecast_step",
    "predicted_demand_mw",
    "model_name",
    "model_version",
    "run_id",
    "created_at_utc",
]


def write_raw_response(
    pages: list[dict[str, Any]],
    directory: Path,
    *,
    now: datetime | None = None,
    secrets: tuple[str, ...] = (),
) -> Path:
    """Write credential-redacted API pages to a timestamped JSON artifact."""
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%S%fZ")
    path = directory / f"eia_region_{timestamp}.json"
    path.write_text(json.dumps(redact_sensitive_value(pages, secrets), indent=2), encoding="utf-8")
    return path


def write_quarantine_parquet(
    frame: pd.DataFrame, directory: Path, *, now: datetime | None = None
) -> Path:
    """Persist excluded missing-demand rows as one timestamped audit artifact."""
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%S%fZ")
    path = directory / f"missing_demand_{timestamp}.parquet"
    try:
        frame.to_parquet(path, index=False, engine="pyarrow")
    except (OSError, ValueError, TypeError, pa.ArrowException) as exc:
        raise StorageError(f"Could not write missing-demand quarantine Parquet to {path}.") from exc
    return path


def write_json_report(report: dict[str, Any], path: Path) -> Path:
    """Write a JSON report, creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return path


def write_processed_parquet(frame: pd.DataFrame, directory: Path) -> Path:
    """Atomically replace an idempotently merged canonical Parquet dataset."""
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}-staging-", dir=directory.parent))
    try:
        incoming = _canonicalize_processed_frame(frame)
        existing = read_processed_parquet(directory)
        working = pd.concat([existing, incoming], ignore_index=True)
        working = working.drop_duplicates(["region", "timestamp_utc"], keep="last")
        working = _canonicalize_processed_frame(working)
        partitioned = working.copy()
        partitioned["year"] = partitioned["timestamp_utc"].dt.year.astype("int32")
        partitioned["month"] = partitioned["timestamp_utc"].dt.month.astype("int8")
        partitioned.to_parquet(
            staging,
            index=False,
            partition_cols=["region", "year", "month"],
            engine="pyarrow",
        )
        staged = read_processed_parquet(staging)
        try:
            pd.testing.assert_frame_equal(staged, working, check_dtype=True)
        except AssertionError as exc:
            raise StorageError(
                "Staged processed Parquet validation did not match the canonical input."
            ) from exc
        _replace_directory(staging, directory)
        return directory
    except StorageError:
        raise
    except (OSError, ValueError, TypeError, KeyError, pa.ArrowException) as exc:
        raise StorageError(
            f"Could not safely write processed Parquet data to {directory}; "
            "previously valid data was left unchanged."
        ) from exc
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def read_processed_parquet(directory: Path) -> pd.DataFrame:
    """Read only Parquet files and reconstruct canonical Hive partition columns."""
    directory = Path(directory)
    if not directory.exists():
        return empty_canonical_dataframe()
    try:
        parquet_files = sorted(path for path in directory.rglob("*.parquet") if path.is_file())
        if not parquet_files:
            return empty_canonical_dataframe()
        frames = [_read_processed_file(path, directory) for path in parquet_files]
        return _canonicalize_processed_frame(pd.concat(frames, ignore_index=True))
    except StorageError:
        raise
    except (OSError, ValueError, TypeError, KeyError, pa.ArrowException) as exc:
        raise StorageError(
            f"Could not read processed Parquet data from {directory}. "
            "Verify that every *.parquet file is a valid GridMind partition."
        ) from exc


def empty_canonical_dataframe() -> pd.DataFrame:
    """Return an empty canonical frame with stable production dtypes."""
    return pd.DataFrame(
        {
            "timestamp_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            "region": pd.Series(dtype="string"),
            "demand_mw": pd.Series(dtype="float64"),
            "forecast_demand_mw": pd.Series(dtype="float64"),
            "net_generation_mw": pd.Series(dtype="float64"),
            "total_interchange_mw": pd.Series(dtype="float64"),
            "ingestion_timestamp_utc": pd.Series(dtype="datetime64[ns, UTC]"),
        },
        columns=CANONICAL_COLUMNS,
    )


def _read_processed_file(path: Path, dataset_root: Path) -> pd.DataFrame:
    """Read one Parquet file and safely restore canonical Hive partitions."""
    frame: pd.DataFrame = pd.read_parquet(path)
    if frame.columns.duplicated().any():
        raise StorageError(f"Processed Parquet file has duplicate columns: {path.name}.")
    partition_values = _hive_partition_values(path, dataset_root)
    for partition_column in ("region", "year", "month"):
        partition_value = partition_values.get(partition_column)
        if partition_value is None or partition_column not in frame:
            continue
        stored_values = set(frame[partition_column].dropna().astype(str).unique())
        if stored_values and stored_values != {partition_value}:
            raise StorageError(
                f"Processed Parquet column '{partition_column}' conflicts with its "
                f"Hive partition: {path.name}."
            )
    region = partition_values.get("region")
    if region is not None:
        if "region" in frame:
            frame["region"] = frame["region"].fillna(region)
        else:
            frame["region"] = region
    return frame


def _hive_partition_values(path: Path, dataset_root: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for component in path.relative_to(dataset_root).parts[:-1]:
        key, separator, value = component.partition("=")
        if separator:
            values[key] = unquote(value)
    return values


def _canonicalize_processed_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(CANONICAL_COLUMNS).difference(frame.columns)
    if missing:
        raise StorageError(
            f"Processed Parquet data is missing canonical columns: {sorted(missing)}."
        )
    if frame.empty:
        return empty_canonical_dataframe()
    result = frame[CANONICAL_COLUMNS].copy()
    try:
        result["timestamp_utc"] = pd.to_datetime(result["timestamp_utc"], utc=True, errors="raise")
        result["ingestion_timestamp_utc"] = pd.to_datetime(
            result["ingestion_timestamp_utc"], utc=True, errors="raise"
        )
        result["region"] = result["region"].astype("string")
        for column in CANONICAL_COLUMNS[2:6]:
            result[column] = pd.to_numeric(result[column], errors="raise").astype("float64")
    except (ValueError, TypeError) as exc:
        raise StorageError("Processed Parquet data contains invalid canonical values.") from exc
    return result.sort_values(["region", "timestamp_utc"], ignore_index=True)


def _replace_directory(staging: Path, destination: Path) -> None:
    """Swap a validated staged dataset into place, restoring the old one on failure."""
    backup = destination.with_name(f".{destination.name}-backup-{uuid4().hex}")
    had_destination = destination.exists()
    if had_destination:
        destination.rename(backup)
    try:
        staging.rename(destination)
    except BaseException:
        if had_destination and backup.exists():
            backup.rename(destination)
        raise
    if backup.exists():
        if backup.is_dir():
            shutil.rmtree(backup, ignore_errors=True)
        else:
            backup.unlink(missing_ok=True)


class DuckDBStorage:
    """Idempotent analytical storage for canonical hourly grid data."""

    table_name = "hourly_grid_data"
    forecast_table_name = "demand_forecasts"

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def upsert(self, frame: pd.DataFrame) -> int:
        """Replace existing region/timestamp keys and return the resulting row count."""
        incoming = _canonicalize_processed_frame(frame)
        with connect_duckdb(self.path) as connection:
            connection.execute("BEGIN TRANSACTION")
            try:
                connection.register("incoming_grid_data", incoming)
                connection.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} AS
                    SELECT * FROM incoming_grid_data WHERE FALSE
                    """
                )
                connection.execute(
                    f"""
                    DELETE FROM {self.table_name} AS target
                    WHERE EXISTS (
                        SELECT 1 FROM incoming_grid_data AS source
                        WHERE source.region = target.region
                          AND source.timestamp_utc = target.timestamp_utc
                    )
                    """
                )
                connection.execute(
                    f"INSERT INTO {self.table_name} SELECT * FROM incoming_grid_data"
                )
                count_row = connection.execute(f"SELECT COUNT(*) FROM {self.table_name}").fetchone()
                if count_row is None:  # pragma: no cover
                    raise RuntimeError("DuckDB did not return a row count after upsert.")
                count = int(count_row[0])
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        return count

    def read_region(
        self, region: str, start_date: str | datetime, end_date: str | datetime
    ) -> pd.DataFrame:
        """Read one region and inclusive date range using bound SQL parameters."""
        with connect_duckdb(self.path, read_only=True) as connection:
            frame = connection.execute(
                f"""
                SELECT * FROM {self.table_name}
                WHERE region = ? AND timestamp_utc >= ? AND timestamp_utc <= ?
                ORDER BY timestamp_utc
                """,
                [region, start_date, end_date],
            ).fetchdf()
        if not frame.empty:
            frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
            frame["ingestion_timestamp_utc"] = pd.to_datetime(
                frame["ingestion_timestamp_utc"], utc=True
            )
            frame["region"] = frame["region"].astype("string")
        return frame

    def read_data(
        self,
        *,
        regions: list[str] | None = None,
        start_date: str | datetime = "1900-01-01",
        end_date: str | datetime = "2100-01-01",
    ) -> pd.DataFrame:
        """Read all or selected regions over a bound date interval."""
        parameters: list[str | datetime] = [start_date, end_date]
        region_clause = ""
        if regions:
            placeholders = ", ".join("?" for _ in regions)
            region_clause = f" AND region IN ({placeholders})"
            parameters.extend(regions)
        with connect_duckdb(self.path, read_only=True) as connection:
            frame = connection.execute(
                f"""
                SELECT * FROM {self.table_name}
                WHERE timestamp_utc >= ? AND timestamp_utc <= ? {region_clause}
                ORDER BY region, timestamp_utc
                """,
                parameters,
            ).fetchdf()
        if not frame.empty:
            frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
            frame["ingestion_timestamp_utc"] = pd.to_datetime(
                frame["ingestion_timestamp_utc"], utc=True
            )
            frame["region"] = frame["region"].astype("string")
        return frame

    def inspect(self) -> dict[str, Any]:
        """Return compact table statistics for CLI inspection."""
        with connect_duckdb(self.path, read_only=True) as connection:
            row = connection.execute(
                f"""
                SELECT COUNT(*), MIN(timestamp_utc), MAX(timestamp_utc),
                       COUNT(*) FILTER (WHERE demand_mw IS NULL)
                FROM {self.table_name}
                """
            ).fetchone()
            if row is None:  # pragma: no cover
                raise RuntimeError("DuckDB did not return table statistics.")
            regions = [
                item[0]
                for item in connection.execute(
                    f"SELECT DISTINCT region FROM {self.table_name} ORDER BY region"
                ).fetchall()
            ]
        return {
            "row_count": int(row[0]),
            "date_range": {
                "start": format_utc_timestamp(row[1]) if row[1] else None,
                "end": format_utc_timestamp(row[2]) if row[2] else None,
            },
            "missing_demand_count": int(row[3]),
            "regions": regions,
        }

    def upsert_forecasts(self, frame: pd.DataFrame) -> int:
        """Idempotently persist demand forecasts and return the table row count."""
        missing = set(FORECAST_COLUMNS).difference(frame.columns)
        if missing:
            raise ValueError(f"Forecast data is missing columns: {sorted(missing)}")
        incoming = frame[FORECAST_COLUMNS].copy()
        for column in ("forecast_origin", "timestamp_utc", "created_at_utc"):
            incoming[column] = pd.to_datetime(incoming[column], utc=True, errors="raise")
        with connect_duckdb(self.path) as connection:
            connection.execute("BEGIN TRANSACTION")
            try:
                connection.register("incoming_forecasts", incoming)
                connection.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.forecast_table_name} AS
                    SELECT * FROM incoming_forecasts WHERE FALSE
                    """
                )
                connection.execute(
                    f"""
                    DELETE FROM {self.forecast_table_name} AS target
                    WHERE EXISTS (
                        SELECT 1 FROM incoming_forecasts AS source
                        WHERE source.region = target.region
                          AND source.forecast_origin = target.forecast_origin
                          AND source.timestamp_utc = target.timestamp_utc
                          AND source.model_name = target.model_name
                          AND source.model_version = target.model_version
                    )
                    """
                )
                connection.execute(
                    f"INSERT INTO {self.forecast_table_name} SELECT * FROM incoming_forecasts"
                )
                row = connection.execute(
                    f"SELECT COUNT(*) FROM {self.forecast_table_name}"
                ).fetchone()
                if row is None:  # pragma: no cover
                    raise RuntimeError("DuckDB did not return a forecast row count.")
                count = int(row[0])
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        return count

    def read_forecasts(self, region: str | None = None) -> pd.DataFrame:
        """Read persisted forecasts, optionally filtering one region."""
        with connect_duckdb(self.path, read_only=True) as connection:
            if region is None:
                frame = connection.execute(
                    f"SELECT * FROM {self.forecast_table_name} "
                    "ORDER BY forecast_origin, region, timestamp_utc"
                ).fetchdf()
            else:
                frame = connection.execute(
                    f"SELECT * FROM {self.forecast_table_name} WHERE region = ? "
                    "ORDER BY forecast_origin, timestamp_utc",
                    [region],
                ).fetchdf()
        for column in ("forecast_origin", "timestamp_utc", "created_at_utc"):
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], utc=True)
        return frame
