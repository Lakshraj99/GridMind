"""Shared pagination, serialization, SQL, and deterministic TTL caching helpers."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd

from gridmind.data.duckdb_connection import connect_duckdb
from gridmind.exceptions import ResourceNotFoundError
from gridmind.time_utils import to_utc_timestamp


@dataclass(frozen=True)
class Page:
    """One stable offset-paginated result."""

    items: list[dict[str, Any]]
    total: int
    limit: int
    offset: int

    def as_dict(self, *, filters: Mapping[str, object] | None = None) -> dict[str, Any]:
        return {
            "items": self.items,
            "pagination": {
                "limit": self.limit,
                "offset": self.offset,
                "returned": len(self.items),
                "total": self.total,
                "has_more": self.offset + len(self.items) < self.total,
            },
            "filters": {key: value for key, value in (filters or {}).items() if value is not None},
        }


class TTLCache:
    """Small process-local cache with an injectable clock for deterministic tests."""

    def __init__(self, ttl_seconds: float, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self._values: dict[str, tuple[float, object]] = {}
        self._lock = Lock()

    def get_or_create(self, key: str, factory: Callable[[], Any]) -> tuple[Any, bool]:
        now = self.clock()
        with self._lock:
            cached = self._values.get(key)
            if cached is not None and cached[0] > now:
                return cached[1], True
        value = factory()
        if self.ttl_seconds > 0:
            with self._lock:
                self._values[key] = (now + self.ttl_seconds, value)
        return value, False

    def clear(self) -> None:
        with self._lock:
            self._values.clear()


def frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DuckDB dataframe into strict JSON-compatible UTC records."""
    result = frame.copy()
    for column in result.columns:
        if column.endswith("_utc") or column == "forecast_origin":
            result[column] = pd.to_datetime(result[column], utc=True, errors="coerce")
    records: list[dict[str, Any]] = []
    for source in result.to_dict(orient="records"):
        record: dict[str, Any] = {}
        for key, value in source.items():
            if isinstance(value, pd.Timestamp):
                timestamp = (
                    value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
                )
                record[str(key)] = timestamp.isoformat().replace("+00:00", "Z")
            elif pd.isna(value) if not isinstance(value, (list, dict, tuple)) else False:
                record[str(key)] = None
            else:
                record[str(key)] = value.item() if hasattr(value, "item") else value
        records.append(record)
    return records


def decode_json_fields(record: dict[str, Any]) -> dict[str, Any]:
    for key in tuple(record):
        if key.endswith("_json") and isinstance(record[key], str):
            try:
                record[key.removesuffix("_json")] = json.loads(record.pop(key))
            except json.JSONDecodeError:
                record.pop(key)
    return record


class DuckDBReadService:
    """Short-lived, parameterized DuckDB query helper for application services."""

    def __init__(self, path: Path | str, *, metrics: Any | None = None) -> None:
        self.path = Path(path)
        self.metrics = metrics

    def query(self, sql: str, parameters: Sequence[object] = ()) -> pd.DataFrame:
        started = time.perf_counter()
        try:
            with connect_duckdb(self.path, read_only=True) as connection:
                return connection.execute(sql, list(parameters)).fetchdf()
        finally:
            if self.metrics is not None:
                self.metrics.query_latency.observe(time.perf_counter() - started)

    def table_exists(self, table: str) -> bool:
        if not self.path.exists():
            return False
        with connect_duckdb(self.path, read_only=True) as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [table]
            ).fetchone()
        return bool(row and row[0])

    def require_table(self, table: str) -> None:
        if not self.table_exists(table):
            raise ResourceNotFoundError(f"Required data table '{table}' was not found.")


def where_clause(filters: Sequence[tuple[str, object | None]]) -> tuple[str, list[object]]:
    clauses: list[str] = []
    parameters: list[object] = []
    for expression, value in filters:
        if value is not None:
            clauses.append(f"{expression} ?")
            parameters.append(
                to_utc_timestamp(value)
                if "timestamp" in expression or "forecast_origin" in expression
                else value
            )
    return (f" WHERE {' AND '.join(clauses)}" if clauses else "", parameters)
