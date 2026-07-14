"""Idempotent UTC DuckDB persistence for anomaly events."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gridmind.anomalies.contracts import (
    ANOMALY_COLUMNS,
    empty_anomaly_frame,
    validate_anomaly_frame,
)
from gridmind.data.duckdb_connection import connect_duckdb


class AnomalyStorage:
    table_name = "anomaly_events"

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _initialize(self) -> None:
        with connect_duckdb(self.path) as connection:
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    anomaly_id VARCHAR PRIMARY KEY, region VARCHAR, target VARCHAR,
                    timestamp_utc TIMESTAMPTZ, detector_name VARCHAR, detector_version VARCHAR,
                    anomaly_type VARCHAR, anomaly_score DOUBLE, severity VARCHAR,
                    observed_value DOUBLE, expected_value DOUBLE, residual DOUBLE, threshold DOUBLE,
                    feature_summary VARCHAR, explanation VARCHAR, forecast_origin TIMESTAMPTZ,
                    model_name VARCHAR, model_version VARCHAR, run_id VARCHAR,
                    detected_at_utc TIMESTAMPTZ, metadata_json VARCHAR
                )
                """
            )
            connection.execute(
                f"CREATE INDEX IF NOT EXISTS anomaly_lookup_idx ON {self.table_name} "
                "(region, target, timestamp_utc)"
            )

    def upsert(self, frame: pd.DataFrame) -> int:
        valid = validate_anomaly_frame(frame)
        if valid.empty:
            return self.count()
        with connect_duckdb(self.path) as connection:
            connection.execute("BEGIN TRANSACTION")
            try:
                connection.register("incoming_anomalies", valid)
                connection.execute(
                    f"DELETE FROM {self.table_name} AS target WHERE EXISTS "
                    "(SELECT 1 FROM incoming_anomalies AS source "
                    "WHERE source.anomaly_id = target.anomaly_id)"
                )
                columns = ", ".join(ANOMALY_COLUMNS)
                connection.execute(
                    f"INSERT INTO {self.table_name} ({columns}) "
                    f"SELECT {columns} FROM incoming_anomalies"
                )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        return self.count()

    def read(
        self,
        *,
        region: str | None = None,
        target: str | None = None,
        severity: str | None = None,
        detector: str | None = None,
        start: object | None = None,
        end: object | None = None,
    ) -> pd.DataFrame:
        clauses: list[str] = []
        parameters: list[object] = []
        for column, value in (
            ("region", region),
            ("target", target),
            ("severity", severity),
            ("detector_name", detector),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                parameters.append(value)
        if start is not None:
            clauses.append("timestamp_utc >= ?")
            parameters.append(start)
        if end is not None:
            clauses.append("timestamp_utc <= ?")
            parameters.append(end)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with connect_duckdb(self.path, read_only=True) as connection:
            frame = connection.execute(
                f"SELECT * FROM {self.table_name} {where} "
                "ORDER BY timestamp_utc, region, target, detector_name",
                parameters,
            ).fetchdf()
        return validate_anomaly_frame(frame) if not frame.empty else empty_anomaly_frame()

    def count(self) -> int:
        with connect_duckdb(self.path, read_only=True) as connection:
            row = connection.execute(f"SELECT COUNT(*) FROM {self.table_name}").fetchone()
        return int(row[0]) if row else 0
