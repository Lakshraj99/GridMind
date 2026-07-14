"""UTC DuckDB storage for current alerts and immutable lifecycle history."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gridmind.alerts.contracts import ALERT_COLUMNS, HISTORY_COLUMNS, validate_alert_frame
from gridmind.data.duckdb_connection import connect_duckdb


class AlertStorage:
    alert_table = "grid_alerts"
    history_table = "alert_history"

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _initialize(self) -> None:
        with connect_duckdb(self.path) as connection:
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.alert_table} (
                    alert_id VARCHAR PRIMARY KEY, region VARCHAR, target VARCHAR,
                    anomaly_type VARCHAR, severity VARCHAR, status VARCHAR,
                    first_seen_utc TIMESTAMPTZ, last_seen_utc TIMESTAMPTZ,
                    occurrence_count BIGINT, latest_anomaly_id VARCHAR, title VARCHAR,
                    summary VARCHAR, acknowledged_at_utc TIMESTAMPTZ,
                    resolved_at_utc TIMESTAMPTZ, created_at_utc TIMESTAMPTZ,
                    updated_at_utc TIMESTAMPTZ, metadata_json VARCHAR
                )
                """
            )
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.history_table} (
                    history_id VARCHAR PRIMARY KEY, alert_id VARCHAR, status VARCHAR,
                    severity VARCHAR, changed_at_utc TIMESTAMPTZ, change_reason VARCHAR,
                    anomaly_id VARCHAR, metadata_json VARCHAR
                )
                """
            )
            connection.execute(
                f"CREATE INDEX IF NOT EXISTS alert_lookup_idx ON {self.alert_table} "
                "(region, target, status, severity)"
            )

    def upsert_alerts(self, frame: pd.DataFrame) -> int:
        valid = validate_alert_frame(frame)
        if valid.empty:
            return self.count()
        self._replace(self.alert_table, valid, ALERT_COLUMNS, "alert_id")
        return self.count()

    def append_history(self, frame: pd.DataFrame) -> int:
        if frame.empty:
            return len(self.read_history())
        valid = frame[HISTORY_COLUMNS].copy()
        valid["changed_at_utc"] = pd.to_datetime(valid["changed_at_utc"], utc=True)
        self._replace(self.history_table, valid, HISTORY_COLUMNS, "history_id")
        return len(self.read_history())

    def _replace(
        self, table: str, frame: pd.DataFrame, columns: list[str], identifier: str
    ) -> None:
        with connect_duckdb(self.path) as connection:
            connection.register("incoming_alert_records", frame)
            connection.execute(
                f"DELETE FROM {table} AS target WHERE EXISTS "
                f"(SELECT 1 FROM incoming_alert_records AS source "
                f"WHERE source.{identifier} = target.{identifier})"
            )
            names = ", ".join(columns)
            connection.execute(
                f"INSERT INTO {table} ({names}) SELECT {names} FROM incoming_alert_records"
            )

    def read_alerts(
        self,
        *,
        region: str | None = None,
        target: str | None = None,
        status: str | None = None,
        severity: str | None = None,
    ) -> pd.DataFrame:
        clauses: list[str] = []
        parameters: list[object] = []
        for column, value in (
            ("region", region),
            ("target", target),
            ("status", status),
            ("severity", severity),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                parameters.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with connect_duckdb(self.path, read_only=True) as connection:
            frame = connection.execute(
                f"SELECT * FROM {self.alert_table} {where} ORDER BY last_seen_utc DESC",
                parameters,
            ).fetchdf()
        return (
            validate_alert_frame(frame) if not frame.empty else pd.DataFrame(columns=ALERT_COLUMNS)
        )

    def read_history(self, alert_id: str | None = None) -> pd.DataFrame:
        with connect_duckdb(self.path, read_only=True) as connection:
            if alert_id is None:
                frame = connection.execute(
                    f"SELECT * FROM {self.history_table} ORDER BY changed_at_utc"
                ).fetchdf()
            else:
                frame = connection.execute(
                    f"SELECT * FROM {self.history_table} WHERE alert_id = ? "
                    "ORDER BY changed_at_utc",
                    [alert_id],
                ).fetchdf()
        if "changed_at_utc" in frame:
            frame["changed_at_utc"] = pd.to_datetime(frame["changed_at_utc"], utc=True)
        return frame

    def count(self) -> int:
        with connect_duckdb(self.path, read_only=True) as connection:
            row = connection.execute(f"SELECT COUNT(*) FROM {self.alert_table}").fetchone()
        return int(row[0]) if row else 0
