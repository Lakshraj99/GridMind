"""Parquet and DuckDB persistence for renewable targets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gridmind.data.duckdb_connection import connect_duckdb
from gridmind.renewables.schemas import RENEWABLE_COLUMNS


def write_renewable_parquet(frame: pd.DataFrame, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "renewable_generation.parquet"
    combined = frame[RENEWABLE_COLUMNS].copy()
    if path.exists():
        combined = pd.concat([pd.read_parquet(path), combined], ignore_index=True)
    combined = combined.drop_duplicates(["region", "timestamp_utc"], keep="last").sort_values(
        ["region", "timestamp_utc"]
    )
    combined.to_parquet(path, index=False)
    return path


class RenewableStorage:
    table_name = "hourly_renewable_generation"

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def upsert(self, frame: pd.DataFrame) -> int:
        incoming = frame[RENEWABLE_COLUMNS].copy()
        for column in ("timestamp_utc", "ingestion_timestamp_utc"):
            incoming[column] = pd.to_datetime(incoming[column], utc=True, errors="raise")
        with connect_duckdb(self.path) as connection:
            connection.register("incoming_renewable", incoming)
            connection.execute(
                f"CREATE TABLE IF NOT EXISTS {self.table_name} AS "
                "SELECT * FROM incoming_renewable WHERE FALSE"
            )
            connection.execute(
                f"DELETE FROM {self.table_name} AS target WHERE EXISTS "
                "(SELECT 1 FROM incoming_renewable AS source WHERE source.region = target.region "
                "AND source.timestamp_utc = target.timestamp_utc)"
            )
            connection.execute(f"INSERT INTO {self.table_name} SELECT * FROM incoming_renewable")
            row = connection.execute(f"SELECT COUNT(*) FROM {self.table_name}").fetchone()
        return int(row[0]) if row else 0

    def read(self, region: str) -> pd.DataFrame:
        with connect_duckdb(self.path, read_only=True) as connection:
            frame = connection.execute(
                f"SELECT * FROM {self.table_name} WHERE region = ? ORDER BY timestamp_utc",
                [region],
            ).fetchdf()
        for column in ("timestamp_utc", "ingestion_timestamp_utc"):
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], utc=True)
        return frame


def create_net_load_view(path: Path | str) -> None:
    """Create a documented complete-case net-load view without filling missing inputs."""
    with connect_duckdb(path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables"
            ).fetchall()
        }
        if not {"hourly_grid_data", "hourly_renewable_generation"}.issubset(tables):
            return
        connection.execute(
            """
            CREATE OR REPLACE VIEW target_net_load AS
            SELECT grid.region, grid.timestamp_utc, grid.demand_mw,
                   renewable.total_renewable_generation_mw,
                   grid.demand_mw - renewable.total_renewable_generation_mw AS net_load_mw
            FROM hourly_grid_data AS grid
            INNER JOIN hourly_renewable_generation AS renewable
              ON grid.region = renewable.region
             AND grid.timestamp_utc = renewable.timestamp_utc
            WHERE grid.demand_mw IS NOT NULL
              AND renewable.total_renewable_generation_mw IS NOT NULL
            """
        )
