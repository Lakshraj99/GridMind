"""Offline dependency readiness checks kept outside HTTP routes."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import ClassVar

from mlflow import MlflowClient

from gridmind.config import Settings
from gridmind.data.duckdb_connection import connect_duckdb


class HealthService:
    required_tables: ClassVar[set[str]] = {"target_forecasts"}

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def readiness(self) -> dict[str, object]:
        components: dict[str, dict[str, object]] = {}
        try:
            if not self.settings.duckdb_path.exists():
                raise FileNotFoundError("DuckDB database does not exist.")
            with connect_duckdb(self.settings.duckdb_path, read_only=True) as connection:
                rows = connection.execute(
                    "SELECT table_name FROM information_schema.tables"
                ).fetchall()
            missing = sorted(self.required_tables - {str(row[0]) for row in rows})
            components["duckdb"] = {"ready": not missing, "missing_tables": missing}
        except Exception:
            components["duckdb"] = {
                "ready": False,
                "message": "DuckDB database is unavailable.",
            }
        try:
            artifact_dir = Path(self.settings.data_quality_dir).resolve()
            artifact_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=artifact_dir):
                pass
            components["artifacts"] = {"ready": True}
        except OSError:
            components["artifacts"] = {
                "ready": False,
                "message": "Artifact directory is not writable.",
            }
        if not self.settings.mlflow_enabled:
            components["mlflow"] = {"ready": True, "enabled": False}
        else:
            try:
                MlflowClient(tracking_uri=self.settings.mlflow_tracking_uri).search_experiments(
                    max_results=1
                )
                components["mlflow"] = {"ready": True, "enabled": True}
            except Exception:
                components["mlflow"] = {
                    "ready": False,
                    "enabled": True,
                    "message": "MLflow backend is unavailable.",
                }
        overall = all(bool(component["ready"]) for component in components.values())
        return {"status": "ready" if overall else "not_ready", "components": components}
