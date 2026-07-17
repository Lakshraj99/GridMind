"""Safe, read-only MLflow Registry metadata service."""

from __future__ import annotations

from typing import Any

import pandas as pd
from mlflow import MlflowClient

from gridmind.config import Settings
from gridmind.exceptions import ResourceNotFoundError, ServiceUnavailableError


class ModelService:
    def __init__(self, settings: Settings, *, client: MlflowClient | None = None) -> None:
        self.settings = settings
        self.client = client

    def _client(self) -> MlflowClient:
        if not self.settings.mlflow_enabled:
            raise ServiceUnavailableError("MLflow is disabled by configuration.")
        return self.client or MlflowClient(tracking_uri=self.settings.mlflow_tracking_uri)

    def list(self) -> list[dict[str, Any]]:
        client = self._client()
        result: list[dict[str, Any]] = []
        try:
            for model in client.search_registered_models():
                aliases = dict(getattr(model, "aliases", {}) or {})
                for version in client.search_model_versions(f"name='{model.name}'"):
                    tags = dict(getattr(version, "tags", {}) or {})
                    run_id = str(getattr(version, "run_id", "") or "")
                    metrics = dict(client.get_run(run_id).data.metrics) if run_id else {}
                    created = (
                        pd.Timestamp(getattr(version, "creation_timestamp", 0), unit="ms", tz="UTC")
                        .isoformat()
                        .replace("+00:00", "Z")
                    )
                    result.append(
                        {
                            "name": str(model.name),
                            "version": str(version.version),
                            "aliases": sorted(
                                alias
                                for alias, number in aliases.items()
                                if str(number) == str(version.version)
                            ),
                            "status": str(getattr(version, "status", "")),
                            "run_id": run_id,
                            "training_metrics": metrics,
                            "created_at_utc": created,
                            "target": tags.get("target"),
                            "region": tags.get("region"),
                        }
                    )
        except Exception as exc:
            raise ServiceUnavailableError("MLflow backend is unavailable.") from exc
        return result

    def get(self, name: str) -> dict[str, Any]:
        versions = [item for item in self.list() if item["name"] == name]
        if not versions:
            raise ResourceNotFoundError(f"Registered model '{name}' was not found.")
        return {"name": name, "versions": versions}

    def summary(self) -> dict[str, Any]:
        items = self.list()
        return {
            "registered_models": len({item["name"] for item in items}),
            "model_versions": len(items),
            "candidate_versions": sum("candidate" in item["aliases"] for item in items),
            "champion_versions": sum("champion" in item["aliases"] for item in items),
        }
