"""Reliable MLflow tracking and artifact-store initialization."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from gridmind.config import Settings


@dataclass(frozen=True)
class MlflowSetup:
    """Resolved tracking and experiment details for one pipeline."""

    tracking_uri: str
    experiment_id: str
    artifact_location: str


def initialize_mlflow(
    settings: Settings,
    experiment_name: str,
    *,
    tracking_uri: str | None = None,
) -> MlflowSetup:
    """Select tracking without consulting legacy stores and ensure an experiment exists."""
    selected_uri = tracking_uri or settings.mlflow_tracking_uri
    _prepare_sqlite_parent(selected_uri)
    mlflow.set_tracking_uri(selected_uri)
    client = MlflowClient(tracking_uri=selected_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        artifact_location = _experiment_artifact_location(
            settings.mlflow_artifact_root, experiment_name, selected_uri
        )
        if artifact_location:
            experiment_id = client.create_experiment(
                experiment_name,
                artifact_location=artifact_location,
            )
        else:
            experiment_id = client.create_experiment(experiment_name)
        experiment = client.get_experiment(experiment_id)
    mlflow.set_experiment(experiment_name)
    return MlflowSetup(
        tracking_uri=selected_uri,
        experiment_id=str(experiment.experiment_id),
        artifact_location=str(experiment.artifact_location),
    )


def _experiment_artifact_location(root: Path, experiment_name: str, tracking_uri: str) -> str:
    if not tracking_uri.startswith("sqlite:"):
        return ""
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", experiment_name).strip("-") or "experiment"
    directory = (root / safe_name).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory.as_uri()


def _prepare_sqlite_parent(tracking_uri: str) -> None:
    if not tracking_uri.startswith("sqlite:///"):
        return
    database = Path(tracking_uri.removeprefix("sqlite:///"))
    database.parent.mkdir(parents=True, exist_ok=True)
