"""Offline SQLite MLflow setup, hierarchy, model, and alias smoke test."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow import MlflowClient

from gridmind.config import Settings
from gridmind.mlflow_config import initialize_mlflow


class _IdentityModel(mlflow.pyfunc.PythonModel):
    def predict(
        self, context: Any, model_input: pd.DataFrame, params: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        del context, params
        return model_input.copy()


def test_sqlite_mlflow_smoke_ignores_malformed_file_store(tmp_path: Path, monkeypatch: Any) -> None:
    malformed = tmp_path / "mlruns" / "1"
    malformed.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    settings = Settings(
        MLFLOW_TRACKING_URI=tracking_uri,
        MLFLOW_ARTIFACT_ROOT=tmp_path / "mlartifacts",
        _env_file=None,
    )
    setup = initialize_mlflow(settings, "sqlite-smoke")

    with (
        mlflow.start_run(run_name="parent") as parent,
        mlflow.start_run(run_name="child", nested=True) as child,
    ):
        model_info = mlflow.pyfunc.log_model(artifact_path="model", python_model=_IdentityModel())

    client = MlflowClient(tracking_uri=tracking_uri)
    runs = client.search_runs([setup.experiment_id])
    assert len(runs) == 2
    child_run = client.get_run(child.info.run_id)
    assert child_run.data.tags["mlflow.parentRunId"] == parent.info.run_id
    assert setup.artifact_location.startswith((tmp_path / "mlartifacts").resolve().as_uri())
    assert not (malformed / "meta.yaml").exists()

    model_name = "gridmind-sqlite-smoke"
    client.create_registered_model(model_name)
    version = client.create_model_version(
        name=model_name,
        source=model_info.model_uri,
        run_id=child.info.run_id,
    )
    client.set_registered_model_alias(model_name, "candidate", str(version.version))
    client.set_registered_model_alias(model_name, "champion", str(version.version))
    assert str(client.get_model_version_by_alias(model_name, "candidate").version) == "1"
    assert str(client.get_model_version_by_alias(model_name, "champion").version) == "1"
    loaded = mlflow.pyfunc.load_model(f"models:/{model_name}@champion")
    predicted = loaded.predict(pd.DataFrame({"value": [1.0]}))
    assert predicted.loc[0, "value"] == 1.0
