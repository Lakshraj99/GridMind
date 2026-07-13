"""End-to-end offline pipeline tests."""

from pathlib import Path
from typing import Any

import httpx
import mlflow
import pandas as pd

from gridmind.config import Settings
from gridmind.data.eia_client import EIAClient
from gridmind.data.storage import DuckDBStorage
from gridmind.forecasting.baselines import LastValueForecaster
from gridmind.pipelines.baseline import run_baseline_pipeline
from gridmind.pipelines.ingest import run_ingestion


def test_fixture_ingestion_pipeline(tmp_path: Path, eia_payload: dict[str, Any]) -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=eia_payload))
    client = EIAClient("test", client=httpx.Client(transport=transport))
    settings = Settings(
        EIA_API_KEY=None,
        DATA_DIR=tmp_path / "data",
        DATA_QUALITY_DIR=tmp_path / "artifacts" / "data_quality",
        DUCKDB_PATH=tmp_path / "grid.duckdb",
        MLFLOW_ENABLED=False,
        _env_file=None,
    )
    result = run_ingestion(
        settings,
        region="PJM",
        start_date="2024-01-01",
        end_date="2024-01-02",
        client=client,
    )
    assert result.rows == 2
    assert result.quality_report_path.exists()
    assert result.quality_report_path.parent == settings.data_quality_dir
    assert not list((settings.data_dir / "processed").rglob("*.json"))
    assert (
        len(DuckDBStorage(settings.duckdb_path).read_region("PJM", "2024-01-01", "2025-01-01")) == 2
    )


def test_end_to_end_baseline_without_mlflow(tmp_path: Path, hourly_frame: pd.DataFrame) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    settings = Settings(
        MLFLOW_ENABLED=False,
        MLFLOW_TRACKING_URI=tracking_uri,
        MLFLOW_ARTIFACT_ROOT=tmp_path / "mlartifacts",
        _env_file=None,
    )
    result = run_baseline_pipeline(
        hourly_frame,
        settings,
        horizon=24,
        windows=2,
        models=[LastValueForecaster()],
        artifact_dir=tmp_path / "artifacts",
    )
    assert result.leaderboard["model_name"].tolist() == ["last_value"]
    assert result.predictions_path.exists()
    assert result.metrics_path.exists()
    assert not (tmp_path / "mlflow.db").exists()
    assert not (tmp_path / "mlartifacts").exists()


def test_baseline_logs_local_mlflow_run(tmp_path: Path, hourly_frame: pd.DataFrame) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    settings = Settings(
        MLFLOW_ENABLED=True,
        MLFLOW_TRACKING_URI=tracking_uri,
        MLFLOW_ARTIFACT_ROOT=tmp_path / "mlartifacts",
        _env_file=None,
    )
    run_baseline_pipeline(
        hourly_frame,
        settings,
        horizon=24,
        windows=1,
        models=[LastValueForecaster()],
        artifact_dir=tmp_path / "artifacts",
    )
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("gridmind-baselines")
    assert experiment is not None
    assert experiment.artifact_location.startswith((tmp_path / "mlartifacts").resolve().as_uri())
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs[0].data.params["baseline_model"] == "last_value"
    artifact_paths = {item.path for item in client.list_artifacts(runs[0].info.run_id)}
    assert artifact_paths == {"configuration", "data", "predictions"}
