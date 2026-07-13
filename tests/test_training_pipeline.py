"""End-to-end Milestone 2 artifacts and MLflow hierarchy tests."""

from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient

from gridmind.config import Settings
from gridmind.models.promotion import effective_registry_uri
from gridmind.pipelines.predict import load_prediction_bundle
from gridmind.pipelines.train import run_training_pipeline


def _single_region_training_frame(ml_hourly_frame: pd.DataFrame) -> pd.DataFrame:
    return ml_hourly_frame.loc[ml_hourly_frame["region"] == "PJM"].iloc[:380].copy()


def test_training_pipeline_writes_complete_comparison_artifacts(
    tmp_path: Path, ml_hourly_frame: pd.DataFrame
) -> None:
    settings = Settings(
        MLFLOW_ENABLED=False,
        MLFLOW_REGISTER_MODEL=False,
        MODEL_N_JOBS=1,
        SHAP_SAMPLE_SIZE=8,
        _env_file=None,
    )
    result = run_training_pipeline(
        settings,
        frame=_single_region_training_frame(ml_hourly_frame),
        model_names=["lightgbm"],
        horizon=2,
        validation_windows=1,
        step_size=2,
        mlflow_enabled=False,
        register_model=False,
        output_dir=tmp_path / "training",
    )
    assert result.selected_model == "lightgbm_global"
    assert len(result.leaderboard) == 5
    assert result.bundle_path.exists()
    for filename in (
        "leaderboard.csv",
        "leaderboard.json",
        "validation_predictions.parquet",
        "window_metrics.csv",
        "horizon_metrics.csv",
        "region_metrics.csv",
        "best_parameters.json",
        "feature_schema.json",
        "feature_build_report.json",
        "window_selection.json",
    ):
        assert (result.artifact_dir / filename).exists()


def test_training_pipeline_creates_parent_children_model_and_candidate(
    tmp_path: Path, ml_hourly_frame: pd.DataFrame
) -> None:
    settings = Settings(
        DATA_DIR=tmp_path / "data",
        MLFLOW_TRACKING_URI=f"sqlite:///{tmp_path / 'mlflow.db'}",
        MLFLOW_ARTIFACT_ROOT=tmp_path / "mlartifacts",
        ENABLE_MLFLOW=True,
        MLFLOW_REGISTER_MODEL=True,
        MLFLOW_MODEL_NAME="gridmind-pipeline-test",
        MLFLOW_EXPERIMENT_NAME="gridmind-pipeline-test",
        MODEL_N_JOBS=1,
        SHAP_SAMPLE_SIZE=5,
        _env_file=None,
    )
    result = run_training_pipeline(
        settings,
        frame=_single_region_training_frame(ml_hourly_frame),
        model_names=["lightgbm"],
        horizon=1,
        validation_windows=1,
        step_size=1,
        mlflow_enabled=True,
        register_model=True,
        output_dir=tmp_path / "training",
    )
    assert result.parent_run_id
    assert result.selected_run_id
    assert result.model_version == "1"
    assert result.promotion is not None
    uri = effective_registry_uri(
        settings.mlflow_tracking_uri, settings.data_dir / "mlflow_registry.db"
    )
    client = MlflowClient(tracking_uri=uri)
    candidate = client.get_model_version_by_alias(settings.mlflow_model_name, "candidate")
    assert str(candidate.version) == "1"
    loaded_bundle = load_prediction_bundle(
        settings,
        model_alias="candidate",
        model_version=None,
        run_id=None,
        bundle_path=None,
    )
    assert loaded_bundle.model.name == "lightgbm_global"
    mlflow.set_tracking_uri(uri)
    logged_model = mlflow.pyfunc.load_model(f"models:/{settings.mlflow_model_name}@candidate")
    assert logged_model.metadata.run_id == result.selected_run_id
    selected_artifacts = {item.path for item in client.list_artifacts(result.selected_run_id)}
    assert {"bundle", "explainability"}.issubset(selected_artifacts)
    parent_artifacts = {item.path for item in client.list_artifacts(result.parent_run_id)}
    assert "training_artifacts" in parent_artifacts
    experiment = client.get_experiment_by_name(settings.mlflow_experiment_name)
    assert experiment is not None
    assert experiment.artifact_location.startswith((tmp_path / "mlartifacts").resolve().as_uri())
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) >= 7
