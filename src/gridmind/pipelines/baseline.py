"""Rolling baseline evaluation, artifacts, and optional MLflow tracking."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from gridmind.config import Settings
from gridmind.data.processing import generate_quality_report
from gridmind.data.storage import write_json_report
from gridmind.forecasting.baselines import BaselineForecaster, all_baseline_models
from gridmind.forecasting.metrics import evaluate_predictions
from gridmind.forecasting.validation import rolling_origin_evaluate
from gridmind.mlflow_config import initialize_mlflow
from gridmind.time_utils import format_utc_timestamp


@dataclass(frozen=True)
class BaselinePipelineResult:
    """Leaderboard and artifact paths produced by baseline evaluation."""

    leaderboard: pd.DataFrame
    predictions_path: Path
    metrics_path: Path
    quality_report_path: Path
    configuration_path: Path


def run_baseline_pipeline(
    frame: pd.DataFrame,
    settings: Settings,
    *,
    horizon: int = 24,
    windows: int = 3,
    step_size: int = 24,
    models: list[BaselineForecaster] | None = None,
    artifact_dir: Path = Path("artifacts"),
    mlflow_enabled: bool | None = None,
) -> BaselinePipelineResult:
    """Evaluate all baselines and persist reproducible local artifacts."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    selected_models = models or all_baseline_models()
    region = str(frame["region"].iloc[0])
    quality = generate_quality_report(frame)
    quality_path = write_json_report(quality, artifact_dir / "data_quality_report.json")
    config_snapshot = {
        "region": region,
        "start_date": format_utc_timestamp(frame["timestamp_utc"].min()),
        "end_date": format_utc_timestamp(frame["timestamp_utc"].max()),
        "forecast_horizon": horizon,
        "validation_windows": windows,
        "step_size": step_size,
        "mlflow_tracking_uri": settings.mlflow_tracking_uri,
    }
    config_path = write_json_report(config_snapshot, artifact_dir / "configuration.json")

    prediction_frames: list[pd.DataFrame] = []
    metric_report: dict[str, Any] = {}
    leaderboard_rows: list[dict[str, float | str]] = []
    use_mlflow = settings.mlflow_enabled if mlflow_enabled is None else mlflow_enabled
    if use_mlflow:
        initialize_mlflow(settings, "gridmind-baselines")

    for model in selected_models:
        predictions = rolling_origin_evaluate(
            frame,
            model,
            horizon=horizon,
            windows=windows,
            step_size=step_size,
        )
        metrics = evaluate_predictions(predictions)
        prediction_frames.append(predictions)
        metric_report[model.name] = metrics
        leaderboard_rows.append({"model_name": model.name, **metrics["overall"]})
        if use_mlflow:
            _log_mlflow_run(
                model=model,
                predictions=predictions,
                metrics=metrics["overall"],
                configuration=config_snapshot,
                quality_path=quality_path,
                config_path=config_path,
                artifact_dir=artifact_dir,
            )

    combined = pd.concat(prediction_frames, ignore_index=True)
    predictions_path = artifact_dir / "validation_predictions.parquet"
    combined.to_parquet(predictions_path, index=False)
    metrics_path = write_json_report(
        _json_safe(metric_report), artifact_dir / "baseline_metrics.json"
    )
    leaderboard = pd.DataFrame(leaderboard_rows).sort_values("mae", ignore_index=True)
    return BaselinePipelineResult(
        leaderboard=leaderboard,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        quality_report_path=quality_path,
        configuration_path=config_path,
    )


def _log_mlflow_run(
    *,
    model: BaselineForecaster,
    predictions: pd.DataFrame,
    metrics: dict[str, float],
    configuration: dict[str, Any],
    quality_path: Path,
    config_path: Path,
    artifact_dir: Path,
) -> None:
    prediction_path = artifact_dir / f"{model.name}_predictions.parquet"
    predictions.to_parquet(prediction_path, index=False)
    with mlflow.start_run(run_name=model.name):
        mlflow.log_params({**configuration, "baseline_model": model.name})
        mlflow.log_metrics({key: value for key, value in metrics.items() if math.isfinite(value)})
        mlflow.log_artifact(str(quality_path), artifact_path="data")
        mlflow.log_artifact(str(prediction_path), artifact_path="predictions")
        mlflow.log_artifact(str(config_path), artifact_path="configuration")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
