"""Offline batch demand forecasting and idempotent persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from gridmind.config import Settings
from gridmind.continuity import detect_contiguous_segments
from gridmind.data.storage import FORECAST_COLUMNS, DuckDBStorage, write_json_report
from gridmind.exceptions import (
    InsufficientHistoryError,
    ModelSerializationError,
    PredictionValidationError,
)
from gridmind.models.promotion import effective_registry_uri
from gridmind.models.serialization import ModelBundle, load_model_bundle
from gridmind.time_utils import format_utc_timestamp


@dataclass(frozen=True)
class PredictionPipelineResult:
    """Validated batch predictions and their persisted locations."""

    predictions: pd.DataFrame
    parquet_path: Path
    duckdb_rows: int
    model_name: str
    model_version: str
    metadata_path: Path


def run_prediction_pipeline(
    settings: Settings,
    *,
    region: str,
    horizon: int | None = None,
    model_alias: str = "champion",
    model_version: str | None = None,
    run_id: str | None = None,
    bundle_path: Path | None = None,
    bundle: ModelBundle | None = None,
    output_dir: Path = Path("artifacts/predictions"),
) -> PredictionPipelineResult:
    """Load a model, forecast from DuckDB history, validate, and persist idempotently."""
    selected_horizon = horizon or settings.forecast_horizon
    if selected_horizon <= 0:
        raise ValueError("Prediction horizon must be positive.")
    loaded = bundle or load_prediction_bundle(
        settings,
        model_alias=model_alias,
        model_version=model_version,
        run_id=run_id,
        bundle_path=bundle_path,
    )
    supported_regions = loaded.metadata.get("regions")
    if supported_regions and region not in supported_regions:
        raise PredictionValidationError(
            f"Region {region} was not present during model training: {supported_regions}"
        )
    history = DuckDBStorage(settings.duckdb_path).read_region(region, "1900-01-01", "2100-01-01")
    required = loaded.model.specification.required_history
    continuity = detect_contiguous_segments(history)
    latest_segment = continuity.segments.sort_values("segment_end").iloc[-1]
    latest_segment_rows = int(latest_segment["row_count"])
    if latest_segment_rows < required:
        raise InsufficientHistoryError(
            f"Model requires {required} contiguous hourly observations for {region}; "
            f"the latest segment {latest_segment['region_segment_id']} contains "
            f"{latest_segment_rows}. Historical rows before its gap cannot be substituted."
        )
    predictions = loaded.model.predict(history, horizon=selected_horizon)
    predictions["forecast_step"] = (
        predictions.sort_values(["region", "timestamp_utc"])
        .groupby("region", observed=True)
        .cumcount()
        + 1
    )
    predictions = validate_batch_predictions(predictions)
    output_dir.mkdir(parents=True, exist_ok=True)
    origin = pd.Timestamp(predictions["forecast_origin"].iloc[0]).strftime("%Y%m%dT%H%M%SZ")
    parquet_path = output_dir / f"demand_forecast_{region}_{origin}.parquet"
    predictions.to_parquet(parquet_path, index=False)
    count = DuckDBStorage(settings.duckdb_path).upsert_forecasts(predictions)
    metadata_path = write_json_report(
        {
            "region": region,
            "forecast_origin": format_utc_timestamp(predictions["forecast_origin"].iloc[0]),
            "horizon": selected_horizon,
            "model_name": str(predictions["model_name"].iloc[0]),
            "model_version": str(predictions["model_version"].iloc[0]),
            "run_id": str(predictions["run_id"].iloc[0]),
            "row_count": len(predictions),
            "duckdb_rows": count,
        },
        output_dir / f"demand_forecast_{region}_{origin}_metadata.json",
    )
    return PredictionPipelineResult(
        predictions=predictions,
        parquet_path=parquet_path,
        duckdb_rows=count,
        model_name=str(predictions["model_name"].iloc[0]),
        model_version=str(predictions["model_version"].iloc[0]),
        metadata_path=metadata_path,
    )


def validate_batch_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Enforce forecast schema, UTC semantics, uniqueness, and finite nonnegative demand."""
    missing = set(FORECAST_COLUMNS).difference(frame.columns)
    if missing:
        raise PredictionValidationError(f"Prediction output is missing: {sorted(missing)}")
    result = frame[FORECAST_COLUMNS].copy()
    for column in ("timestamp_utc", "forecast_origin", "created_at_utc"):
        result[column] = pd.to_datetime(result[column], utc=True, errors="raise")
    values = result["predicted_demand_mw"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise PredictionValidationError("Predictions contain non-finite demand values.")
    if (values < 0).any():
        raise PredictionValidationError(
            "Predictions contain negative demand; GridMind does not silently clamp forecasts."
        )
    key = ["region", "forecast_origin", "timestamp_utc", "model_name", "model_version"]
    if result.duplicated(key).any():
        raise PredictionValidationError("Prediction output contains duplicate forecast keys.")
    return result.sort_values(["region", "timestamp_utc"], ignore_index=True)


def load_prediction_bundle(
    settings: Settings,
    *,
    model_alias: str,
    model_version: str | None,
    run_id: str | None,
    bundle_path: Path | None,
) -> ModelBundle:
    """Load by local path, run ID, registry version, or registry alias."""
    if bundle_path is not None:
        return load_model_bundle(bundle_path)
    tracking_uri = effective_registry_uri(
        settings.mlflow_tracking_uri, settings.data_dir / "mlflow_registry.db"
    )
    client = MlflowClient(tracking_uri=tracking_uri)
    selected_run_id = run_id
    selected_version = model_version
    if selected_run_id is None:
        if selected_version is None:
            version_info = client.get_model_version_by_alias(
                settings.mlflow_model_name, model_alias
            )
        else:
            version_info = client.get_model_version(settings.mlflow_model_name, selected_version)
        selected_run_id = version_info.run_id
        selected_version = str(version_info.version)
    if not selected_run_id:
        raise ModelSerializationError("Selected registry entry does not contain a run ID.")
    mlflow.set_tracking_uri(tracking_uri)
    downloaded = mlflow.artifacts.download_artifacts(
        run_id=selected_run_id, artifact_path="bundle/model_bundle.joblib"
    )
    loaded = load_model_bundle(Path(downloaded))
    loaded.model.run_id = selected_run_id
    if selected_version:
        loaded.model.model_version = selected_version
    return loaded
