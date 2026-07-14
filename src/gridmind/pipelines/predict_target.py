"""Registry/local target prediction and shared forecast persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient

from gridmind.config import Settings
from gridmind.data.target_storage import TargetForecastStorage, validate_target_forecasts
from gridmind.features.weather import build_weather_features
from gridmind.models.serialization import ModelBundle, load_model_bundle
from gridmind.models.target_factory import TargetForecaster
from gridmind.pipelines.train_target import _load_target_frame
from gridmind.renewables.targets import WeatherMode, get_target_definition
from gridmind.weather.storage import WeatherStorage


@dataclass(frozen=True)
class TargetPredictionResult:
    forecasts: pd.DataFrame
    parquet_path: Path
    duckdb_rows: int


def run_target_prediction(
    settings: Settings,
    *,
    target: str,
    region: str,
    horizon: int = 24,
    model_alias: str = "champion",
    weather_mode: WeatherMode = "realistic_forecast",
    bundle_path: Path | None = None,
    bundle: ModelBundle | None = None,
    output_dir: Path = Path("artifacts/target_predictions"),
) -> TargetPredictionResult:
    loaded = bundle or (
        load_model_bundle(bundle_path)
        if bundle_path is not None
        else _load_registry_bundle(settings, target, model_alias)
    )
    if not isinstance(loaded.model, TargetForecaster) or loaded.model.target != target:
        raise ValueError(f"Selected bundle does not forecast target {target}.")
    history = _load_target_frame(settings, target, region)
    forecast_weather = WeatherStorage(settings.duckdb_path).read_regions(
        region, data_type="forecast"
    )
    if forecast_weather.empty:
        raise ValueError("Target prediction requires provider forecast weather; none is stored.")
    weather_features = build_weather_features(
        forecast_weather,
        mode=weather_mode,
        lags=settings.weather_lags,
        rolling_windows=settings.weather_rolling_windows,
    ).frame
    prediction_input = history.merge(
        weather_features,
        on=["region", "timestamp_utc"],
        how="outer",
        suffixes=("", "_weather"),
    )
    forecasts = validate_target_forecasts(loaded.model.predict(prediction_input, horizon=horizon))
    output_dir.mkdir(parents=True, exist_ok=True)
    origin = forecasts["forecast_origin"].iloc[0].strftime("%Y%m%dT%H%M%SZ")
    path = output_dir / f"{target}_{region}_{origin}.parquet"
    forecasts.to_parquet(path, index=False)
    count = TargetForecastStorage(settings.duckdb_path).upsert(forecasts)
    return TargetPredictionResult(forecasts, path, count)


def _load_registry_bundle(settings: Settings, target: str, alias: str) -> ModelBundle:
    definition = get_target_definition(target)
    model_name = str(getattr(settings, definition.registry_setting))
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    version = client.get_model_version_by_alias(model_name, alias)
    run_id = version.run_id or ""
    if not run_id:
        raise ValueError("Selected target model version has no MLflow run ID.")
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="target_training_artifacts/bundle/model_bundle.joblib",
    )
    loaded = load_model_bundle(Path(path))
    loaded.model.model_version = str(version.version)
    loaded.model.run_id = run_id
    return loaded
