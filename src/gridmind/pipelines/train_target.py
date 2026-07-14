"""Target-specific weather-aware training, evaluation, tracking, and registration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow import MlflowClient

from gridmind.config import Settings
from gridmind.continuity import select_gap_aware_windows
from gridmind.data.storage import DuckDBStorage, write_json_report
from gridmind.explainability.shap_analysis import generate_shap_artifacts
from gridmind.features.weather import build_weather_features, simulate_forecast_weather
from gridmind.forecasting.baselines import LastValueForecaster, SeasonalNaiveForecaster
from gridmind.forecasting.metrics import calculate_metrics
from gridmind.mlflow_config import initialize_mlflow
from gridmind.models.promotion import apply_promotion_gate, create_model_version
from gridmind.models.serialization import load_model_bundle, log_mlflow_model, save_model_bundle
from gridmind.models.target_factory import TargetForecaster, create_target_model
from gridmind.renewables.storage import RenewableStorage
from gridmind.renewables.targets import WeatherMode, compute_net_load, get_target_definition
from gridmind.training.evaluator import evaluate_baseline
from gridmind.training.multi_target import (
    TargetEvaluationResult,
    evaluate_target_model,
    tune_target_model,
)
from gridmind.weather.storage import WeatherStorage


@dataclass(frozen=True)
class TargetTrainingResult:
    target: str
    leaderboard: pd.DataFrame
    artifact_dir: Path
    bundle_path: Path
    selected_model: str
    model_version: str | None
    candidate_assigned: bool
    champion_promoted: bool


def run_target_training(
    settings: Settings,
    *,
    target: str,
    region: str,
    weather_mode: WeatherMode = "realistic_forecast",
    model_names: list[str] | None = None,
    horizon: int = 24,
    validation_windows: int = 5,
    step_size: int = 24,
    mlflow_enabled: bool = True,
    register_model: bool = True,
    tune: bool = False,
    trials: int = 10,
    frame: pd.DataFrame | None = None,
    weather: pd.DataFrame | None = None,
    output_dir: Path | None = None,
) -> TargetTrainingResult:
    definition = get_target_definition(target)
    source = frame if frame is not None else _load_target_frame(settings, target, region)
    weather_data = weather if weather is not None else _load_weather(settings, region, weather_mode)
    weather_result = build_weather_features(
        weather_data,
        mode=weather_mode,
        lags=settings.weather_lags,
        rolling_windows=settings.weather_rolling_windows,
    )
    joined = source.merge(
        weather_result.frame[["region", "timestamp_utc", *weather_result.feature_names]],
        on=["region", "timestamp_utc"],
        how="inner",
    )
    if joined.empty:
        raise ValueError("No target/weather timestamps overlap after feature validation.")
    models = model_names or ["lightgbm", "catboost"]
    factory = partial(
        create_target_model,
        models[0],
        target,
        weather_features=weather_result.feature_names,
        weather_mode=weather_mode,
        random_seed=settings.model_random_seed,
        n_jobs=settings.model_n_jobs,
    )
    required_history = factory().specification.required_history
    selection = select_gap_aware_windows(
        joined.rename(columns={target: "demand_mw"}),
        horizon=horizon,
        windows=validation_windows,
        step_size=step_size,
        required_history=required_history,
    )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifacts = output_dir or Path("artifacts/training_targets") / target / stamp
    artifacts.mkdir(parents=True, exist_ok=True)
    write_json_report(weather_result.report, artifacts / "weather_coverage_report.json")
    write_json_report(selection.to_dict(), artifacts / "window_selection.json")

    rows: list[dict[str, Any]] = []
    results: dict[str, TargetEvaluationResult] = {}
    best_parameters: dict[str, dict[str, Any]] = {name: {} for name in models}
    if tune:
        earliest_final = min(window.validation_timestamps[0] for window in selection.windows)
        tuning_history = joined.loc[joined["timestamp_utc"] < earliest_final].copy()
        for model_name in models:

            def tuned_factory(params: dict[str, Any], name: str = model_name) -> TargetForecaster:
                return create_target_model(
                    name,
                    target,
                    weather_features=weather_result.feature_names,
                    weather_mode=weather_mode,
                    random_seed=settings.model_random_seed,
                    n_jobs=settings.model_n_jobs,
                    params=params,
                )

            tuned = tune_target_model(
                tuning_history,
                target,
                model_name,
                tuned_factory,
                horizon=horizon,
                windows=settings.tuning_windows,
                step_size=step_size,
                required_history=required_history,
                trials=trials,
                random_seed=settings.model_random_seed,
                output_dir=artifacts / "tuning" / model_name,
            )
            if (
                max(window.validation_timestamps[-1] for window in tuned.selection.windows)
                >= earliest_final
            ):
                raise AssertionError("Target tuning windows overlap final evaluation windows.")
            best_parameters[model_name] = tuned.best_params
    baseline_factories: dict[str, Any] = {"last_value": LastValueForecaster}
    if target != "net_load_mw":
        baseline_factories["seasonal_naive_168h"] = lambda: SeasonalNaiveForecaster(168)
    baseline_factories["seasonal_naive_24h"] = lambda: SeasonalNaiveForecaster(24)
    baseline_frame = joined[["region", "timestamp_utc", target]].rename(
        columns={target: "demand_mw"}
    )
    for name, baseline_factory in baseline_factories.items():
        evaluated = evaluate_baseline(
            baseline_frame,
            baseline_factory,
            horizon=horizon,
            windows=validation_windows,
            step_size=step_size,
            window_selection=selection,
        )
        rows.append({"model_name": name, **evaluated.overall_metrics})
    if target == "net_load_mw":
        component_metrics = _evaluate_component_net_load(joined, selection, horizon)
        rows.append({"model_name": "component_demand_solar_wind", **component_metrics})

    for model_name in models:
        model_factory = partial(
            create_target_model,
            model_name,
            target,
            weather_features=weather_result.feature_names,
            weather_mode=weather_mode,
            random_seed=settings.model_random_seed,
            n_jobs=settings.model_n_jobs,
            params=best_parameters[model_name],
        )
        result = evaluate_target_model(
            joined,
            target,
            model_factory,
            horizon=horizon,
            windows=validation_windows,
            step_size=step_size,
            selection=selection,
        )
        name = f"{model_name}_{target}"
        results[name] = result
        evaluation_dir = artifacts / "evaluation" / name
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        result.predictions.to_parquet(
            evaluation_dir / "validation_predictions.parquet", index=False
        )
        result.window_metrics.to_csv(evaluation_dir / "window_metrics.csv", index=False)
        result.horizon_metrics.to_csv(evaluation_dir / "horizon_metrics.csv", index=False)
        result.region_metrics.to_csv(evaluation_dir / "region_metrics.csv", index=False)
        write_json_report(
            {
                key: value if math.isfinite(value) else None
                for key, value in result.overall_metrics.items()
            },
            evaluation_dir / "overall_metrics.json",
        )
        write_json_report(
            {
                key: value if math.isfinite(value) else None
                for key, value in result.target_metrics.items()
            },
            evaluation_dir / "target_metrics.json",
        )
        rows.append({"model_name": name, **result.overall_metrics, **result.target_metrics})
    leaderboard = pd.DataFrame(rows).sort_values(
        [settings.primary_selection_metric, "mae"], ignore_index=True
    )
    leaderboard.insert(0, "rank", range(1, len(leaderboard) + 1))
    leaderboard.to_csv(artifacts / "leaderboard.csv", index=False)
    leaderboard.to_json(artifacts / "leaderboard.json", orient="records", indent=2)
    factory().specification.save(artifacts / "feature_schema.json")
    best_name = str(leaderboard.loc[leaderboard["model_name"].isin(results), "model_name"].iloc[0])
    best_kind = best_name.split("_", 1)[0]
    final_model = create_target_model(
        best_kind,
        target,
        weather_features=weather_result.feature_names,
        weather_mode=weather_mode,
        random_seed=settings.model_random_seed,
        n_jobs=settings.model_n_jobs,
        params=best_parameters[best_kind],
    ).fit(joined)
    bundle = save_model_bundle(
        final_model,
        artifacts / "bundle" / "model_bundle.joblib",
        metadata={"target": target, "weather_mode": weather_mode, "region": region},
    )
    load_model_bundle(bundle)
    try:
        generate_shap_artifacts(
            final_model,
            joined,
            artifacts / "explainability",
            sample_size=settings.shap_sample_size,
            random_seed=settings.model_random_seed,
        )
    except Exception as exc:
        (artifacts / "explainability").mkdir(parents=True, exist_ok=True)
        (artifacts / "explainability" / "shap_failure.txt").write_text(str(exc), encoding="utf-8")

    model_version: str | None = None
    candidate = False
    champion = False
    if mlflow_enabled:
        experiment = (
            "gridmind-weather-demand"
            if target == "demand_mw"
            else f"gridmind-{target.replace('_mw', '').replace('_', '-')}"
        )
        initialize_mlflow(settings, experiment)
        with mlflow.start_run(run_name=f"{target}-{weather_mode}") as run:
            mlflow.log_params(
                {
                    "target": target,
                    "weather_mode": weather_mode,
                    "weather_provider": settings.weather_provider,
                    "region": region,
                    "mapping_config": str(settings.grid_location_config),
                    "horizon": horizon,
                    "validation_windows": validation_windows,
                }
            )
            metrics = results[best_name].overall_metrics
            mlflow.log_metrics({key: value for key, value in metrics.items() if pd.notna(value)})
            mlflow.log_artifacts(str(artifacts), artifact_path="target_training_artifacts")
            final_model.run_id = run.info.run_id
            logged = log_mlflow_model(final_model)
            if register_model:
                registry_name = str(getattr(settings, definition.registry_setting))
                client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
                model_version = create_model_version(
                    client,
                    model_name=registry_name,
                    run_id=run.info.run_id,
                    source=str(logged.model_uri),
                )
                reference = next(
                    row[settings.primary_selection_metric]
                    for row in rows
                    if row["model_name"] == "seasonal_naive_24h"
                )
                decision = apply_promotion_gate(
                    client,
                    registered_model_name=registry_name,
                    version=model_version,
                    metrics=metrics,
                    reference_metric=float(reference),
                    primary_metric=settings.primary_selection_metric,
                    threshold=settings.model_promotion_threshold,
                    bundle_path=bundle,
                )
                candidate = decision.candidate_assigned
                champion = decision.champion_promoted
    return TargetTrainingResult(
        target,
        leaderboard,
        artifacts,
        bundle,
        best_name,
        model_version,
        candidate,
        champion,
    )


def _load_target_frame(settings: Settings, target: str, region: str) -> pd.DataFrame:
    if target == "demand_mw":
        return DuckDBStorage(settings.duckdb_path).read_data(regions=[region])
    renewable = RenewableStorage(settings.duckdb_path).read(region)
    if target == "net_load_mw":
        demand = DuckDBStorage(settings.duckdb_path).read_data(regions=[region])
        frame, _report = compute_net_load(demand, renewable)
        return frame.merge(
            renewable[["region", "timestamp_utc", "solar_generation_mw", "wind_generation_mw"]],
            on=["region", "timestamp_utc"],
            how="left",
        )
    return renewable


def _load_weather(settings: Settings, region: str, mode: WeatherMode) -> pd.DataFrame:
    storage = WeatherStorage(settings.duckdb_path)
    if mode == "historical_oracle":
        return storage.read_regions(region, data_type="historical")
    try:
        forecast = storage.read_regions(region, data_type="forecast")
    except Exception:
        forecast = pd.DataFrame()
    if not forecast.empty:
        return forecast
    historical = storage.read_regions(region, data_type="historical")
    return simulate_forecast_weather(historical)


def _evaluate_component_net_load(
    frame: pd.DataFrame, selection: Any, horizon: int
) -> dict[str, float]:
    predictions: list[pd.DataFrame] = []
    for window in selection.windows:
        parts: dict[str, pd.DataFrame] = {}
        for target in ("demand_mw", "solar_generation_mw", "wind_generation_mw"):
            history = frame.loc[
                frame["timestamp_utc"] <= window.origin,
                [
                    "region",
                    "timestamp_utc",
                    target,
                ],
            ].rename(columns={target: "demand_mw"})
            parts[target] = SeasonalNaiveForecaster(24).predict(history, horizon=horizon)
        component = parts["demand_mw"][["region", "timestamp_utc"]].copy()
        component["predicted_value"] = (
            parts["demand_mw"]["predicted_demand_mw"]
            - parts["solar_generation_mw"]["predicted_demand_mw"]
            - parts["wind_generation_mw"]["predicted_demand_mw"]
        )
        actual = frame.loc[
            frame["timestamp_utc"].isin(window.validation_timestamps),
            ["region", "timestamp_utc", "net_load_mw"],
        ].rename(columns={"net_load_mw": "actual_value"})
        predictions.append(component.merge(actual, on=["region", "timestamp_utc"]))
    combined = pd.concat(predictions, ignore_index=True)
    metrics = calculate_metrics(combined["actual_value"], combined["predicted_value"])
    metrics["forecast_bias"] = metrics.pop("bias")
    return metrics
