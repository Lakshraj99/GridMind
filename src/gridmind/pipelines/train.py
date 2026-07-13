"""Complete Milestone 2 training, comparison, tracking, registry, and SHAP pipeline."""

from __future__ import annotations

import math
import subprocess
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow import MlflowClient

from gridmind.config import Settings
from gridmind.continuity import (
    WindowSelection,
    detect_contiguous_segments,
    select_gap_aware_windows,
)
from gridmind.data.processing import generate_quality_report
from gridmind.data.storage import DuckDBStorage, write_json_report
from gridmind.explainability.shap_analysis import ShapArtifacts, generate_shap_artifacts
from gridmind.features.builder import FeatureBuilder
from gridmind.features.contracts import FeatureSpecification
from gridmind.forecasting.baselines import (
    AveragedSeasonalNaiveForecaster,
    LastValueForecaster,
    SeasonalNaiveForecaster,
)
from gridmind.mlflow_config import initialize_mlflow
from gridmind.models.factories import SUPPORTED_MODELS, create_model
from gridmind.models.promotion import (
    PromotionDecision,
    apply_promotion_gate,
    create_model_version,
    effective_registry_uri,
)
from gridmind.models.protocols import TrainableForecastModel
from gridmind.models.serialization import (
    load_model_bundle,
    log_mlflow_model,
    save_model_bundle,
)
from gridmind.time_utils import format_utc_timestamp
from gridmind.training.datasets import reserve_final_evaluation_history
from gridmind.training.evaluator import EvaluationResult, evaluate_baseline, evaluate_model
from gridmind.training.leaderboard import ModelEvaluationRecord, create_leaderboard
from gridmind.training.trainer import fit_final_model
from gridmind.training.tuning import TuningResult, tune_model


@dataclass(frozen=True)
class TrainingPipelineResult:
    """Leaderboard, selected bundle, registry decision, and complete artifact directory."""

    leaderboard: pd.DataFrame
    selected_model: str
    bundle_path: Path
    artifact_dir: Path
    parent_run_id: str
    selected_run_id: str
    model_version: str | None
    promotion: PromotionDecision | None
    shap_artifacts: ShapArtifacts | None
    continuity_summary: dict[str, Any]
    window_selection_path: Path
    tuning_origins: tuple[str, ...]
    final_validation_origins: tuple[str, ...]


def run_training_pipeline(
    settings: Settings,
    *,
    frame: pd.DataFrame | None = None,
    regions: list[str] | None = None,
    model_names: list[str] | None = None,
    horizon: int | None = None,
    validation_windows: int | None = None,
    step_size: int | None = None,
    tune: bool = False,
    trials: int | None = None,
    mlflow_enabled: bool | None = None,
    register_model: bool | None = None,
    random_seed: int | None = None,
    output_dir: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> TrainingPipelineResult:
    """Train, tune, compare, explain, serialize, and optionally register global models."""
    selected_models = model_names or list(SUPPORTED_MODELS)
    unsupported = set(selected_models).difference(SUPPORTED_MODELS)
    if unsupported:
        raise ValueError(f"Unsupported models: {sorted(unsupported)}")
    selected_horizon = horizon or settings.forecast_horizon
    selected_windows = validation_windows or settings.validation_windows
    selected_step = step_size or settings.validation_step_size
    selected_trials = trials or settings.optuna_trials
    seed = random_seed if random_seed is not None else settings.model_random_seed
    use_mlflow = settings.mlflow_enabled if mlflow_enabled is None else mlflow_enabled
    use_registry = settings.mlflow_register_model if register_model is None else register_model
    if use_registry and not use_mlflow:
        use_registry = False
    data = (
        frame
        if frame is not None
        else DuckDBStorage(settings.duckdb_path).read_data(regions=regions)
    )
    if data.empty:
        raise ValueError("No validated grid data is available for training.")
    data = data.sort_values(["timestamp_utc", "region"], ignore_index=True)
    specification = FeatureSpecification.create()
    continuity_summary = detect_contiguous_segments(data).summary()
    final_selection = select_gap_aware_windows(
        data,
        horizon=selected_horizon,
        windows=selected_windows,
        step_size=selected_step,
        required_history=specification.required_history,
    )
    tuning_selection: WindowSelection | None = None
    if tune:
        tuning_history, _final_data, _final_start = reserve_final_evaluation_history(
            data,
            horizon=selected_horizon,
            windows=selected_windows,
            step_size=selected_step,
            required_history=specification.required_history,
            selection=final_selection,
        )
        tuning_selection = select_gap_aware_windows(
            tuning_history,
            horizon=selected_horizon,
            windows=settings.tuning_windows,
            step_size=selected_step,
            required_history=specification.required_history,
        )
        if max(window.validation_timestamps[-1] for window in tuning_selection.windows) >= min(
            window.validation_timestamps[0] for window in final_selection.windows
        ):
            raise AssertionError("Tuning and final evaluation windows overlap.")
    run_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifacts = output_dir or Path("artifacts") / "training" / run_stamp
    artifacts.mkdir(parents=True, exist_ok=True)
    final_origin_strings = tuple(format_utc_timestamp(value) for value in final_selection.origins)
    tuning_origin_strings = (
        tuple(format_utc_timestamp(value) for value in tuning_selection.origins)
        if tuning_selection is not None
        else ()
    )
    if progress is not None:
        progress(
            "Continuity: "
            f"regions={continuity_summary['region_count']}; "
            f"timestamp gaps={continuity_summary['timestamp_gap_count']}; "
            f"segments={continuity_summary['contiguous_segment_count']}; "
            "longest segment="
            f"{continuity_summary['longest_contiguous_segment_rows']} hours"
        )
        progress(
            "Selected tuning origins: "
            + (", ".join(tuning_origin_strings) if tuning_origin_strings else "none")
        )
        progress("Selected final evaluation origins: " + ", ".join(final_origin_strings))
    window_selection_path = write_json_report(
        {
            "continuity": continuity_summary,
            "tuning": tuning_selection.to_dict() if tuning_selection is not None else None,
            "final_evaluation": final_selection.to_dict(),
            "tuning_final_overlap": False,
        },
        artifacts / "window_selection.json",
    )
    specification.save(artifacts / "feature_schema.json")
    feature_build = FeatureBuilder(specification).build_training(data)
    write_json_report(feature_build.report.to_dict(), artifacts / "feature_build_report.json")
    write_json_report(generate_quality_report(data), artifacts / "data_quality_report.json")
    config_snapshot = _configuration_snapshot(
        settings,
        data=data,
        model_names=selected_models,
        horizon=selected_horizon,
        windows=selected_windows,
        step_size=selected_step,
        tune=tune,
        trials=selected_trials,
        seed=seed,
    )
    config_snapshot.update(
        {
            "timestamp_gap_count": continuity_summary["timestamp_gap_count"],
            "contiguous_segment_count": continuity_summary["contiguous_segment_count"],
            "tuning_origins": ",".join(tuning_origin_strings),
            "final_validation_origins": ",".join(final_origin_strings),
        }
    )
    write_json_report(config_snapshot, artifacts / "configuration.json")

    tracking_uri = settings.mlflow_tracking_uri
    if use_registry:
        tracking_uri = effective_registry_uri(
            settings.mlflow_tracking_uri, settings.data_dir / "mlflow_registry.db"
        )
    if use_mlflow:
        initialize_mlflow(
            settings,
            settings.mlflow_experiment_name,
            tracking_uri=tracking_uri,
        )

    records: list[ModelEvaluationRecord] = []
    results_by_name: dict[str, EvaluationResult] = {}
    best_parameters: dict[str, dict[str, Any]] = {}
    tuning_results: dict[str, TuningResult] = {}
    with _parent_run(use_mlflow, config_snapshot) as parent_run_id:
        baseline_factories = {
            "last_value": LastValueForecaster,
            "seasonal_naive_24h": lambda: SeasonalNaiveForecaster(24),
            "seasonal_naive_168h": lambda: SeasonalNaiveForecaster(168),
            "seasonal_naive_average_24h_168h": AveragedSeasonalNaiveForecaster,
        }
        for name, factory in baseline_factories.items():
            result = evaluate_baseline(
                data,
                factory,
                horizon=selected_horizon,
                windows=selected_windows,
                step_size=selected_step,
                window_selection=final_selection,
            )
            run_id = _log_evaluation_child(name, result, {}, use_mlflow)
            records.append(ModelEvaluationRecord(name, result, run_id))
            results_by_name[name] = result

        for model_name in selected_models:
            untuned_result = evaluate_model(
                data,
                partial(
                    create_model,
                    model_name,
                    specification=specification,
                    random_seed=seed,
                    n_jobs=settings.model_n_jobs,
                    params={},
                ),
                horizon=selected_horizon,
                windows=selected_windows,
                step_size=selected_step,
                window_selection=final_selection,
            )
            untuned_run_id = _log_evaluation_child(
                f"{model_name}_untuned", untuned_result, {}, use_mlflow
            )
            params: dict[str, Any] = {}
            if tune:
                tuning = tune_model(
                    data,
                    model_name,
                    final_horizon=selected_horizon,
                    final_windows=selected_windows,
                    step_size=selected_step,
                    tuning_windows=settings.tuning_windows,
                    trials=selected_trials,
                    timeout_seconds=settings.optuna_timeout_seconds,
                    random_seed=seed,
                    n_jobs=settings.model_n_jobs,
                    output_dir=artifacts / "tuning" / model_name,
                    metric=settings.primary_selection_metric,
                    mlflow_enabled=use_mlflow,
                    final_window_selection=final_selection,
                    tuning_window_selection=tuning_selection,
                )
                params = tuning.best_params
                tuning_results[model_name] = tuning
            best_parameters[model_name] = params
            if tune:
                model_factory = partial(
                    create_model,
                    model_name,
                    specification=specification,
                    random_seed=seed,
                    n_jobs=settings.model_n_jobs,
                    params=params,
                )
                result = evaluate_model(
                    data,
                    model_factory,
                    horizon=selected_horizon,
                    windows=selected_windows,
                    step_size=selected_step,
                    window_selection=final_selection,
                )
                run_id = _log_evaluation_child(f"{model_name}_tuned", result, params, use_mlflow)
            else:
                result = untuned_result
                run_id = untuned_run_id
            canonical_name = f"{model_name}_global"
            records.append(ModelEvaluationRecord(canonical_name, result, run_id))
            results_by_name[canonical_name] = result

        leaderboard = create_leaderboard(records, primary_metric=settings.primary_selection_metric)
        _write_evaluation_artifacts(leaderboard, records, best_parameters, artifacts)
        selected_row = leaderboard.loc[
            leaderboard["model_name"].isin([f"{name}_global" for name in selected_models])
        ].iloc[0]
        selected_model_name = str(selected_row["model_name"]).removesuffix("_global")
        final_model = fit_final_model(
            data,
            selected_model_name,
            params=best_parameters[selected_model_name],
            random_seed=seed,
            n_jobs=settings.model_n_jobs,
        )
        bundle_path = save_model_bundle(
            final_model,
            artifacts / "bundle" / "model_bundle.joblib",
            metadata={
                "training_start": format_utc_timestamp(data["timestamp_utc"].min()),
                "training_end": format_utc_timestamp(data["timestamp_utc"].max()),
                "regions": sorted(str(value) for value in data["region"].unique()),
                "row_count": len(data),
            },
        )
        reloaded = load_model_bundle(bundle_path)
        _validate_reloaded_model(reloaded.model, data)
        shap_artifacts = _safe_shap(
            reloaded.model,
            data,
            artifacts / "explainability",
            settings=settings,
        )
        selected_run_id = ""
        logged_model_uri = ""
        model_version: str | None = None
        promotion: PromotionDecision | None = None
        if use_mlflow:
            with mlflow.start_run(run_name="selected-final-model", nested=True) as selected_run:
                selected_run_id = selected_run.info.run_id
                reloaded.model.run_id = selected_run_id
                mlflow.log_params(reloaded.model.get_params())
                mlflow.log_metrics(
                    _finite_metrics(
                        results_by_name[f"{selected_model_name}_global"].overall_metrics
                    )
                )
                mlflow.log_artifacts(str(artifacts / "bundle"), artifact_path="bundle")
                if shap_artifacts is not None:
                    mlflow.log_artifacts(
                        str(artifacts / "explainability"), artifact_path="explainability"
                    )
                logged_model = log_mlflow_model(reloaded.model)
                logged_model_uri = str(logged_model.model_uri)
            if use_registry:
                client = MlflowClient(tracking_uri=tracking_uri)
                model_version = create_model_version(
                    client,
                    model_name=settings.mlflow_model_name,
                    run_id=selected_run_id,
                    source=logged_model_uri,
                )
                reloaded.model.model_version = model_version
                ml_metrics = results_by_name[f"{selected_model_name}_global"].overall_metrics
                baseline_metric = results_by_name["seasonal_naive_24h"].overall_metrics[
                    settings.primary_selection_metric
                ]
                promotion = apply_promotion_gate(
                    client,
                    registered_model_name=settings.mlflow_model_name,
                    version=model_version,
                    metrics=ml_metrics,
                    reference_metric=baseline_metric,
                    primary_metric=settings.primary_selection_metric,
                    threshold=settings.model_promotion_threshold,
                    bundle_path=bundle_path,
                )
            mlflow.log_artifacts(str(artifacts), artifact_path="training_artifacts")
    return TrainingPipelineResult(
        leaderboard=leaderboard,
        selected_model=f"{selected_model_name}_global",
        bundle_path=bundle_path,
        artifact_dir=artifacts,
        parent_run_id=parent_run_id,
        selected_run_id=selected_run_id,
        model_version=model_version,
        promotion=promotion,
        shap_artifacts=shap_artifacts,
        continuity_summary=continuity_summary,
        window_selection_path=window_selection_path,
        tuning_origins=tuning_origin_strings,
        final_validation_origins=final_origin_strings,
    )


@contextmanager
def _parent_run(enabled: bool, configuration: dict[str, Any]) -> Iterator[str]:
    if not enabled:
        yield ""
        return
    with mlflow.start_run(run_name="gridmind-training") as run:
        mlflow.log_params(configuration)
        yield run.info.run_id


def _log_evaluation_child(
    name: str,
    result: EvaluationResult,
    params: dict[str, Any],
    enabled: bool,
) -> str:
    if not enabled:
        return ""
    with mlflow.start_run(run_name=name, nested=True) as run:
        if params:
            mlflow.log_params(params)
        mlflow.log_metrics(_finite_metrics(result.overall_metrics))
        mlflow.log_metric("training_seconds", result.training_seconds)
        mlflow.log_metric("prediction_seconds", result.prediction_seconds)
        return str(run.info.run_id)


def _write_evaluation_artifacts(
    leaderboard: pd.DataFrame,
    records: list[ModelEvaluationRecord],
    best_parameters: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)
    leaderboard.to_json(output_dir / "leaderboard.json", orient="records", indent=2)
    predictions: list[pd.DataFrame] = []
    windows: list[pd.DataFrame] = []
    horizons: list[pd.DataFrame] = []
    regions: list[pd.DataFrame] = []
    for record in records:
        prediction = record.result.predictions.copy()
        prediction["evaluated_model"] = record.model_name
        predictions.append(prediction)
        for collection, frame in (
            (windows, record.result.window_metrics),
            (horizons, record.result.horizon_metrics),
            (regions, record.result.region_metrics),
        ):
            metrics = frame.copy()
            metrics.insert(0, "model_name", record.model_name)
            collection.append(metrics)
    pd.concat(predictions, ignore_index=True).to_parquet(
        output_dir / "validation_predictions.parquet", index=False
    )
    pd.concat(windows, ignore_index=True).to_csv(output_dir / "window_metrics.csv", index=False)
    pd.concat(horizons, ignore_index=True).to_csv(output_dir / "horizon_metrics.csv", index=False)
    pd.concat(regions, ignore_index=True).to_csv(output_dir / "region_metrics.csv", index=False)
    write_json_report(dict(best_parameters), output_dir / "best_parameters.json")


def _safe_shap(
    model: TrainableForecastModel,
    data: pd.DataFrame,
    output_dir: Path,
    *,
    settings: Settings,
) -> ShapArtifacts | None:
    try:
        return generate_shap_artifacts(
            model,
            data,
            output_dir,
            sample_size=settings.shap_sample_size,
            random_seed=settings.model_random_seed,
        )
    except Exception as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "shap_failure.txt").write_text(str(exc), encoding="utf-8")
        return None


def _validate_reloaded_model(model: TrainableForecastModel, data: pd.DataFrame) -> None:
    predictions = model.predict(data, horizon=1)
    values = predictions["predicted_demand_mw"]
    if predictions.empty or not values.map(math.isfinite).all():
        raise ValueError("Reloaded selected model failed batch-prediction validation.")


def _finite_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {key: value for key, value in metrics.items() if math.isfinite(value)}


def _configuration_snapshot(
    settings: Settings,
    *,
    data: pd.DataFrame,
    model_names: list[str],
    horizon: int,
    windows: int,
    step_size: int,
    tune: bool,
    trials: int,
    seed: int,
) -> dict[str, Any]:
    specification = FeatureSpecification.create()
    return {
        "dataset_start": format_utc_timestamp(data["timestamp_utc"].min()),
        "dataset_end": format_utc_timestamp(data["timestamp_utc"].max()),
        "row_count": len(data),
        "region_count": int(data["region"].nunique()),
        "regions": ",".join(sorted(str(value) for value in data["region"].unique())),
        "forecast_horizon": horizon,
        "validation_windows": windows,
        "validation_step_size": step_size,
        "models": ",".join(model_names),
        "tuning_enabled": tune,
        "optuna_trials": trials,
        "random_seed": seed,
        "primary_metric": settings.primary_selection_metric,
        "feature_count": len(specification.feature_names),
        "demand_lags": ",".join(str(value) for value in specification.lags),
        "rolling_windows": ",".join(str(value) for value in specification.rolling_windows),
        "git_commit": _git_commit(),
    }


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"
