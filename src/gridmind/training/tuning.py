"""Nested rolling Optuna optimization isolated from final evaluation windows."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import optuna
import pandas as pd

from gridmind.continuity import WindowSelection, select_gap_aware_windows
from gridmind.features.builder import FeatureBuilder
from gridmind.features.contracts import FeatureSpecification
from gridmind.models.factories import create_model
from gridmind.time_utils import to_utc_timestamp
from gridmind.training.datasets import reserve_final_evaluation_history
from gridmind.training.evaluator import evaluate_model


@dataclass(frozen=True)
class TuningResult:
    """Best parameters, study, data boundary, and persisted tuning artifacts."""

    best_params: dict[str, Any]
    best_value: float
    study: optuna.Study
    tuning_end: pd.Timestamp
    final_evaluation_start: pd.Timestamp
    window_selection: WindowSelection
    artifact_paths: tuple[Path, ...]


def tune_model(
    frame: pd.DataFrame,
    model_name: str,
    *,
    final_horizon: int,
    final_windows: int,
    step_size: int,
    tuning_windows: int,
    trials: int,
    timeout_seconds: int | None,
    random_seed: int,
    n_jobs: int,
    output_dir: Path,
    metric: str = "wape",
    mlflow_enabled: bool = False,
    final_window_selection: WindowSelection | None = None,
    tuning_window_selection: WindowSelection | None = None,
) -> TuningResult:
    """Optimize inner-window mean error without reading final evaluation targets."""
    if trials <= 0 or tuning_windows <= 0:
        raise ValueError("Optuna trials and tuning windows must be positive.")
    specification = FeatureSpecification.create()
    final_selection = final_window_selection or select_gap_aware_windows(
        frame,
        horizon=final_horizon,
        windows=final_windows,
        step_size=step_size,
        required_history=specification.required_history,
    )
    tuning_history, _final, final_start = reserve_final_evaluation_history(
        frame,
        horizon=final_horizon,
        windows=final_windows,
        step_size=step_size,
        required_history=specification.required_history,
        selection=final_selection,
    )
    if tuning_history.empty:
        raise ValueError("No older history remains after reserving final evaluation windows.")
    tuning_end = to_utc_timestamp(tuning_history["timestamp_utc"].max())
    tuning_selection = tuning_window_selection or select_gap_aware_windows(
        tuning_history,
        horizon=final_horizon,
        windows=tuning_windows,
        step_size=step_size,
        required_history=specification.required_history,
    )
    latest_tuning_target = max(
        window.validation_timestamps[-1] for window in tuning_selection.windows
    )
    earliest_final_target = min(
        window.validation_timestamps[0] for window in final_selection.windows
    )
    if latest_tuning_target >= earliest_final_target:
        raise AssertionError("Tuning windows overlap the untouched final evaluation period.")
    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    shared_builder = FeatureBuilder(specification, cache_enabled=True)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_parameters(trial, model_name)

        def evaluate() -> float:
            result = evaluate_model(
                tuning_history,
                lambda: create_model(
                    model_name,
                    builder=shared_builder,
                    random_seed=random_seed,
                    n_jobs=n_jobs,
                    params=params,
                ),
                horizon=final_horizon,
                windows=tuning_windows,
                step_size=step_size,
                window_selection=tuning_selection,
            )
            value = float(result.overall_metrics[metric])
            if not math.isfinite(value):
                raise ValueError(f"Tuning metric {metric} is not finite.")
            return value

        try:
            if mlflow_enabled and mlflow.active_run() is not None:
                with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
                    mlflow.log_params(params)
                    value = evaluate()
                    mlflow.log_metric(metric, value)
                    return value
            return evaluate()
        except Exception as exc:
            trial.set_user_attr("failure", str(exc))
            raise optuna.TrialPruned(str(exc)) from exc

    study.optimize(objective, n_trials=trials, timeout=timeout_seconds, gc_after_trial=True)
    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise ValueError(f"Every Optuna trial failed for {model_name}.")
    paths = write_study_artifacts(study, output_dir)
    return TuningResult(
        best_params=dict(study.best_params),
        best_value=float(study.best_value),
        study=study,
        tuning_end=tuning_end,
        final_evaluation_start=final_start,
        window_selection=tuning_selection,
        artifact_paths=tuple(paths),
    )


def suggest_parameters(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    """Return a compact, production-relevant search space."""
    if model_name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 80, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 3.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 3.0, log=True),
        }
    if model_name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 80, 300),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 3.0),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        }
    raise ValueError(f"No tuning search space exists for model '{model_name}'.")


def write_study_artifacts(study: optuna.Study, output_dir: Path) -> list[Path]:
    """Persist best parameters, study summary, history, and available importance."""
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_parameters.json"
    best_path.write_text(json.dumps(study.best_params, indent=2), encoding="utf-8")
    summary_path = output_dir / "optuna_study_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "direction": study.direction.name,
                "best_value": study.best_value,
                "completed_trials": len(study.trials),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    history_path = output_dir / "optimization_history.csv"
    study.trials_dataframe().to_csv(history_path, index=False)
    importance_path = output_dir / "parameter_importance.json"
    try:
        importance = optuna.importance.get_param_importances(study)
    except (RuntimeError, ValueError, ZeroDivisionError):
        importance = {}
    importance_path.write_text(json.dumps(importance, indent=2), encoding="utf-8")
    return [best_path, summary_path, history_path, importance_path]
