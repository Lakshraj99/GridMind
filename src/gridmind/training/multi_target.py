"""Shared-window evaluation and metrics for independently configured targets."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

from gridmind.continuity import WindowSelection, select_gap_aware_windows
from gridmind.forecasting.metrics import calculate_metrics
from gridmind.models.target_factory import TargetForecaster
from gridmind.training.tuning import suggest_parameters, write_study_artifacts


@dataclass(frozen=True)
class TargetEvaluationResult:
    predictions: pd.DataFrame
    overall_metrics: dict[str, float]
    window_metrics: pd.DataFrame
    horizon_metrics: pd.DataFrame
    region_metrics: pd.DataFrame
    target_metrics: dict[str, float]
    selection: WindowSelection


@dataclass(frozen=True)
class TargetTuningResult:
    best_params: dict[str, Any]
    selection: WindowSelection
    artifact_paths: tuple[Path, ...]


def tune_target_model(
    frame: pd.DataFrame,
    target: str,
    model_name: str,
    factory: Callable[[dict[str, Any]], TargetForecaster],
    *,
    horizon: int,
    windows: int,
    step_size: int,
    required_history: int,
    trials: int,
    random_seed: int,
    output_dir: Path,
) -> TargetTuningResult:
    """Tune only older gap-safe windows supplied by the caller."""
    selection = select_gap_aware_windows(
        frame.rename(columns={target: "demand_mw"}),
        horizon=horizon,
        windows=windows,
        step_size=step_size,
        required_history=required_history,
    )
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_seed)
    )

    def objective(trial: optuna.Trial) -> float:
        params = suggest_parameters(trial, model_name)
        result = evaluate_target_model(
            frame,
            target,
            lambda: factory(params),
            horizon=horizon,
            windows=windows,
            step_size=step_size,
            selection=selection,
        )
        return float(result.overall_metrics["wape"])

    study.optimize(objective, n_trials=trials)
    return TargetTuningResult(
        dict(study.best_params), selection, tuple(write_study_artifacts(study, output_dir))
    )


def evaluate_target_model(
    frame: pd.DataFrame,
    target: str,
    factory: Callable[[], TargetForecaster],
    *,
    horizon: int,
    windows: int,
    step_size: int,
    selection: WindowSelection | None = None,
) -> TargetEvaluationResult:
    """Evaluate recursive target forecasts without exposing validation targets."""
    probe = factory()
    selected = selection or select_gap_aware_windows(
        frame.rename(columns={target: "demand_mw"}),
        horizon=horizon,
        windows=windows,
        step_size=step_size,
        required_history=probe.specification.required_history,
    )
    outputs: list[pd.DataFrame] = []
    clipping_count = 0
    for window_id, window in enumerate(selected.windows, start=1):
        training = frame.loc[frame["timestamp_utc"] <= window.origin].copy()
        validation = frame.loc[frame["timestamp_utc"].isin(window.validation_timestamps)].copy()
        prediction_input = pd.concat(
            [training, validation.assign(**{target: np.nan})], ignore_index=True
        )
        model = factory().fit(training)
        predicted = model.predict(prediction_input, horizon=horizon)
        clipping_count += model.clipping_count
        actual_columns = ["region", "timestamp_utc", target]
        if "solar_radiation_daylight" in validation:
            actual_columns.append("solar_radiation_daylight")
        actual = validation[actual_columns].rename(columns={target: "actual_value"})
        merged = predicted.merge(actual, on=["region", "timestamp_utc"], validate="one_to_one")
        merged["validation_window"] = window_id
        outputs.append(merged)
    predictions = pd.concat(outputs, ignore_index=True)
    predictions.attrs["forecast_clipping_count"] = clipping_count
    overall = _metrics(predictions)
    return TargetEvaluationResult(
        predictions,
        overall,
        _metrics_by(predictions, "validation_window"),
        _metrics_by(predictions, "forecast_step"),
        _metrics_by(predictions, "region"),
        target_specific_metrics(predictions, target),
        selected,
    )


def target_specific_metrics(predictions: pd.DataFrame, target: str) -> dict[str, float]:
    """Add scale-normalized and zero/daylight slices for renewable targets."""
    actual = predictions["actual_value"].to_numpy(dtype=float)
    predicted = predictions["predicted_value"].to_numpy(dtype=float)
    scale = float(np.nanmax(actual) - np.nanmin(actual)) if len(actual) else 0.0
    result = {
        "normalized_mae": float(np.mean(np.abs(actual - predicted)) / scale)
        if scale
        else float("nan"),
        "normalized_rmse": float(np.sqrt(np.mean((actual - predicted) ** 2)) / scale)
        if scale
        else float("nan"),
        "forecast_clipping_count": float(predictions.attrs.get("forecast_clipping_count", 0)),
    }
    zero = actual == 0
    result["zero_generation_mae"] = (
        float(np.mean(np.abs(predicted[zero]))) if zero.any() else float("nan")
    )
    if target == "solar_generation_mw" and "solar_radiation_daylight" in predictions:
        daytime = predictions["solar_radiation_daylight"].to_numpy(dtype=float) > 0
        result["daytime_mae"] = (
            float(np.mean(np.abs(actual[daytime] - predicted[daytime])))
            if daytime.any()
            else float("nan")
        )
    return result


def _metrics(frame: pd.DataFrame) -> dict[str, float]:
    metrics = calculate_metrics(frame["actual_value"], frame["predicted_value"])
    metrics["forecast_bias"] = metrics.pop("bias")
    return metrics


def _metrics_by(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {column: value, **_metrics(group)}
            for value, group in frame.groupby(column, observed=True)
        ]
    )
