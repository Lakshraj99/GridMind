"""Rolling-origin evaluation for global ML models and existing baselines."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from gridmind.continuity import WindowSelection, select_gap_aware_windows
from gridmind.exceptions import PredictionValidationError
from gridmind.forecasting.baselines import BaselineForecaster
from gridmind.forecasting.metrics import calculate_metrics, evaluate_predictions
from gridmind.models.protocols import TrainableForecastModel


@dataclass(frozen=True)
class EvaluationResult:
    """Predictions, metric slices, and measured durations for rolling evaluation."""

    predictions: pd.DataFrame
    overall_metrics: dict[str, float]
    window_metrics: pd.DataFrame
    horizon_metrics: pd.DataFrame
    region_metrics: pd.DataFrame
    training_seconds: float
    prediction_seconds: float
    window_selection: WindowSelection | None = None


def rolling_windows(
    frame: pd.DataFrame,
    *,
    horizon: int,
    windows: int,
    step_size: int,
    required_history: int = 1,
) -> list[tuple[pd.Timestamp, pd.DatetimeIndex]]:
    """Create chronological, fully contiguous origins shared by every region."""
    selection = select_gap_aware_windows(
        frame,
        horizon=horizon,
        windows=windows,
        step_size=step_size,
        required_history=required_history,
    )
    return [(window.origin, window.validation_timestamps) for window in selection.windows]


def evaluate_model(
    frame: pd.DataFrame,
    model_factory: Callable[[], TrainableForecastModel],
    *,
    horizon: int = 24,
    windows: int = 5,
    step_size: int = 24,
    window_selection: WindowSelection | None = None,
) -> EvaluationResult:
    """Refit a global model at every origin and evaluate recursive forecasts."""
    selection = window_selection
    if selection is None:
        probe = model_factory()
        selection = select_gap_aware_windows(
            frame,
            horizon=horizon,
            windows=windows,
            step_size=step_size,
            required_history=probe.specification.required_history,
        )
    prediction_frames: list[pd.DataFrame] = []
    training_seconds = 0.0
    prediction_seconds = 0.0
    for window_id, (origin, validation_times) in enumerate(
        ((window.origin, window.validation_timestamps) for window in selection.windows), start=1
    ):
        training = frame.loc[frame["timestamp_utc"] <= origin].copy()
        validation = frame.loc[frame["timestamp_utc"].isin(validation_times)].copy()
        if training.empty or training["timestamp_utc"].max() >= validation["timestamp_utc"].min():
            raise ValueError("Training observations must occur before validation observations.")
        model = model_factory()
        started = time.perf_counter()
        model.fit(training)
        training_seconds += time.perf_counter() - started
        started = time.perf_counter()
        forecast = model.predict(training, horizon=horizon)
        prediction_seconds += time.perf_counter() - started
        prediction_frames.append(
            _attach_validation(forecast, validation, window_id=window_id, horizon=horizon)
        )
    return summarize_predictions(
        pd.concat(prediction_frames, ignore_index=True),
        training_seconds=training_seconds,
        prediction_seconds=prediction_seconds,
        window_selection=selection,
    )


def evaluate_baseline(
    frame: pd.DataFrame,
    baseline_factory: Callable[[], BaselineForecaster],
    *,
    horizon: int = 24,
    windows: int = 5,
    step_size: int = 24,
    window_selection: WindowSelection | None = None,
) -> EvaluationResult:
    """Evaluate an existing single-region baseline over all regions and origins."""
    selection = window_selection
    if selection is None:
        probe = baseline_factory()
        required_history = int(getattr(probe, "lag", 168 if "average" in probe.name else 1))
        selection = select_gap_aware_windows(
            frame,
            horizon=horizon,
            windows=windows,
            step_size=step_size,
            required_history=required_history,
        )
    prediction_frames: list[pd.DataFrame] = []
    prediction_seconds = 0.0
    for window_id, (origin, validation_times) in enumerate(
        ((window.origin, window.validation_timestamps) for window in selection.windows), start=1
    ):
        for _region, region_frame in frame.groupby("region", sort=True, observed=True):
            training = region_frame.loc[region_frame["timestamp_utc"] <= origin].copy()
            validation = region_frame.loc[
                region_frame["timestamp_utc"].isin(validation_times)
            ].copy()
            started = time.perf_counter()
            forecast = baseline_factory().predict(training, horizon=horizon)
            prediction_seconds += time.perf_counter() - started
            forecast["model_version"] = "baseline"
            forecast["run_id"] = ""
            forecast["created_at_utc"] = pd.Timestamp.now(tz="UTC")
            prediction_frames.append(
                _attach_validation(forecast, validation, window_id=window_id, horizon=horizon)
            )
    return summarize_predictions(
        pd.concat(prediction_frames, ignore_index=True),
        training_seconds=0.0,
        prediction_seconds=prediction_seconds,
        window_selection=selection,
    )


def summarize_predictions(
    predictions: pd.DataFrame,
    *,
    training_seconds: float,
    prediction_seconds: float,
    window_selection: WindowSelection | None = None,
) -> EvaluationResult:
    """Validate forecast values and calculate overall, window, step, and region metrics."""
    values = predictions["predicted_demand_mw"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise PredictionValidationError("Validation predictions contain non-finite values.")
    if (values < 0).any():
        raise PredictionValidationError("Validation predictions contain negative demand.")
    evaluated = evaluate_predictions(predictions)
    overall = dict(evaluated["overall"])
    overall["forecast_bias"] = overall.pop("bias")
    window_metrics = _metrics_by(predictions, "validation_window")
    horizon_metrics = _metrics_by(predictions, "forecast_step")
    region_metrics = _metrics_by(predictions, "region")
    return EvaluationResult(
        predictions=predictions,
        overall_metrics=overall,
        window_metrics=window_metrics,
        horizon_metrics=horizon_metrics,
        region_metrics=region_metrics,
        training_seconds=training_seconds,
        prediction_seconds=prediction_seconds,
        window_selection=window_selection,
    )


def relative_improvement(model_metric: float, baseline_metric: float) -> float:
    """Return honest relative error reduction; negative means the model is worse."""
    if (
        not math.isfinite(model_metric)
        or not math.isfinite(baseline_metric)
        or baseline_metric == 0
    ):
        return float("nan")
    return (baseline_metric - model_metric) / baseline_metric


def _attach_validation(
    forecast: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    window_id: int,
    horizon: int,
) -> pd.DataFrame:
    actuals = validation[["region", "timestamp_utc", "demand_mw"]].rename(
        columns={"demand_mw": "actual_demand_mw"}
    )
    result = forecast.drop(columns=["actual_demand_mw"], errors="ignore").merge(
        actuals, on=["region", "timestamp_utc"], how="left", validate="one_to_one"
    )
    if len(result) != len(actuals) or result["actual_demand_mw"].isna().any():
        raise PredictionValidationError("Forecast timestamps do not match validation targets.")
    result["validation_window"] = window_id
    result["forecast_step"] = (
        result.sort_values(["region", "timestamp_utc"]).groupby("region", observed=True).cumcount()
        + 1
    )
    if result["forecast_step"].max() != horizon:
        raise PredictionValidationError("Forecast output does not cover every horizon step.")
    return result


def _metrics_by(predictions: pd.DataFrame, column: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for value, group in predictions.groupby(column, sort=True, observed=True):
        metrics = calculate_metrics(group["actual_demand_mw"], group["predicted_demand_mw"])
        metrics["forecast_bias"] = metrics.pop("bias")
        rows.append({column: value, **metrics})
    return pd.DataFrame(rows)
