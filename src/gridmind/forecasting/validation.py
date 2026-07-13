"""Rolling-origin evaluation without shuffled splits or target leakage."""

from __future__ import annotations

import pandas as pd

from gridmind.forecasting.baselines import BaselineForecaster


def rolling_origin_evaluate(
    frame: pd.DataFrame,
    model: BaselineForecaster,
    *,
    horizon: int = 24,
    windows: int = 3,
    step_size: int = 24,
) -> pd.DataFrame:
    """Return forecasts for chronological expanding-window validation splits."""
    if horizon <= 0 or windows <= 0 or step_size <= 0:
        raise ValueError("Horizon, windows, and step size must all be positive.")
    ordered = frame.sort_values("timestamp_utc", ignore_index=True)
    if ordered["region"].nunique() != 1:
        raise ValueError("Rolling evaluation requires exactly one region.")
    first_validation_start = len(ordered) - horizon - (windows - 1) * step_size
    if first_validation_start <= 0:
        raise ValueError(
            "Not enough observations for the requested rolling validation configuration."
        )

    predictions: list[pd.DataFrame] = []
    for window_id in range(windows):
        validation_start = first_validation_start + window_id * step_size
        validation_end = validation_start + horizon
        training = ordered.iloc[:validation_start].copy()
        validation = ordered.iloc[validation_start:validation_end].copy()
        if validation.empty or training["timestamp_utc"].max() >= validation["timestamp_utc"].min():
            raise ValueError("Training observations must precede validation observations.")
        forecast = model.predict(training, horizon=horizon)
        actual_by_time = validation.set_index("timestamp_utc")["demand_mw"]
        forecast["actual_demand_mw"] = forecast["timestamp_utc"].map(actual_by_time)
        if forecast["actual_demand_mw"].isna().any():
            raise ValueError("Validation timestamps are not contiguous hourly observations.")
        forecast["validation_window"] = window_id + 1
        predictions.append(forecast)
    return pd.concat(predictions, ignore_index=True)
