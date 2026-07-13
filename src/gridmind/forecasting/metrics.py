"""Forecast error metrics and their rolling-window aggregation."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def calculate_metrics(
    actual: pd.Series, predicted: pd.Series, *, mase_scale: float | None = None
) -> dict[str, float]:
    """Calculate MAE, RMSE, WAPE, MASE, and signed forecast bias.

    Lower MAE/RMSE/WAPE/MASE is better. MASE below one beats the scale's naive
    benchmark. Positive bias means over-forecasting; negative bias means under-forecasting.
    Undefined ratio metrics return NaN when their denominator is zero.
    """
    paired = pd.DataFrame({"actual": actual, "predicted": predicted}).dropna()
    if paired.empty:
        return {name: float("nan") for name in ("mae", "rmse", "wape", "mase", "bias")}
    errors = paired["predicted"] - paired["actual"]
    absolute_errors = errors.abs()
    mae = float(absolute_errors.mean())
    rmse = float(math.sqrt((errors**2).mean()))
    denominator = float(paired["actual"].abs().sum())
    wape = float(absolute_errors.sum() / denominator) if denominator != 0 else float("nan")
    if mase_scale is None:
        mase_scale = float(paired["actual"].diff().abs().dropna().mean())
    mase = mae / mase_scale if mase_scale is not None and mase_scale > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "wape": wape, "mase": mase, "bias": float(errors.mean())}


def evaluate_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    """Return overall metrics plus metrics for each rolling validation window."""
    overall = calculate_metrics(predictions["actual_demand_mw"], predictions["predicted_demand_mw"])
    by_window: list[dict[str, float | int]] = []
    for window, group in predictions.groupby("validation_window", sort=True):
        metrics = calculate_metrics(group["actual_demand_mw"], group["predicted_demand_mw"])
        by_window.append({"validation_window": int(str(window)), **metrics})
    return {"overall": overall, "by_window": by_window}
