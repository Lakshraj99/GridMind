"""Transparent objective coefficient and post-dispatch contribution helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from gridmind.exceptions import BatteryOptimizationError
from gridmind.optimization.contracts import ObjectiveMode, ObjectiveWeights


def objective_coefficients(
    frame: pd.DataFrame,
    *,
    mode: ObjectiveMode,
    weights: ObjectiveWeights,
    degradation_cost_per_mwh: float,
    duration_hours: float,
) -> tuple[
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
    float,
]:
    """Return charge, discharge, and peak coefficients for a linear objective."""
    prices = frame["energy_price"].to_numpy(dtype=float)
    renewable = frame["renewable_forecast_mw"].to_numpy(dtype=float)
    net_load = frame["net_load_before_battery_mw"].to_numpy(dtype=float)
    count = len(frame)
    charge = np.zeros(count)
    discharge = np.zeros(count)
    peak = 0.0
    if mode == "peak_shaving":
        peak = 1.0
        charge += 1e-8 * duration_hours
        discharge += 1e-8 * duration_hours
    elif mode == "energy_arbitrage":
        charge = prices * duration_hours + degradation_cost_per_mwh * duration_hours
        discharge = -prices * duration_hours + degradation_cost_per_mwh * duration_hours
    elif mode == "renewable_utilization":
        scale = max(float(np.max(np.abs(renewable))), 1.0)
        charge = -(renewable / scale) * duration_hours
        charge += degradation_cost_per_mwh * duration_hours / max(degradation_cost_per_mwh, 1.0)
        discharge += degradation_cost_per_mwh * duration_hours / max(degradation_cost_per_mwh, 1.0)
    elif mode == "balanced":
        peak_scale = max(float(np.max(np.abs(net_load))), 1.0)
        price_scale = max(float(np.max(np.abs(prices))) * duration_hours * count, 1.0)
        energy_scale = max(float(np.sum(np.abs(net_load))) * duration_hours, 1.0)
        degradation_scale = max(degradation_cost_per_mwh * duration_hours * count, 1.0)
        renewable_scale = max(float(np.max(np.abs(renewable))), 1.0)
        peak = weights.peak / peak_scale
        charge += weights.energy_cost * prices * duration_hours / price_scale
        discharge -= weights.energy_cost * prices * duration_hours / price_scale
        charge -= (
            weights.renewable_utilization
            * (renewable / renewable_scale)
            * duration_hours
            / energy_scale
        )
        charge += (
            weights.degradation * degradation_cost_per_mwh * duration_hours / degradation_scale
        )
        discharge += (
            weights.degradation * degradation_cost_per_mwh * duration_hours / degradation_scale
        )
    else:  # pragma: no cover - protected by public Literal/CLI enum
        raise BatteryOptimizationError(f"Unsupported dispatch objective '{mode}'.")
    return charge, discharge, peak


def objective_breakdown(
    schedule: pd.DataFrame,
    *,
    weights: ObjectiveWeights,
    duration_hours: float,
    terminal_target_mwh: float,
) -> dict[str, float]:
    renewable_share = np.divide(
        schedule["renewable_forecast_mw"].to_numpy(dtype=float),
        np.maximum(schedule["demand_forecast_mw"].to_numpy(dtype=float), 1e-12),
    )
    renewable_share = np.clip(renewable_share, 0.0, 1.0)
    peak = float(schedule["net_load_after_battery_mw"].max())
    energy_cost = float(
        (schedule["net_load_after_battery_mw"] * schedule["energy_price"] * duration_hours).sum()
    )
    degradation = float(
        (
            (schedule["charge_mw"] + schedule["discharge_mw"])
            * schedule["marginal_degradation_cost"]
            * duration_hours
        ).sum()
    )
    renewable_aligned = float(
        (schedule["charge_mw"].to_numpy(dtype=float) * renewable_share * duration_hours).sum()
    )
    terminal_deviation = abs(float(schedule["soc_end_mwh"].iloc[-1]) - terminal_target_mwh)
    return {
        "peak_load_mw": peak,
        "energy_cost": energy_cost,
        "renewable_aligned_charge_mwh": renewable_aligned,
        "degradation_cost": degradation,
        "terminal_soc_deviation_mwh": terminal_deviation,
        "weighted_peak_contribution": weights.peak * peak,
        "weighted_energy_cost_contribution": weights.energy_cost * energy_cost,
        "weighted_renewable_utilization_contribution": (
            -weights.renewable_utilization * renewable_aligned
        ),
        "weighted_degradation_contribution": weights.degradation * degradation,
        "weighted_terminal_soc_contribution": weights.terminal_soc * terminal_deviation,
    }
