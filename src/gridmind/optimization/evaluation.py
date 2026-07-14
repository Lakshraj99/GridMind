"""Battery dispatch and strategy-comparison metrics."""

from __future__ import annotations

import pandas as pd

from gridmind.optimization.contracts import BatterySpecification


def evaluate_dispatch(
    schedule: pd.DataFrame,
    spec: BatterySpecification,
    *,
    duration_hours: float = 1.0,
) -> dict[str, float]:
    original_peak = float(schedule["net_load_before_battery_mw"].max())
    optimized_peak = float(schedule["net_load_after_battery_mw"].max())
    peak_reduction = original_peak - optimized_peak
    original_cost = float(
        (schedule["net_load_before_battery_mw"] * schedule["energy_price"] * duration_hours).sum()
    )
    optimized_cost = float(
        (schedule["net_load_after_battery_mw"] * schedule["energy_price"] * duration_hours).sum()
    )
    charge_energy = float((schedule["charge_mw"] * duration_hours).sum())
    discharge_energy = float((schedule["discharge_mw"] * duration_hours).sum())
    throughput = charge_energy + discharge_energy
    stored_charge = charge_energy * spec.charge_efficiency
    withdrawn = discharge_energy / spec.discharge_efficiency
    self_discharge_loss = float(
        (
            schedule["soc_start_mwh"] * (1.0 - (1.0 - spec.self_discharge_rate) ** duration_hours)
        ).sum()
    )
    losses = max(0.0, charge_energy - stored_charge + withdrawn - discharge_energy)
    losses += self_discharge_loss
    degradation = throughput * spec.degradation_cost_per_mwh
    renewable_share = (
        schedule["renewable_forecast_mw"] / schedule["demand_forecast_mw"].clip(lower=1e-12)
    ).clip(0.0, 1.0)
    return {
        "original_peak_load_mw": original_peak,
        "optimized_peak_load_mw": optimized_peak,
        "absolute_peak_reduction_mw": peak_reduction,
        "percentage_peak_reduction": (
            100.0 * peak_reduction / original_peak if original_peak else 0.0
        ),
        "energy_imported_mwh": float(
            (schedule["net_load_after_battery_mw"].clip(lower=0) * duration_hours).sum()
        ),
        "original_energy_cost": original_cost,
        "energy_cost": optimized_cost,
        "energy_cost_savings": original_cost - optimized_cost,
        "renewable_energy_charged_mwh": float(
            (schedule["charge_mw"] * renewable_share * duration_hours).sum()
        ),
        "charge_energy_mwh": charge_energy,
        "discharge_energy_mwh": discharge_energy,
        "battery_losses_mwh": losses,
        "equivalent_full_cycles": throughput / (2.0 * spec.capacity_mwh),
        "total_throughput_mwh": throughput,
        "degradation_cost": degradation,
        "terminal_soc_deviation_mwh": abs(
            float(schedule["soc_end_mwh"].iloc[-1]) - spec.terminal_soc_target_mwh
        ),
        "soc_violations": float(
            (
                (schedule["soc_end_mwh"] < spec.effective_min_soc_mwh - 1e-5)
                | (schedule["soc_end_mwh"] > spec.max_soc_mwh + 1e-5)
            ).sum()
        ),
        "power_violations": float(
            (
                (schedule["charge_mw"] > spec.max_charge_mw + 1e-5)
                | (schedule["discharge_mw"] > spec.max_discharge_mw + 1e-5)
                | ((schedule["charge_mw"] > 1e-5) & (schedule["discharge_mw"] > 1e-5))
            ).sum()
        ),
    }


def compare_strategies(
    schedules: dict[str, pd.DataFrame],
    spec: BatterySpecification,
    *,
    duration_hours: float = 1.0,
) -> pd.DataFrame:
    rows = [
        {"strategy": name, **evaluate_dispatch(frame, spec, duration_hours=duration_hours)}
        for name, frame in schedules.items()
    ]
    return pd.DataFrame(rows).sort_values("strategy").reset_index(drop=True)
