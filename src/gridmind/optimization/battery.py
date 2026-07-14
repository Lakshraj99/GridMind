"""Battery state transitions and post-solve physical validation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gridmind.config import Settings
from gridmind.exceptions import BatteryOptimizationError
from gridmind.optimization.contracts import BatterySpecification


def battery_specification_from_settings(
    settings: Settings, battery_id: str
) -> BatterySpecification:
    return BatterySpecification(
        battery_id=battery_id,
        capacity_mwh=settings.battery_capacity_mwh,
        max_charge_mw=settings.battery_max_charge_mw,
        max_discharge_mw=settings.battery_max_discharge_mw,
        min_soc_mwh=settings.battery_min_soc_mwh,
        max_soc_mwh=settings.battery_max_soc_mwh,
        initial_soc_mwh=settings.battery_initial_soc_mwh,
        terminal_soc_target_mwh=settings.battery_terminal_soc_mwh,
        charge_efficiency=settings.battery_charge_efficiency,
        discharge_efficiency=settings.battery_discharge_efficiency,
        self_discharge_rate=settings.battery_self_discharge_per_hour,
        maximum_daily_cycles=settings.battery_max_equivalent_cycles_per_day,
        degradation_cost_per_mwh=settings.battery_degradation_cost_per_mwh,
        reserve_soc_mwh=max(
            settings.battery_min_soc_mwh,
            settings.battery_capacity_mwh * settings.battery_reserve_soc_pct,
        ),
        metadata={"source": "GridMind Settings", "illustrative_unless_operator_supplied": True},
    )


def soc_transition(
    soc_start_mwh: float,
    charge_mw: float,
    discharge_mw: float,
    spec: BatterySpecification,
    duration_hours: float,
) -> float:
    if duration_hours <= 0:
        raise BatteryOptimizationError("Battery transition duration must be positive.")
    return float(
        soc_start_mwh * (1.0 - spec.self_discharge_rate) ** duration_hours
        + charge_mw * spec.charge_efficiency * duration_hours
        - discharge_mw / spec.discharge_efficiency * duration_hours
    )


def validate_dispatch_physics(
    schedule: pd.DataFrame,
    spec: BatterySpecification,
    *,
    duration_hours: float,
    terminal_required: bool = True,
    tolerance: float = 1e-5,
) -> dict[str, float | int | bool]:
    if schedule.empty:
        raise BatteryOptimizationError("Solved dispatch schedule is empty.")
    power_violations = int(
        (schedule["charge_mw"] < -tolerance).sum()
        + (schedule["discharge_mw"] < -tolerance).sum()
        + (schedule["charge_mw"] > spec.max_charge_mw + tolerance).sum()
        + (schedule["discharge_mw"] > spec.max_discharge_mw + tolerance).sum()
        + ((schedule["charge_mw"] > tolerance) & (schedule["discharge_mw"] > tolerance)).sum()
    )
    effective_min = spec.effective_min_soc_mwh
    soc_violations = int(
        (schedule["soc_end_mwh"] < effective_min - tolerance).sum()
        + (schedule["soc_end_mwh"] > spec.max_soc_mwh + tolerance).sum()
    )
    continuity_violations = 0
    expected_start = spec.initial_soc_mwh
    for row in schedule.itertuples(index=False):
        start_value = float(np.asarray(row.soc_start_mwh).item())
        end_value = float(np.asarray(row.soc_end_mwh).item())
        expected_end = soc_transition(
            expected_start,
            float(np.asarray(row.charge_mw).item()),
            float(np.asarray(row.discharge_mw).item()),
            spec,
            duration_hours,
        )
        continuity_violations += int(abs(start_value - expected_start) > tolerance)
        continuity_violations += int(abs(end_value - expected_end) > tolerance)
        expected_start = end_value
    throughput = float(((schedule["charge_mw"] + schedule["discharge_mw"]) * duration_hours).sum())
    cycles = throughput / (2.0 * spec.capacity_mwh)
    days = pd.to_datetime(schedule["timestamp_utc"], utc=True).dt.floor("D")
    daily = ((schedule["charge_mw"] + schedule["discharge_mw"]) * duration_hours).groupby(
        days
    ).sum() / (2.0 * spec.capacity_mwh)
    cycle_violations = int((daily > spec.maximum_daily_cycles + tolerance).sum())
    terminal_deviation = abs(float(schedule["soc_end_mwh"].iloc[-1]) - spec.terminal_soc_target_mwh)
    terminal_violations = int(terminal_required and terminal_deviation > tolerance)
    valid = not any(
        (
            power_violations,
            soc_violations,
            continuity_violations,
            cycle_violations,
            terminal_violations,
        )
    )
    report: dict[str, float | int | bool] = {
        "valid": valid,
        "power_violations": power_violations,
        "soc_violations": soc_violations,
        "continuity_violations": continuity_violations,
        "cycle_violations": cycle_violations,
        "terminal_violations": terminal_violations,
        "terminal_soc_deviation_mwh": terminal_deviation,
        "throughput_mwh": throughput,
        "equivalent_full_cycles": cycles,
    }
    if not valid:
        raise BatteryOptimizationError(f"Solved dispatch violates battery physics: {report}")
    if not np.isfinite(schedule.select_dtypes(include=["number"]).to_numpy()).all():
        raise BatteryOptimizationError("Solved dispatch contains non-finite numeric values.")
    return report
