"""SciPy/HiGHS mixed-integer battery dispatch solver."""

from __future__ import annotations

import json
import time
from datetime import UTC
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, milp

from gridmind.exceptions import BatteryOptimizationError
from gridmind.optimization.battery import validate_dispatch_physics
from gridmind.optimization.constraints import physical_constraints
from gridmind.optimization.contracts import (
    DISPATCH_POINT_COLUMNS,
    BatterySpecification,
    DispatchOptimizationResult,
    ObjectiveMode,
    ObjectiveWeights,
    SolverDiagnostics,
    SolverStatus,
    deterministic_dispatch_id,
    validate_dispatch_input,
)
from gridmind.optimization.objectives import objective_breakdown, objective_coefficients


def optimize_battery_dispatch(
    frame: pd.DataFrame,
    spec: BatterySpecification,
    *,
    objective_mode: ObjectiveMode = "peak_shaving",
    weights: ObjectiveWeights | None = None,
    duration_hours: float = 1.0,
    timeout_seconds: float = 60.0,
    robust: bool = False,
    demand_uplift_pct: float = 0.03,
    renewable_reduction_pct: float = 0.10,
    extra_reserve_pct: float = 0.05,
    prior_daily_throughput_mwh: dict[str, float] | None = None,
) -> DispatchOptimizationResult:
    selected_weights = weights or ObjectiveWeights()
    source = validate_dispatch_input(frame, horizon=len(frame), step_hours=duration_hours)
    if objective_mode in {"energy_arbitrage", "balanced"} and source["energy_price"].isna().any():
        raise BatteryOptimizationError("Energy prices are required for the selected objective.")
    adjusted, adjusted_spec = _apply_robust_margins(
        source,
        spec,
        enabled=robust,
        demand_uplift_pct=demand_uplift_pct,
        renewable_reduction_pct=renewable_reduction_pct,
        extra_reserve_pct=extra_reserve_pct,
    )
    count = len(adjusted)
    charge_slice = slice(0, count)
    discharge_slice = slice(count, 2 * count)
    soc_slice = slice(2 * count, 3 * count)
    binary_slice = slice(3 * count, 4 * count)
    peak_index = 4 * count
    variable_count = peak_index + 1
    charge_coeff, discharge_coeff, peak_coeff = objective_coefficients(
        adjusted,
        mode=objective_mode,
        weights=selected_weights,
        degradation_cost_per_mwh=adjusted_spec.degradation_cost_per_mwh,
        duration_hours=duration_hours,
    )
    coefficients = np.zeros(variable_count)
    coefficients[charge_slice] = charge_coeff
    coefficients[discharge_slice] = discharge_coeff
    coefficients[peak_index] = peak_coeff
    lower = np.full(variable_count, -np.inf)
    upper = np.full(variable_count, np.inf)
    lower[charge_slice] = 0.0
    upper[charge_slice] = adjusted_spec.max_charge_mw
    lower[discharge_slice] = 0.0
    upper[discharge_slice] = adjusted_spec.max_discharge_mw
    lower[soc_slice] = adjusted_spec.effective_min_soc_mwh
    upper[soc_slice] = adjusted_spec.max_soc_mwh
    lower[binary_slice] = 0.0
    upper[binary_slice] = 1.0
    integrality = np.zeros(variable_count, dtype=int)
    integrality[binary_slice] = 1
    started = time.perf_counter()
    try:
        result = milp(
            coefficients,
            integrality=integrality,
            bounds=Bounds(lower, upper),
            constraints=physical_constraints(
                adjusted,
                adjusted_spec,
                duration_hours=duration_hours,
                prior_daily_throughput_mwh=prior_daily_throughput_mwh,
            ),
            options={"time_limit": timeout_seconds, "presolve": True},
        )
    except Exception as exc:
        raise BatteryOptimizationError(f"HiGHS dispatch solver failed: {exc}") from exc
    elapsed = time.perf_counter() - started
    status = _solver_status(int(result.status), result.x is not None)
    gap = _optional_float(getattr(result, "mip_gap", None))
    objective_value = _optional_float(result.fun)
    if result.x is None or status in {"infeasible", "unbounded", "error", "timeout"}:
        raise BatteryOptimizationError(
            f"Battery dispatch did not produce a feasible solution: {status}; {result.message}"
        )
    run_id = deterministic_dispatch_id(
        adjusted_spec, adjusted, objective_mode, selected_weights, robust=robust
    )
    schedule = _schedule_from_solution(
        adjusted,
        adjusted_spec,
        np.asarray(result.x, dtype=float),
        run_id=run_id,
        status=status,
        duration_hours=duration_hours,
        robust=robust,
    )
    physical = validate_dispatch_physics(schedule, adjusted_spec, duration_hours=duration_hours)
    breakdown = objective_breakdown(
        schedule,
        weights=selected_weights,
        duration_hours=duration_hours,
        terminal_target_mwh=adjusted_spec.terminal_soc_target_mwh,
    )
    breakdown.update(
        throughput_mwh=float(physical["throughput_mwh"]),
        equivalent_full_cycles=float(physical["equivalent_full_cycles"]),
    )
    diagnostics = SolverDiagnostics(
        "scipy.optimize.milp/HiGHS",
        status,
        objective_value,
        elapsed,
        gap,
        bool(physical["valid"]),
        str(result.message),
    )
    lineage = {
        "forecast_origin": adjusted["forecast_origin"].iloc[0].isoformat(),
        "input_metadata": [json.loads(value) for value in adjusted["metadata_json"]],
        "robust_mode": robust,
        "demand_uplift_pct": demand_uplift_pct if robust else 0.0,
        "renewable_reduction_pct": renewable_reduction_pct if robust else 0.0,
        "extra_reserve_pct": extra_reserve_pct if robust else 0.0,
        "prior_daily_throughput_mwh": prior_daily_throughput_mwh or {},
    }
    return DispatchOptimizationResult(run_id, schedule, breakdown, diagnostics, lineage)


def _apply_robust_margins(
    frame: pd.DataFrame,
    spec: BatterySpecification,
    *,
    enabled: bool,
    demand_uplift_pct: float,
    renewable_reduction_pct: float,
    extra_reserve_pct: float,
) -> tuple[pd.DataFrame, BatterySpecification]:
    adjusted = frame.copy()
    if not enabled:
        return adjusted, spec
    if min(demand_uplift_pct, renewable_reduction_pct, extra_reserve_pct) < 0:
        raise BatteryOptimizationError("Robust safety margins must be non-negative.")
    original_demand = adjusted["demand_forecast_mw"].copy()
    original_renewable = adjusted["renewable_forecast_mw"].copy()
    adjusted["demand_forecast_mw"] *= 1.0 + demand_uplift_pct
    adjusted["solar_forecast_mw"] *= 1.0 - renewable_reduction_pct
    adjusted["wind_forecast_mw"] *= 1.0 - renewable_reduction_pct
    adjusted["renewable_forecast_mw"] *= 1.0 - renewable_reduction_pct
    adjusted["net_load_before_battery_mw"] = (
        adjusted["demand_forecast_mw"] - adjusted["renewable_forecast_mw"]
    )
    for position, index in enumerate(adjusted.index):
        metadata = json.loads(str(adjusted.at[index, "metadata_json"]))
        metadata["unadjusted_demand_forecast_mw"] = float(original_demand.iloc[position])
        metadata["unadjusted_renewable_forecast_mw"] = float(original_renewable.iloc[position])
        metadata["robust_adjustments"] = {
            "demand_uplift_pct": demand_uplift_pct,
            "renewable_reduction_pct": renewable_reduction_pct,
            "extra_reserve_pct": extra_reserve_pct,
        }
        adjusted.at[index, "metadata_json"] = json.dumps(metadata, sort_keys=True)
    values = spec.as_dict()
    values["reserve_soc_mwh"] = max(
        spec.reserve_soc_mwh,
        spec.capacity_mwh * extra_reserve_pct + spec.reserve_soc_mwh,
    )
    if float(values["reserve_soc_mwh"]) > spec.max_soc_mwh:
        raise BatteryOptimizationError("Robust reserve margin exceeds maximum battery SOC.")
    return adjusted, BatterySpecification(**values)


def _schedule_from_solution(
    frame: pd.DataFrame,
    spec: BatterySpecification,
    solution: np.ndarray[Any, np.dtype[np.float64]],
    *,
    run_id: str,
    status: SolverStatus,
    duration_hours: float,
    robust: bool,
) -> pd.DataFrame:
    count = len(frame)
    charge = solution[:count]
    discharge = solution[count : 2 * count]
    soc_end = solution[2 * count : 3 * count]
    soc_start = np.concatenate([[spec.initial_soc_mwh], soc_end[:-1]])
    created = pd.Timestamp.now(tz=UTC)
    rows: list[dict[str, Any]] = []
    for position, (_, source) in enumerate(frame.iterrows()):
        charging = float(charge[position])
        discharging = float(discharge[position])
        mode = "charge" if charging > 1e-7 else "discharge" if discharging > 1e-7 else "idle"
        metadata = json.loads(str(source["metadata_json"]))
        metadata.update({"robust_mode": robust, "duration_hours": duration_hours})
        rows.append(
            {
                "dispatch_run_id": run_id,
                "region": source["region"],
                "battery_id": spec.battery_id,
                "forecast_origin": source["forecast_origin"],
                "timestamp_utc": source["timestamp_utc"],
                "forecast_step": int(source["forecast_step"]),
                "demand_forecast_mw": float(source["demand_forecast_mw"]),
                "solar_forecast_mw": float(source["solar_forecast_mw"]),
                "wind_forecast_mw": float(source["wind_forecast_mw"]),
                "renewable_forecast_mw": float(source["renewable_forecast_mw"]),
                "net_load_before_battery_mw": float(source["net_load_before_battery_mw"]),
                "charge_mw": charging,
                "discharge_mw": discharging,
                "net_battery_power_mw": discharging - charging,
                "soc_start_mwh": float(soc_start[position]),
                "soc_end_mwh": float(soc_end[position]),
                "net_load_after_battery_mw": float(
                    source["net_load_before_battery_mw"] + charging - discharging
                ),
                "energy_price": float(source["energy_price"]),
                "marginal_degradation_cost": spec.degradation_cost_per_mwh,
                "operating_mode": mode,
                "solver_status": status,
                "created_at_utc": created,
                "metadata_json": json.dumps(metadata, sort_keys=True),
            }
        )
    return pd.DataFrame(rows, columns=DISPATCH_POINT_COLUMNS)


def _solver_status(status: int, has_solution: bool) -> SolverStatus:
    if status == 0:
        return "optimal"
    if status == 1:
        return "feasible" if has_solution else "timeout"
    if status == 2:
        return "infeasible"
    if status == 3:
        return "unbounded"
    return "error"


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    result = float(np.asarray(value).item())
    return result if np.isfinite(result) else None
