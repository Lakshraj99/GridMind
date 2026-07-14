"""Transparent no-battery and rule-based dispatch baselines."""

from __future__ import annotations

import json
from datetime import UTC
from typing import Any

import pandas as pd

from gridmind.optimization.battery import soc_transition, validate_dispatch_physics
from gridmind.optimization.contracts import (
    DISPATCH_POINT_COLUMNS,
    BatterySpecification,
    validate_dispatch_input,
)


def no_battery_baseline(
    frame: pd.DataFrame,
    spec: BatterySpecification,
    *,
    duration_hours: float = 1.0,
    run_id: str = "no-battery",
) -> pd.DataFrame:
    source = validate_dispatch_input(frame, horizon=len(frame), step_hours=duration_hours)
    return _sequential_schedule(
        source,
        spec,
        duration_hours=duration_hours,
        run_id=run_id,
        policy=lambda _row, _low, _high: (0.0, 0.0),
    )


def rule_based_baseline(
    frame: pd.DataFrame,
    spec: BatterySpecification,
    *,
    duration_hours: float = 1.0,
    low_load_quantile: float = 0.25,
    high_load_quantile: float = 0.75,
    run_id: str = "rule-based",
) -> pd.DataFrame:
    source = validate_dispatch_input(frame, horizon=len(frame), step_hours=duration_hours)
    low = float(source["net_load_before_battery_mw"].quantile(low_load_quantile))
    high = float(source["net_load_before_battery_mw"].quantile(high_load_quantile))

    def policy(row: pd.Series, low_threshold: float, high_threshold: float) -> tuple[float, float]:
        net_load = float(row["net_load_before_battery_mw"])
        renewable_surplus = max(
            0.0, float(row["renewable_forecast_mw"] - row["demand_forecast_mw"])
        )
        if renewable_surplus > 0 or net_load <= low_threshold:
            return spec.max_charge_mw, 0.0
        if net_load > high_threshold:
            return 0.0, spec.max_discharge_mw
        return 0.0, 0.0

    return _sequential_schedule(
        source,
        spec,
        duration_hours=duration_hours,
        run_id=run_id,
        policy=policy,
        low=low,
        high=high,
    )


def _sequential_schedule(
    source: pd.DataFrame,
    spec: BatterySpecification,
    *,
    duration_hours: float,
    run_id: str,
    policy: Any,
    low: float = 0.0,
    high: float = 0.0,
) -> pd.DataFrame:
    soc = spec.initial_soc_mwh
    created = pd.Timestamp.now(tz=UTC)
    rows: list[dict[str, Any]] = []
    decay = (1.0 - spec.self_discharge_rate) ** duration_hours
    for _, source_row in source.iterrows():
        requested_charge, requested_discharge = policy(source_row, low, high)
        max_charge_by_soc = max(
            0.0,
            (spec.max_soc_mwh - decay * soc) / (spec.charge_efficiency * duration_hours),
        )
        max_discharge_by_soc = max(
            0.0,
            (decay * soc - spec.effective_min_soc_mwh) * spec.discharge_efficiency / duration_hours,
        )
        charge = min(float(requested_charge), spec.max_charge_mw, max_charge_by_soc)
        discharge = min(float(requested_discharge), spec.max_discharge_mw, max_discharge_by_soc)
        soc_end = soc_transition(soc, charge, discharge, spec, duration_hours)
        metadata = json.loads(str(source_row["metadata_json"]))
        metadata.update(
            {
                "baseline": run_id,
                "low_load_threshold_mw": low,
                "high_load_threshold_mw": high,
            }
        )
        mode = "charge" if charge > 1e-7 else "discharge" if discharge > 1e-7 else "idle"
        rows.append(
            {
                "dispatch_run_id": run_id,
                "region": source_row["region"],
                "battery_id": spec.battery_id,
                "forecast_origin": source_row["forecast_origin"],
                "timestamp_utc": source_row["timestamp_utc"],
                "forecast_step": int(source_row["forecast_step"]),
                "demand_forecast_mw": float(source_row["demand_forecast_mw"]),
                "solar_forecast_mw": float(source_row["solar_forecast_mw"]),
                "wind_forecast_mw": float(source_row["wind_forecast_mw"]),
                "renewable_forecast_mw": float(source_row["renewable_forecast_mw"]),
                "net_load_before_battery_mw": float(source_row["net_load_before_battery_mw"]),
                "charge_mw": charge,
                "discharge_mw": discharge,
                "net_battery_power_mw": discharge - charge,
                "soc_start_mwh": soc,
                "soc_end_mwh": soc_end,
                "net_load_after_battery_mw": float(
                    source_row["net_load_before_battery_mw"] + charge - discharge
                ),
                "energy_price": float(source_row["energy_price"]),
                "marginal_degradation_cost": spec.degradation_cost_per_mwh,
                "operating_mode": mode,
                "solver_status": "feasible",
                "created_at_utc": created,
                "metadata_json": json.dumps(metadata, sort_keys=True),
            }
        )
        soc = soc_end
    schedule = pd.DataFrame(rows, columns=DISPATCH_POINT_COLUMNS)
    validate_dispatch_physics(
        schedule, spec, duration_hours=duration_hours, terminal_required=False
    )
    return schedule
