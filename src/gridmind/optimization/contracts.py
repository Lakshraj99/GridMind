"""Canonical Milestone 5 battery, input, dispatch, and solver contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from gridmind.exceptions import BatteryOptimizationError
from gridmind.time_utils import format_utc_timestamp

ObjectiveMode = Literal["peak_shaving", "energy_arbitrage", "renewable_utilization", "balanced"]
SolverStatus = Literal["optimal", "feasible", "infeasible", "unbounded", "timeout", "error"]

DISPATCH_INPUT_COLUMNS = [
    "region",
    "forecast_origin",
    "timestamp_utc",
    "forecast_step",
    "demand_forecast_mw",
    "solar_forecast_mw",
    "wind_forecast_mw",
    "renewable_forecast_mw",
    "net_load_before_battery_mw",
    "energy_price",
    "metadata_json",
]

DISPATCH_POINT_COLUMNS = [
    "dispatch_run_id",
    "region",
    "battery_id",
    "forecast_origin",
    "timestamp_utc",
    "forecast_step",
    "demand_forecast_mw",
    "solar_forecast_mw",
    "wind_forecast_mw",
    "renewable_forecast_mw",
    "net_load_before_battery_mw",
    "charge_mw",
    "discharge_mw",
    "net_battery_power_mw",
    "soc_start_mwh",
    "soc_end_mwh",
    "net_load_after_battery_mw",
    "energy_price",
    "marginal_degradation_cost",
    "operating_mode",
    "solver_status",
    "created_at_utc",
    "metadata_json",
]


@dataclass(frozen=True)
class BatterySpecification:
    battery_id: str
    capacity_mwh: float
    max_charge_mw: float
    max_discharge_mw: float
    min_soc_mwh: float
    max_soc_mwh: float
    initial_soc_mwh: float
    terminal_soc_target_mwh: float
    charge_efficiency: float
    discharge_efficiency: float
    self_discharge_rate: float
    maximum_daily_cycles: float
    degradation_cost_per_mwh: float
    reserve_soc_mwh: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.battery_id.strip():
            raise BatteryOptimizationError("Battery identifier must not be empty.")
        if min(self.capacity_mwh, self.max_charge_mw, self.max_discharge_mw) <= 0:
            raise BatteryOptimizationError("Battery capacity and power limits must be positive.")
        if not 0 <= self.min_soc_mwh < self.max_soc_mwh <= self.capacity_mwh:
            raise BatteryOptimizationError(
                "Battery SOC bounds must satisfy 0 <= min < max <= capacity."
            )
        for name, value in (
            ("initial", self.initial_soc_mwh),
            ("terminal", self.terminal_soc_target_mwh),
            ("reserve", self.reserve_soc_mwh),
        ):
            if not self.min_soc_mwh <= value <= self.max_soc_mwh:
                raise BatteryOptimizationError(f"Battery {name} SOC must be within SOC bounds.")
        if not 0 < self.charge_efficiency <= 1 or not 0 < self.discharge_efficiency <= 1:
            raise BatteryOptimizationError("Battery efficiencies must be in (0, 1].")
        if not 0 <= self.self_discharge_rate < 1:
            raise BatteryOptimizationError("Battery self-discharge must be in [0, 1).")
        if self.maximum_daily_cycles <= 0 or self.degradation_cost_per_mwh < 0:
            raise BatteryOptimizationError(
                "Cycle limit must be positive and degradation non-negative."
            )

    @property
    def effective_min_soc_mwh(self) -> float:
        return max(self.min_soc_mwh, self.reserve_soc_mwh)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_initial_soc(self, value: float) -> BatterySpecification:
        values = self.as_dict()
        values["initial_soc_mwh"] = value
        return BatterySpecification(**values)


@dataclass(frozen=True)
class ObjectiveWeights:
    peak: float = 1.0
    energy_cost: float = 1.0
    renewable_utilization: float = 0.5
    degradation: float = 1.0
    terminal_soc: float = 10.0

    def __post_init__(self) -> None:
        if min(asdict(self).values()) < 0:
            raise BatteryOptimizationError("Dispatch objective weights must be non-negative.")


@dataclass(frozen=True)
class SolverDiagnostics:
    solver_name: str
    status: SolverStatus
    objective_value: float | None
    solve_time_seconds: float
    optimality_gap: float | None
    constraint_validation_passed: bool
    message: str


@dataclass(frozen=True)
class DispatchOptimizationResult:
    dispatch_run_id: str
    schedule: pd.DataFrame
    objective_breakdown: dict[str, float]
    diagnostics: SolverDiagnostics
    lineage: dict[str, Any]


def validate_dispatch_input(
    frame: pd.DataFrame, *, horizon: int, step_hours: float
) -> pd.DataFrame:
    missing = set(DISPATCH_INPUT_COLUMNS).difference(frame.columns)
    if missing:
        raise BatteryOptimizationError(f"Dispatch input is missing columns: {sorted(missing)}")
    if horizon <= 0 or step_hours <= 0:
        raise BatteryOptimizationError("Dispatch horizon and time step must be positive.")
    result = frame[DISPATCH_INPUT_COLUMNS].copy()
    for column in ("forecast_origin", "timestamp_utc"):
        result[column] = pd.to_datetime(result[column], utc=True, errors="raise")
    if len(result) != horizon:
        raise BatteryOptimizationError(
            f"Dispatch requires {horizon} forecast rows; received {len(result)}."
        )
    if result["region"].nunique() != 1 or result["forecast_origin"].nunique() != 1:
        raise BatteryOptimizationError("Dispatch rows must use one region and one forecast origin.")
    result = result.sort_values("timestamp_utc").reset_index(drop=True)
    expected = pd.date_range(
        result["forecast_origin"].iloc[0] + pd.Timedelta(hours=step_hours),
        periods=horizon,
        freq=pd.Timedelta(hours=step_hours),
        tz="UTC",
    )
    actual = pd.DatetimeIndex(result["timestamp_utc"])
    if not actual.equals(expected):
        missing_times = expected.difference(actual)
        raise BatteryOptimizationError(
            "Dispatch forecast horizon is not complete and contiguous; missing UTC timestamps: "
            f"{[format_utc_timestamp(value) for value in missing_times]}."
        )
    numeric = [
        "demand_forecast_mw",
        "solar_forecast_mw",
        "wind_forecast_mw",
        "renewable_forecast_mw",
        "net_load_before_battery_mw",
        "energy_price",
    ]
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if not np.isfinite(result[numeric].to_numpy(dtype=float)).all():
        raise BatteryOptimizationError("Dispatch forecast values and prices must be finite.")
    if result["forecast_step"].astype(int).tolist() != list(range(1, horizon + 1)):
        raise BatteryOptimizationError("Dispatch forecast steps must be consecutive from one.")
    try:
        result["metadata_json"].map(json.loads)
    except (TypeError, json.JSONDecodeError) as exc:
        raise BatteryOptimizationError("Dispatch input metadata must contain valid JSON.") from exc
    return result


def deterministic_dispatch_id(
    spec: BatterySpecification,
    frame: pd.DataFrame,
    objective_mode: ObjectiveMode,
    weights: ObjectiveWeights,
    *,
    robust: bool,
) -> str:
    material = {
        "battery": spec.as_dict(),
        "region": str(frame["region"].iloc[0]),
        "forecast_origin": format_utc_timestamp(frame["forecast_origin"].iloc[0]),
        "timestamps": [format_utc_timestamp(value) for value in frame["timestamp_utc"]],
        "values": frame[
            [
                "demand_forecast_mw",
                "renewable_forecast_mw",
                "net_load_before_battery_mw",
                "energy_price",
            ]
        ]
        .round(9)
        .to_dict(orient="records"),
        "objective": objective_mode,
        "weights": asdict(weights),
        "robust": robust,
    }
    payload = json.dumps(material, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:32]
