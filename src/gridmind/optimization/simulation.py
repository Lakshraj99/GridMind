"""Forecast alignment and leakage-safe rolling-horizon dispatch simulation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from gridmind.exceptions import BatteryOptimizationError
from gridmind.optimization.contracts import (
    BatterySpecification,
    ObjectiveMode,
    ObjectiveWeights,
    validate_dispatch_input,
)
from gridmind.optimization.solver import optimize_battery_dispatch
from gridmind.time_utils import format_utc_timestamp, to_utc_timestamp

SimulationMode = Literal["forecast_based", "oracle"]


@dataclass(frozen=True)
class RollingSimulationResult:
    applied_dispatch: pd.DataFrame
    horizon_results: pd.DataFrame
    successful_optimizations: int
    solver_failures: int
    mode: SimulationMode


def build_dispatch_input(
    forecasts: pd.DataFrame,
    *,
    region: str,
    forecast_origin: object,
    horizon: int,
    step_hours: float = 1.0,
    energy_prices: pd.Series | None = None,
    fallback_price: float | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    origin = to_utc_timestamp(forecast_origin)
    required = {
        "region",
        "target",
        "forecast_origin",
        "timestamp_utc",
        "forecast_step",
        "predicted_value",
        "model_name",
        "model_version",
        "run_id",
        "weather_mode",
    }
    missing = required.difference(forecasts.columns)
    if missing:
        raise BatteryOptimizationError(f"Target forecasts are missing columns: {sorted(missing)}")
    source = forecasts.copy()
    source["forecast_origin"] = pd.to_datetime(source["forecast_origin"], utc=True)
    source["timestamp_utc"] = pd.to_datetime(source["timestamp_utc"], utc=True)
    source = source.loc[(source["region"] == region) & (source["forecast_origin"] == origin)]
    if source.empty:
        raise BatteryOptimizationError(
            f"No target forecasts exist for {region} at {format_utc_timestamp(origin)}."
        )
    lineage: dict[str, object] = {}
    selected_parts: list[pd.DataFrame] = []
    for target, group in source.groupby("target", sort=True):
        identities = group[
            ["model_name", "model_version", "run_id", "weather_mode"]
        ].drop_duplicates()
        if len(identities) != 1:
            raise BatteryOptimizationError(
                f"Target {target} has incompatible model lineage at the requested origin."
            )
        identity = identities.iloc[0].to_dict()
        lineage[str(target)] = {str(key): str(value) for key, value in identity.items()}
        selected_parts.append(group.sort_values("timestamp_utc").head(horizon))
    selected = pd.concat(selected_parts, ignore_index=True)
    pivot = selected.pivot(index="timestamp_utc", columns="target", values="predicted_value")
    expected = pd.date_range(
        origin + pd.Timedelta(hours=step_hours),
        periods=horizon,
        freq=pd.Timedelta(hours=step_hours),
        tz="UTC",
    )
    pivot = pivot.reindex(expected)
    if "net_load_mw" not in pivot and "demand_mw" not in pivot:
        raise BatteryOptimizationError("Dispatch requires demand or net-load forecasts.")
    solar = pivot.get("solar_generation_mw", pd.Series(0.0, index=expected)).astype(float)
    wind = pivot.get("wind_generation_mw", pd.Series(0.0, index=expected)).astype(float)
    renewable = pivot.get("total_renewable_generation_mw", solar + wind).astype(float)
    if "demand_mw" in pivot:
        demand = pivot["demand_mw"].astype(float)
    else:
        demand = pivot["net_load_mw"].astype(float) + renewable
    net_load = pivot["net_load_mw"].astype(float) if "net_load_mw" in pivot else demand - renewable
    if energy_prices is not None:
        prices = pd.to_numeric(energy_prices.reindex(expected), errors="coerce")
    elif fallback_price is not None:
        prices = pd.Series(float(fallback_price), index=expected)
    else:
        prices = pd.Series(0.0, index=expected)
    input_frame = pd.DataFrame(
        {
            "region": region,
            "forecast_origin": origin,
            "timestamp_utc": expected,
            "forecast_step": range(1, horizon + 1),
            "demand_forecast_mw": demand.to_numpy(),
            "solar_forecast_mw": solar.to_numpy(),
            "wind_forecast_mw": wind.to_numpy(),
            "renewable_forecast_mw": renewable.to_numpy(),
            "net_load_before_battery_mw": net_load.to_numpy(),
            "energy_price": prices.to_numpy(),
            "metadata_json": json.dumps({"forecast_lineage": lineage}, sort_keys=True),
        }
    )
    if not np.isfinite(input_frame.select_dtypes(include=["number"]).to_numpy()).all():
        missing_times = input_frame.loc[
            input_frame.select_dtypes(include=["number"]).isna().any(axis=1), "timestamp_utc"
        ]
        raise BatteryOptimizationError(
            "Target forecast horizon is incomplete at UTC timestamps: "
            f"{[format_utc_timestamp(value) for value in missing_times]}."
        )
    return validate_dispatch_input(input_frame, horizon=horizon, step_hours=step_hours), lineage


def rolling_horizon_simulation(
    forecast_inputs: pd.DataFrame,
    spec: BatterySpecification,
    *,
    horizon: int,
    objective_mode: ObjectiveMode,
    mode: SimulationMode = "forecast_based",
    weights: ObjectiveWeights | None = None,
    duration_hours: float = 1.0,
    timeout_seconds: float = 60.0,
    oracle_actuals: pd.DataFrame | None = None,
    realized_actuals: pd.DataFrame | None = None,
) -> RollingSimulationResult:
    if mode == "oracle" and oracle_actuals is None:
        raise BatteryOptimizationError("Oracle simulation requires explicitly supplied actuals.")
    source = forecast_inputs.copy()
    source["forecast_origin"] = pd.to_datetime(source["forecast_origin"], utc=True)
    source["timestamp_utc"] = pd.to_datetime(source["timestamp_utc"], utc=True)
    origins = pd.DatetimeIndex(sorted(source["forecast_origin"].unique()))
    if len(origins) > 1:
        expected_origins = pd.date_range(
            origins.min(), origins.max(), freq=pd.Timedelta(hours=duration_hours), tz="UTC"
        )
        if not origins.equals(expected_origins):
            missing = expected_origins.difference(origins)
            raise BatteryOptimizationError(
                "Rolling simulation is missing forecast origins: "
                f"{[format_utc_timestamp(value) for value in missing]}."
            )
    actuals = oracle_actuals.copy() if oracle_actuals is not None else None
    if actuals is not None:
        actuals["timestamp_utc"] = pd.to_datetime(actuals["timestamp_utc"], utc=True)
    realized = realized_actuals.copy() if realized_actuals is not None else None
    if realized is not None:
        realized["timestamp_utc"] = pd.to_datetime(realized["timestamp_utc"], utc=True)
    soc = spec.initial_soc_mwh
    daily_throughput: dict[str, float] = {}
    applied: list[pd.DataFrame] = []
    horizons: list[dict[str, object]] = []
    failures = 0
    for origin in origins:
        horizon_frame = source.loc[source["forecast_origin"] == origin].copy()
        horizon_frame = validate_dispatch_input(
            horizon_frame, horizon=horizon, step_hours=duration_hours
        )
        if mode == "oracle":
            horizon_frame = _replace_with_oracle(horizon_frame, actuals)
        rolling_spec = spec.with_initial_soc(soc)
        try:
            result = optimize_battery_dispatch(
                horizon_frame,
                rolling_spec,
                objective_mode=objective_mode,
                weights=weights,
                duration_hours=duration_hours,
                timeout_seconds=timeout_seconds,
                prior_daily_throughput_mwh=daily_throughput,
            )
        except BatteryOptimizationError:
            failures += 1
            raise
        first = result.schedule.iloc[[0]].copy()
        if mode == "forecast_based" and realized is not None:
            first = apply_realized_actuals(first, realized)
        metadata = json.loads(str(first.iloc[0]["metadata_json"]))
        metadata.update(
            {
                "simulation_mode": mode,
                "oracle_non_deployable": mode == "oracle",
                "applied_first_action_only": True,
            }
        )
        first.loc[:, "metadata_json"] = json.dumps(metadata, sort_keys=True)
        applied.append(first)
        soc = float(first["soc_end_mwh"].iloc[0])
        day_key = pd.Timestamp(first["timestamp_utc"].iloc[0]).strftime("%Y-%m-%d")
        daily_throughput[day_key] = daily_throughput.get(day_key, 0.0) + float(
            (first["charge_mw"].iloc[0] + first["discharge_mw"].iloc[0]) * duration_hours
        )
        horizons.append(
            {
                "forecast_origin": origin,
                "dispatch_run_id": result.dispatch_run_id,
                "solver_status": result.diagnostics.status,
                "objective_value": result.diagnostics.objective_value,
                "solve_time_seconds": result.diagnostics.solve_time_seconds,
                "applied_timestamp_utc": first["timestamp_utc"].iloc[0],
                "soc_end_mwh": soc,
                "simulation_mode": mode,
                "oracle_non_deployable": mode == "oracle",
            }
        )
    return RollingSimulationResult(
        pd.concat(applied, ignore_index=True) if applied else pd.DataFrame(),
        pd.DataFrame(horizons),
        len(horizons),
        failures,
        mode,
    )


def _replace_with_oracle(frame: pd.DataFrame, actuals: pd.DataFrame | None) -> pd.DataFrame:
    if actuals is None:
        raise BatteryOptimizationError("Oracle actuals were not supplied.")
    columns = [
        column
        for column in (
            "demand_forecast_mw",
            "solar_forecast_mw",
            "wind_forecast_mw",
            "renewable_forecast_mw",
            "net_load_before_battery_mw",
        )
        if column in actuals
    ]
    aligned = frame[["timestamp_utc"]].merge(
        actuals[["timestamp_utc", *columns]], on="timestamp_utc", how="left", validate="one_to_one"
    )
    if aligned[columns].isna().any().any():
        raise BatteryOptimizationError(
            "Oracle actuals do not cover the complete optimization horizon."
        )
    result = frame.copy()
    for column in columns:
        result[column] = aligned[column].to_numpy()
    for index in result.index:
        metadata = json.loads(str(result.at[index, "metadata_json"]))
        metadata["oracle_non_deployable"] = True
        result.at[index, "metadata_json"] = json.dumps(metadata, sort_keys=True)
    return result


def apply_realized_actuals(schedule: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """Apply already-selected actions to realized values without changing the decision."""
    columns = [
        "demand_forecast_mw",
        "solar_forecast_mw",
        "wind_forecast_mw",
        "renewable_forecast_mw",
        "net_load_before_battery_mw",
    ]
    missing = set(["timestamp_utc", *columns]).difference(actuals.columns)
    if missing:
        raise BatteryOptimizationError(
            f"Realized dispatch evaluation is missing: {sorted(missing)}"
        )
    result = schedule.copy()
    aligned = result[["timestamp_utc"]].merge(
        actuals[["timestamp_utc", *columns]],
        on="timestamp_utc",
        how="left",
        validate="many_to_one",
    )
    if aligned[columns].isna().any().any():
        raise BatteryOptimizationError(
            "Realized observations do not cover applied dispatch timestamps."
        )
    for position, index in enumerate(result.index):
        metadata = json.loads(str(result.at[index, "metadata_json"]))
        metadata["optimization_forecast"] = {
            column: _number(result.at[index, column]) for column in columns
        }
        metadata["evaluated_on_realized_observation"] = True
        for column in columns:
            result.at[index, column] = _number(aligned.iloc[position][column])
        result.at[index, "net_load_after_battery_mw"] = (
            _number(result.at[index, "net_load_before_battery_mw"])
            + _number(result.at[index, "charge_mw"])
            - _number(result.at[index, "discharge_mw"])
        )
        result.at[index, "metadata_json"] = json.dumps(metadata, sort_keys=True)
    return result


def _number(value: object) -> float:
    return float(np.asarray(value).item())
