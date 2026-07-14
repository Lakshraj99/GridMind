"""Leakage-safe rolling battery dispatch backtesting pipeline."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC
from pathlib import Path
from typing import Literal

import mlflow
import pandas as pd

from gridmind.config import Settings
from gridmind.data.storage import DuckDBStorage, write_json_report
from gridmind.data.target_storage import TargetForecastStorage
from gridmind.exceptions import BatteryOptimizationError
from gridmind.mlflow_config import initialize_mlflow
from gridmind.optimization.baselines import no_battery_baseline, rule_based_baseline
from gridmind.optimization.battery import battery_specification_from_settings
from gridmind.optimization.contracts import BatterySpecification, ObjectiveMode, ObjectiveWeights
from gridmind.optimization.evaluation import compare_strategies, evaluate_dispatch
from gridmind.optimization.simulation import (
    RollingSimulationResult,
    SimulationMode,
    apply_realized_actuals,
    build_dispatch_input,
    rolling_horizon_simulation,
)
from gridmind.optimization.storage import BatteryDispatchStorage
from gridmind.renewables.storage import RenewableStorage
from gridmind.time_utils import to_utc_timestamp


@dataclass(frozen=True)
class BatteryBacktestResult:
    backtest_run_id: str
    rolling: RollingSimulationResult
    strategy_comparison: pd.DataFrame
    artifact_dir: Path
    mlflow_run_id: str | None
    duckdb_run_rows: int
    duckdb_metric_rows: int


def run_battery_backtest(
    settings: Settings,
    *,
    region: str,
    battery_id: str,
    start_date: object,
    end_date: object,
    objective_mode: ObjectiveMode = "balanced",
    mode: SimulationMode = "forecast_based",
    horizon: int | None = None,
    model_alias: str = "champion",
    forecast_inputs: pd.DataFrame | None = None,
    target_forecasts: pd.DataFrame | None = None,
    oracle_actuals: pd.DataFrame | None = None,
    historical_actuals: pd.DataFrame | None = None,
    energy_prices: pd.Series | None = None,
    battery: BatterySpecification | None = None,
    mlflow_enabled: bool | None = None,
    artifact_root: Path = Path("artifacts/battery_backtests"),
) -> BatteryBacktestResult:
    if not settings.battery_optimization_enabled:
        raise BatteryOptimizationError("Battery optimization is disabled by configuration.")
    selected_horizon = horizon or settings.dispatch_horizon_hours
    start = to_utc_timestamp(start_date)
    end = to_utc_timestamp(end_date)
    if end < start:
        raise BatteryOptimizationError("Battery backtest end must not precede its start.")
    if (
        objective_mode in {"energy_arbitrage", "balanced"}
        and energy_prices is None
        and settings.fallback_energy_price_per_mwh is None
    ):
        raise BatteryOptimizationError(
            "Backtest energy prices require input data or FALLBACK_ENERGY_PRICE_PER_MWH."
        )
    canonical = (
        forecast_inputs.copy()
        if forecast_inputs is not None
        else _build_multi_origin_inputs(
            target_forecasts
            if target_forecasts is not None
            else TargetForecastStorage(settings.duckdb_path).read(),
            region=region,
            start=start,
            end=end,
            horizon=selected_horizon,
            step_hours=settings.dispatch_step_hours,
            energy_prices=energy_prices,
            fallback_price=settings.fallback_energy_price_per_mwh,
        )
    )
    canonical["forecast_origin"] = pd.to_datetime(canonical["forecast_origin"], utc=True)
    canonical = canonical.loc[canonical["forecast_origin"].between(start, end)].copy()
    if canonical.empty:
        raise BatteryOptimizationError("No aligned forecast origins exist for the backtest range.")
    actuals = historical_actuals
    if actuals is None and forecast_inputs is None:
        actuals = _load_historical_actuals(
            settings,
            region,
            start + pd.Timedelta(hours=settings.dispatch_step_hours),
            end + pd.Timedelta(hours=selected_horizon * settings.dispatch_step_hours),
        )
    if oracle_actuals is not None:
        actuals = oracle_actuals
    spec = battery or battery_specification_from_settings(settings, battery_id)
    weights = ObjectiveWeights(
        peak=settings.peak_shaving_weight,
        energy_cost=settings.energy_cost_weight,
        renewable_utilization=settings.renewable_utilization_weight,
        degradation=settings.degradation_weight,
        terminal_soc=settings.terminal_soc_penalty_weight,
    )
    rolling = rolling_horizon_simulation(
        canonical,
        spec,
        horizon=selected_horizon,
        objective_mode=objective_mode,
        mode=mode,
        weights=weights,
        duration_hours=settings.dispatch_step_hours,
        timeout_seconds=settings.dispatch_solver_timeout_seconds,
        oracle_actuals=actuals if mode == "oracle" else None,
        realized_actuals=actuals,
    )
    no_battery = _rolling_baseline(
        canonical, spec, selected_horizon, "no_battery", settings, actuals
    )
    rule_based = _rolling_baseline(
        canonical, spec, selected_horizon, "rule_based", settings, actuals
    )
    optimized_name = (
        "optimized_oracle_non_deployable" if mode == "oracle" else "optimized_forecast_based"
    )
    schedules = {
        "no_battery": no_battery,
        "rule_based": rule_based,
        optimized_name: rolling.applied_dispatch,
    }
    comparison = compare_strategies(schedules, spec, duration_hours=settings.dispatch_step_hours)
    run_id = _backtest_id(region, battery_id, start, end, objective_mode, mode, model_alias)
    artifact_dir = artifact_root / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    all_points = pd.concat(
        [frame.assign(strategy=name) for name, frame in schedules.items()], ignore_index=True
    )
    all_points.to_parquet(artifact_dir / "dispatch_points.parquet", index=False)
    rolling.horizon_results.to_parquet(artifact_dir / "horizon_results.parquet", index=False)
    comparison.to_csv(artifact_dir / "strategy_comparison.csv", index=False)
    optimized_metrics = evaluate_dispatch(
        rolling.applied_dispatch, spec, duration_hours=settings.dispatch_step_hours
    )
    overall = {
        **optimized_metrics,
        "evaluated_horizons": rolling.successful_optimizations,
        "successful_optimizations": rolling.successful_optimizations,
        "solver_failures": rolling.solver_failures,
        "solver_success_rate": (
            rolling.successful_optimizations
            / max(rolling.successful_optimizations + rolling.solver_failures, 1)
        ),
        "solver_runtime_seconds": float(rolling.horizon_results["solve_time_seconds"].sum()),
        "infeasible_run_count": rolling.solver_failures,
        "simulation_mode": mode,
        "oracle_non_deployable": mode == "oracle",
    }
    write_json_report(overall, artifact_dir / "overall_metrics.json")
    _daily_metrics(rolling.applied_dispatch, spec, settings.dispatch_step_hours).to_csv(
        artifact_dir / "daily_metrics.csv", index=False
    )
    pd.DataFrame(columns=["forecast_origin", "solver_status", "failure_reason"]).to_csv(
        artifact_dir / "solver_failures.csv", index=False
    )
    configuration = {
        "battery": spec.as_dict(),
        "objective_mode": objective_mode,
        "weights": asdict(weights),
        "mode": mode,
        "oracle_non_deployable": mode == "oracle",
        "model_alias": model_alias,
        "horizon": selected_horizon,
    }
    write_json_report(configuration, artifact_dir / "configuration.json")
    enabled = settings.mlflow_enabled if mlflow_enabled is None else mlflow_enabled
    mlflow_run_id = _log_backtest_mlflow(
        settings,
        artifact_dir,
        run_id,
        region,
        battery_id,
        objective_mode,
        mode,
        overall,
        enabled=enabled,
    )
    run_record = {
        "backtest_run_id": run_id,
        "region": region,
        "battery_id": battery_id,
        "objective_mode": objective_mode,
        "simulation_mode": mode,
        "start_utc": start,
        "end_utc": end,
        "evaluated_horizons": rolling.successful_optimizations,
        "successful_optimizations": rolling.successful_optimizations,
        "solver_failures": rolling.solver_failures,
        "created_at_utc": pd.Timestamp.now(tz=UTC),
        "configuration_json": json.dumps(configuration, sort_keys=True, default=str),
        "artifact_path": str(artifact_dir),
        "mlflow_run_id": mlflow_run_id or "",
    }
    run_count, metric_count = BatteryDispatchStorage(settings.duckdb_path).upsert_backtest(
        run_record, comparison
    )
    return BatteryBacktestResult(
        run_id,
        rolling,
        comparison,
        artifact_dir,
        mlflow_run_id,
        run_count,
        metric_count,
    )


def _build_multi_origin_inputs(
    forecasts: pd.DataFrame,
    *,
    region: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    horizon: int,
    step_hours: float,
    energy_prices: pd.Series | None,
    fallback_price: float | None,
) -> pd.DataFrame:
    source = forecasts.copy()
    source["forecast_origin"] = pd.to_datetime(source["forecast_origin"], utc=True)
    origins = sorted(
        source.loc[
            (source["region"] == region) & source["forecast_origin"].between(start, end),
            "forecast_origin",
        ].unique()
    )
    parts = [
        build_dispatch_input(
            source,
            region=region,
            forecast_origin=origin,
            horizon=horizon,
            step_hours=step_hours,
            energy_prices=energy_prices,
            fallback_price=fallback_price,
        )[0]
        for origin in origins
    ]
    if not parts:
        raise BatteryOptimizationError("No target forecast origins cover the backtest range.")
    return pd.concat(parts, ignore_index=True)


def _rolling_baseline(
    canonical: pd.DataFrame,
    spec: BatterySpecification,
    horizon: int,
    strategy: Literal["no_battery", "rule_based"],
    settings: Settings,
    actuals: pd.DataFrame | None,
) -> pd.DataFrame:
    soc = spec.initial_soc_mwh
    applied: list[pd.DataFrame] = []
    for _, group in canonical.groupby("forecast_origin", sort=True):
        rolling_spec = spec.with_initial_soc(soc)
        schedule = (
            no_battery_baseline(
                group,
                rolling_spec,
                duration_hours=settings.dispatch_step_hours,
                run_id="no-battery",
            )
            if strategy == "no_battery"
            else rule_based_baseline(
                group,
                rolling_spec,
                duration_hours=settings.dispatch_step_hours,
                run_id="rule-based",
            )
        )
        if len(schedule) != horizon:
            raise BatteryOptimizationError(
                "Baseline horizon length differs from optimized horizon."
            )
        first = schedule.iloc[[0]].copy()
        if actuals is not None:
            first = apply_realized_actuals(first, actuals)
        applied.append(first)
        soc = float(first["soc_end_mwh"].iloc[0])
    return pd.concat(applied, ignore_index=True)


def _load_historical_actuals(
    settings: Settings,
    region: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    demand = DuckDBStorage(settings.duckdb_path).read_data(
        regions=[region], start_date=start.to_pydatetime(), end_date=end.to_pydatetime()
    )[["region", "timestamp_utc", "demand_mw"]]
    renewable = RenewableStorage(settings.duckdb_path).read(region)
    renewable = renewable.loc[renewable["timestamp_utc"].between(start, end)]
    columns = [
        "region",
        "timestamp_utc",
        "solar_generation_mw",
        "wind_generation_mw",
        "total_renewable_generation_mw",
    ]
    merged = demand.merge(
        renewable[columns], on=["region", "timestamp_utc"], how="inner", validate="one_to_one"
    )
    if merged.empty:
        raise BatteryOptimizationError(
            "Historical battery evaluation requires aligned demand and renewable actuals."
        )
    merged["net_load_mw"] = merged["demand_mw"] - merged["total_renewable_generation_mw"]
    return merged.rename(
        columns={
            "demand_mw": "demand_forecast_mw",
            "solar_generation_mw": "solar_forecast_mw",
            "wind_generation_mw": "wind_forecast_mw",
            "total_renewable_generation_mw": "renewable_forecast_mw",
            "net_load_mw": "net_load_before_battery_mw",
        }
    )


def _daily_metrics(
    schedule: pd.DataFrame, spec: BatterySpecification, duration_hours: float
) -> pd.DataFrame:
    source = schedule.copy()
    source["day_utc"] = pd.to_datetime(source["timestamp_utc"], utc=True).dt.strftime("%Y-%m-%d")
    rows = [
        {"day_utc": day, **evaluate_dispatch(group, spec, duration_hours=duration_hours)}
        for day, group in source.groupby("day_utc", sort=True)
    ]
    return pd.DataFrame(rows)


def _backtest_id(
    region: str,
    battery_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    objective: ObjectiveMode,
    mode: SimulationMode,
    alias: str,
) -> str:
    material = (
        f"{region}|{battery_id}|{start.isoformat()}|{end.isoformat()}|{objective}|{mode}|{alias}"
    )
    return hashlib.sha256(material.encode()).hexdigest()[:32]


def _log_backtest_mlflow(
    settings: Settings,
    artifact_dir: Path,
    run_id: str,
    region: str,
    battery_id: str,
    objective_mode: ObjectiveMode,
    mode: SimulationMode,
    metrics: Mapping[str, object],
    *,
    enabled: bool,
) -> str | None:
    if not enabled:
        return None
    setup = initialize_mlflow(settings, settings.battery_experiment_name)
    with mlflow.start_run(experiment_id=setup.experiment_id, run_name=f"backtest-{run_id}") as run:
        mlflow.log_params(
            {
                "region": region,
                "battery_id": battery_id,
                "objective_mode": objective_mode,
                "simulation_mode": mode,
                "oracle_non_deployable": mode == "oracle",
            }
        )
        mlflow.log_metrics(
            {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}
        )
        mlflow.log_artifacts(str(artifact_dir), artifact_path="battery_backtest")
        return str(run.info.run_id)
