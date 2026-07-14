"""One-horizon forecast-aligned battery optimization pipeline."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient

from gridmind.config import Settings
from gridmind.data.storage import write_json_report
from gridmind.data.target_storage import TargetForecastStorage
from gridmind.exceptions import BatteryOptimizationError
from gridmind.mlflow_config import initialize_mlflow
from gridmind.optimization.battery import battery_specification_from_settings
from gridmind.optimization.contracts import (
    BatterySpecification,
    DispatchOptimizationResult,
    ObjectiveMode,
    ObjectiveWeights,
)
from gridmind.optimization.evaluation import evaluate_dispatch
from gridmind.optimization.simulation import build_dispatch_input
from gridmind.optimization.solver import optimize_battery_dispatch
from gridmind.optimization.storage import BatteryDispatchStorage
from gridmind.renewables.targets import get_target_definition


@dataclass(frozen=True)
class DispatchPipelineResult:
    optimization: DispatchOptimizationResult
    battery: BatterySpecification
    metrics: dict[str, float]
    artifact_dir: Path
    duckdb_rows: int
    mlflow_run_id: str | None


def run_dispatch_optimization(
    settings: Settings,
    *,
    region: str,
    battery_id: str,
    forecast_origin: object,
    horizon: int | None = None,
    objective_mode: ObjectiveMode = "peak_shaving",
    model_alias: str = "champion",
    robust: bool = False,
    forecast_frame: pd.DataFrame | None = None,
    energy_prices: pd.Series | None = None,
    battery: BatterySpecification | None = None,
    mlflow_enabled: bool | None = None,
    artifact_root: Path = Path("artifacts/battery_dispatch"),
) -> DispatchPipelineResult:
    if not settings.battery_optimization_enabled:
        raise BatteryOptimizationError("Battery optimization is disabled by configuration.")
    selected_horizon = horizon or settings.dispatch_horizon_hours
    forecasts = (
        forecast_frame.copy()
        if forecast_frame is not None
        else TargetForecastStorage(settings.duckdb_path).read()
    )
    if (
        objective_mode in {"energy_arbitrage", "balanced"}
        and energy_prices is None
        and settings.fallback_energy_price_per_mwh is None
    ):
        raise BatteryOptimizationError(
            "Energy prices are required for this objective. Supply prices or configure "
            "FALLBACK_ENERGY_PRICE_PER_MWH explicitly."
        )
    dispatch_input, lineage = build_dispatch_input(
        forecasts,
        region=region,
        forecast_origin=forecast_origin,
        horizon=selected_horizon,
        step_hours=settings.dispatch_step_hours,
        energy_prices=energy_prices,
        fallback_price=settings.fallback_energy_price_per_mwh,
    )
    if forecast_frame is None:
        _validate_model_alias_lineage(settings, lineage, model_alias)
    lineage["requested_model_alias"] = model_alias
    spec = battery or battery_specification_from_settings(settings, battery_id)
    weights = ObjectiveWeights(
        peak=settings.peak_shaving_weight,
        energy_cost=settings.energy_cost_weight,
        renewable_utilization=settings.renewable_utilization_weight,
        degradation=settings.degradation_weight,
        terminal_soc=settings.terminal_soc_penalty_weight,
    )
    optimized = optimize_battery_dispatch(
        dispatch_input,
        spec,
        objective_mode=objective_mode,
        weights=weights,
        duration_hours=settings.dispatch_step_hours,
        timeout_seconds=settings.dispatch_solver_timeout_seconds,
        robust=robust,
        demand_uplift_pct=settings.robust_demand_uplift_pct,
        renewable_reduction_pct=settings.robust_renewable_reduction_pct,
        extra_reserve_pct=settings.robust_extra_reserve_pct,
    )
    optimized.lineage.update(lineage)
    artifact_dir = artifact_root / optimized.dispatch_run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_dispatch(
        optimized.schedule, spec, duration_hours=settings.dispatch_step_hours
    )
    _write_artifacts(artifact_dir, optimized, spec, objective_mode, weights, metrics)
    enabled = settings.mlflow_enabled if mlflow_enabled is None else mlflow_enabled
    mlflow_run_id = _log_mlflow(
        settings,
        artifact_dir,
        optimized,
        spec,
        objective_mode,
        weights,
        metrics,
        enabled=enabled,
    )
    configuration = {
        "battery": spec.as_dict(),
        "objective_mode": objective_mode,
        "weights": asdict(weights),
        "horizon": selected_horizon,
        "step_hours": settings.dispatch_step_hours,
        "robust": robust,
    }
    count = BatteryDispatchStorage(settings.duckdb_path).upsert_dispatch(
        optimized,
        objective_mode=objective_mode,
        horizon_hours=selected_horizon * settings.dispatch_step_hours,
        configuration=configuration,
        artifact_path=artifact_dir,
        mlflow_run_id=mlflow_run_id,
    )
    return DispatchPipelineResult(optimized, spec, metrics, artifact_dir, count, mlflow_run_id)


def _write_artifacts(
    directory: Path,
    result: DispatchOptimizationResult,
    spec: BatterySpecification,
    objective_mode: ObjectiveMode,
    weights: ObjectiveWeights,
    metrics: dict[str, float],
) -> None:
    result.schedule.to_parquet(directory / "dispatch_schedule.parquet", index=False)
    result.schedule[["timestamp_utc", "soc_start_mwh", "soc_end_mwh"]].to_csv(
        directory / "soc_trajectory.csv", index=False
    )
    write_json_report(
        {
            "dispatch_run_id": result.dispatch_run_id,
            "objective_mode": objective_mode,
            "metrics": metrics,
            "decision_support_only": True,
        },
        directory / "dispatch_summary.json",
    )
    write_json_report(spec.as_dict(), directory / "battery_configuration.json")
    write_json_report(
        {"weights": asdict(weights), "contributions": result.objective_breakdown},
        directory / "objective_breakdown.json",
    )
    write_json_report(asdict(result.diagnostics), directory / "solver_diagnostics.json")


def _log_mlflow(
    settings: Settings,
    artifact_dir: Path,
    result: DispatchOptimizationResult,
    spec: BatterySpecification,
    objective_mode: ObjectiveMode,
    weights: ObjectiveWeights,
    metrics: dict[str, float],
    *,
    enabled: bool,
) -> str | None:
    if not enabled:
        return None
    setup = initialize_mlflow(settings, settings.battery_experiment_name)
    with mlflow.start_run(
        experiment_id=setup.experiment_id, run_name=f"dispatch-{spec.battery_id}"
    ) as run:
        mlflow.log_params(
            {
                "battery_id": spec.battery_id,
                "objective_mode": objective_mode,
                "forecast_origin": result.lineage["forecast_origin"],
                "solver": result.diagnostics.solver_name,
                "solver_status": result.diagnostics.status,
                "objective_weights": json.dumps(asdict(weights), sort_keys=True),
                "battery_specification": json.dumps(spec.as_dict(), sort_keys=True),
                "git_commit": _git_commit(),
                "python_version": platform.python_version(),
                "package_versions": ",".join(
                    f"{name}={importlib.metadata.version(name)}"
                    for name in ("pandas", "numpy", "scipy", "duckdb")
                ),
            }
        )
        mlflow.log_metrics(
            {
                **metrics,
                "solve_time_seconds": result.diagnostics.solve_time_seconds,
                "objective_value": result.diagnostics.objective_value or 0.0,
            }
        )
        mlflow.log_artifacts(str(artifact_dir), artifact_path="battery_dispatch")
        return str(run.info.run_id)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _validate_model_alias_lineage(
    settings: Settings, lineage: dict[str, object], alias: str
) -> None:
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    for target, raw_identity in lineage.items():
        if not isinstance(raw_identity, dict):
            continue
        definition = get_target_definition(target)
        registered_name = str(getattr(settings, definition.registry_setting))
        version = client.get_model_version_by_alias(registered_name, alias)
        stored_version = str(raw_identity.get("model_version", ""))
        if stored_version != str(version.version):
            raise BatteryOptimizationError(
                f"Stored {target} forecast version {stored_version} does not match "
                f"{registered_name}@{alias} version {version.version}."
            )
