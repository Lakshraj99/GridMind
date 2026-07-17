"""Idempotent UTC DuckDB storage for dispatch and backtest records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from gridmind.data.duckdb_connection import connect_duckdb
from gridmind.optimization.contracts import DispatchOptimizationResult


class BatteryDispatchStorage:
    run_table = "battery_dispatch_runs"
    point_table = "battery_dispatch_points"
    backtest_table = "battery_backtest_runs"
    metric_table = "battery_backtest_metrics"

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _initialize(self) -> None:
        with connect_duckdb(self.path) as connection:
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.run_table} (
                    dispatch_run_id VARCHAR PRIMARY KEY, region VARCHAR, battery_id VARCHAR,
                    objective_mode VARCHAR, forecast_origin TIMESTAMPTZ, horizon_hours DOUBLE,
                    solver_name VARCHAR, solver_status VARCHAR, objective_value DOUBLE,
                    solve_time_seconds DOUBLE, optimality_gap DOUBLE,
                    constraint_validation_passed BOOLEAN, peak_before_mw DOUBLE,
                    peak_after_mw DOUBLE, total_charge_mwh DOUBLE,
                    total_discharge_mwh DOUBLE, estimated_cost DOUBLE,
                    degradation_cost DOUBLE, created_at_utc TIMESTAMPTZ,
                    configuration_json VARCHAR, lineage_json VARCHAR,
                    artifact_path VARCHAR, mlflow_run_id VARCHAR,
                    objective_breakdown_json VARCHAR
                )
                """
            )
            connection.execute(
                f"ALTER TABLE {self.run_table} ADD COLUMN IF NOT EXISTS "
                "objective_breakdown_json VARCHAR"
            )
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.point_table} (
                    dispatch_run_id VARCHAR, region VARCHAR, battery_id VARCHAR,
                    forecast_origin TIMESTAMPTZ, timestamp_utc TIMESTAMPTZ,
                    forecast_step BIGINT, demand_forecast_mw DOUBLE,
                    solar_forecast_mw DOUBLE, wind_forecast_mw DOUBLE,
                    renewable_forecast_mw DOUBLE, net_load_before_battery_mw DOUBLE,
                    charge_mw DOUBLE, discharge_mw DOUBLE, net_battery_power_mw DOUBLE,
                    soc_start_mwh DOUBLE, soc_end_mwh DOUBLE,
                    net_load_after_battery_mw DOUBLE, energy_price DOUBLE,
                    marginal_degradation_cost DOUBLE, operating_mode VARCHAR,
                    solver_status VARCHAR, created_at_utc TIMESTAMPTZ, metadata_json VARCHAR,
                    PRIMARY KEY (dispatch_run_id, timestamp_utc)
                )
                """
            )
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.backtest_table} (
                    backtest_run_id VARCHAR PRIMARY KEY, region VARCHAR, battery_id VARCHAR,
                    objective_mode VARCHAR, simulation_mode VARCHAR, start_utc TIMESTAMPTZ,
                    end_utc TIMESTAMPTZ, evaluated_horizons BIGINT,
                    successful_optimizations BIGINT, solver_failures BIGINT,
                    created_at_utc TIMESTAMPTZ, configuration_json VARCHAR,
                    artifact_path VARCHAR, mlflow_run_id VARCHAR
                )
                """
            )
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.metric_table} (
                    backtest_run_id VARCHAR, strategy VARCHAR, metric_name VARCHAR,
                    metric_value DOUBLE, PRIMARY KEY (backtest_run_id, strategy, metric_name)
                )
                """
            )

    def upsert_dispatch(
        self,
        result: DispatchOptimizationResult,
        *,
        objective_mode: str,
        horizon_hours: float,
        configuration: dict[str, Any],
        artifact_path: Path | None = None,
        mlflow_run_id: str | None = None,
    ) -> int:
        schedule = result.schedule.copy()
        duration_hours = float(configuration.get("step_hours", 1.0))
        run = pd.DataFrame(
            [
                {
                    "dispatch_run_id": result.dispatch_run_id,
                    "region": str(schedule["region"].iloc[0]),
                    "battery_id": str(schedule["battery_id"].iloc[0]),
                    "objective_mode": objective_mode,
                    "forecast_origin": schedule["forecast_origin"].iloc[0],
                    "horizon_hours": horizon_hours,
                    "solver_name": result.diagnostics.solver_name,
                    "solver_status": result.diagnostics.status,
                    "objective_value": result.diagnostics.objective_value,
                    "solve_time_seconds": result.diagnostics.solve_time_seconds,
                    "optimality_gap": result.diagnostics.optimality_gap,
                    "constraint_validation_passed": (
                        result.diagnostics.constraint_validation_passed
                    ),
                    "peak_before_mw": float(schedule["net_load_before_battery_mw"].max()),
                    "peak_after_mw": float(schedule["net_load_after_battery_mw"].max()),
                    "total_charge_mwh": float(schedule["charge_mw"].sum() * duration_hours),
                    "total_discharge_mwh": float(schedule["discharge_mw"].sum() * duration_hours),
                    "estimated_cost": result.objective_breakdown["energy_cost"],
                    "degradation_cost": result.objective_breakdown["degradation_cost"],
                    "created_at_utc": schedule["created_at_utc"].iloc[0],
                    "configuration_json": json.dumps(configuration, sort_keys=True, default=str),
                    "lineage_json": json.dumps(result.lineage, sort_keys=True, default=str),
                    "artifact_path": str(artifact_path or ""),
                    "mlflow_run_id": mlflow_run_id or "",
                    "objective_breakdown_json": json.dumps(
                        result.objective_breakdown, sort_keys=True, default=str
                    ),
                }
            ]
        )
        with connect_duckdb(self.path) as connection:
            connection.register("incoming_battery_run", run)
            connection.register("incoming_battery_points", schedule)
            connection.execute(
                f"DELETE FROM {self.point_table} WHERE dispatch_run_id = ?",
                [result.dispatch_run_id],
            )
            connection.execute(
                f"DELETE FROM {self.run_table} WHERE dispatch_run_id = ?",
                [result.dispatch_run_id],
            )
            connection.execute(f"INSERT INTO {self.run_table} SELECT * FROM incoming_battery_run")
            connection.execute(
                f"INSERT INTO {self.point_table} SELECT * FROM incoming_battery_points"
            )
            row = connection.execute(f"SELECT COUNT(*) FROM {self.point_table}").fetchone()
        return int(row[0]) if row else 0

    def read_dispatches(
        self,
        *,
        region: str | None = None,
        battery_id: str | None = None,
        objective_mode: str | None = None,
        solver_status: str | None = None,
        forecast_origin: object | None = None,
        start: object | None = None,
        end: object | None = None,
    ) -> pd.DataFrame:
        clauses: list[str] = []
        parameters: list[object] = []
        for column, value in (
            ("region", region),
            ("battery_id", battery_id),
            ("objective_mode", objective_mode),
            ("solver_status", solver_status),
            ("forecast_origin", forecast_origin),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                parameters.append(value)
        if start is not None:
            clauses.append("forecast_origin >= ?")
            parameters.append(start)
        if end is not None:
            clauses.append("forecast_origin <= ?")
            parameters.append(end)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with connect_duckdb(self.path, read_only=True) as connection:
            frame = connection.execute(
                f"SELECT * FROM {self.run_table} {where} ORDER BY forecast_origin, dispatch_run_id",
                parameters,
            ).fetchdf()
        for column in ("forecast_origin", "created_at_utc"):
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], utc=True)
        return frame

    def read_points(self, dispatch_run_id: str) -> pd.DataFrame:
        with connect_duckdb(self.path, read_only=True) as connection:
            frame = connection.execute(
                f"SELECT * FROM {self.point_table} WHERE dispatch_run_id = ? "
                "ORDER BY timestamp_utc",
                [dispatch_run_id],
            ).fetchdf()
        for column in ("forecast_origin", "timestamp_utc", "created_at_utc"):
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], utc=True)
        return frame

    def upsert_backtest(
        self,
        run: dict[str, Any],
        strategy_metrics: pd.DataFrame,
    ) -> tuple[int, int]:
        run_frame = pd.DataFrame([run])
        metric_rows = [
            {
                "backtest_run_id": run["backtest_run_id"],
                "strategy": row["strategy"],
                "metric_name": column,
                "metric_value": float(row[column]),
            }
            for _, row in strategy_metrics.iterrows()
            for column in strategy_metrics.columns
            if column != "strategy"
        ]
        metrics = pd.DataFrame(metric_rows)
        with connect_duckdb(self.path) as connection:
            connection.execute(
                f"DELETE FROM {self.metric_table} WHERE backtest_run_id = ?",
                [run["backtest_run_id"]],
            )
            connection.execute(
                f"DELETE FROM {self.backtest_table} WHERE backtest_run_id = ?",
                [run["backtest_run_id"]],
            )
            connection.register("incoming_backtest_run", run_frame)
            connection.execute(
                f"INSERT INTO {self.backtest_table} SELECT * FROM incoming_backtest_run"
            )
            if not metrics.empty:
                connection.register("incoming_backtest_metrics", metrics)
                connection.execute(
                    f"INSERT INTO {self.metric_table} SELECT * FROM incoming_backtest_metrics"
                )
            run_count = connection.execute(f"SELECT COUNT(*) FROM {self.backtest_table}").fetchone()
            metric_count = connection.execute(
                f"SELECT COUNT(*) FROM {self.metric_table}"
            ).fetchone()
        return (
            int(run_count[0]) if run_count else 0,
            int(metric_count[0]) if metric_count else 0,
        )
