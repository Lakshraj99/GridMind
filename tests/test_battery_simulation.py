"""Forecast alignment, rolling simulation, persistence, pipeline, and MLflow tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from mlflow import MlflowClient

from gridmind.config import Settings
from gridmind.exceptions import BatteryOptimizationError
from gridmind.optimization.simulation import build_dispatch_input, rolling_horizon_simulation
from gridmind.optimization.storage import BatteryDispatchStorage
from gridmind.pipelines.backtest_dispatch import run_battery_backtest
from gridmind.pipelines.optimize_dispatch import run_dispatch_optimization
from test_battery_physics import battery_spec


def target_forecasts(origins: int = 1, horizon: int = 4) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    first = pd.Timestamp("2026-01-01T00:00:00Z")
    for origin_offset in range(origins):
        origin = first + pd.Timedelta(hours=origin_offset)
        timestamps = pd.date_range(origin + pd.Timedelta(hours=1), periods=horizon, freq="h")
        for target, base in (
            ("demand_mw", 100.0),
            ("solar_generation_mw", 10.0),
            ("wind_generation_mw", 10.0),
            ("net_load_mw", 80.0),
        ):
            for step, timestamp in enumerate(timestamps, 1):
                value = base + (50 if target in {"demand_mw", "net_load_mw"} and step == 3 else 0)
                rows.append(
                    {
                        "region": "PJM",
                        "target": target,
                        "forecast_origin": origin,
                        "timestamp_utc": timestamp,
                        "forecast_step": step,
                        "predicted_value": value,
                        "model_name": f"model-{target}",
                        "model_version": "1",
                        "run_id": f"run-{target}",
                        "weather_mode": "realistic_forecast",
                        "created_at_utc": origin,
                    }
                )
    return pd.DataFrame(rows)


def canonical_inputs(origins: int = 2, horizon: int = 4) -> pd.DataFrame:
    source = target_forecasts(origins, horizon)
    parts = [
        build_dispatch_input(
            source,
            region="PJM",
            forecast_origin=pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(hours=offset),
            horizon=horizon,
            fallback_price=50,
        )[0]
        for offset in range(origins)
    ]
    return pd.concat(parts, ignore_index=True)


def test_forecast_alignment_preserves_lineage_and_rejects_missing_or_mixed_data() -> None:
    source = target_forecasts()
    frame, lineage = build_dispatch_input(
        source,
        region="PJM",
        forecast_origin="2026-01-01T00:00:00Z",
        horizon=4,
        fallback_price=42,
    )
    assert len(frame) == 4
    assert (frame["net_load_before_battery_mw"] == [80, 80, 130, 80]).all()
    assert set(lineage) == {
        "demand_mw",
        "solar_generation_mw",
        "wind_generation_mw",
        "net_load_mw",
    }
    assert json.loads(frame.iloc[0]["metadata_json"])["forecast_lineage"]
    with pytest.raises(BatteryOptimizationError, match="No target forecasts"):
        build_dispatch_input(source, region="PJM", forecast_origin="2026-01-02", horizon=4)
    incomplete = source.loc[
        ~(
            (source["target"] == "demand_mw")
            & (source["timestamp_utc"] == pd.Timestamp("2026-01-01T04:00:00Z"))
        )
    ]
    with pytest.raises(BatteryOptimizationError, match="incomplete"):
        build_dispatch_input(
            incomplete,
            region="PJM",
            forecast_origin="2026-01-01",
            horizon=4,
            fallback_price=1,
        )
    mixed = pd.concat([source, source.iloc[[0]].assign(model_version="2")])
    with pytest.raises(BatteryOptimizationError, match="incompatible"):
        build_dispatch_input(
            mixed,
            region="PJM",
            forecast_origin="2026-01-01",
            horizon=4,
            fallback_price=1,
        )


def test_rolling_horizon_applies_first_action_carries_soc_and_labels_oracle() -> None:
    canonical = canonical_inputs()
    spec = battery_spec(self_discharge_rate=0, terminal_soc_target_mwh=50)
    forecast = rolling_horizon_simulation(
        canonical,
        spec,
        horizon=4,
        objective_mode="peak_shaving",
        mode="forecast_based",
    )
    assert len(forecast.applied_dispatch) == 2
    assert forecast.successful_optimizations == 2
    assert forecast.solver_failures == 0
    assert forecast.applied_dispatch.iloc[1]["soc_start_mwh"] == pytest.approx(
        forecast.applied_dispatch.iloc[0]["soc_end_mwh"]
    )
    assert forecast.horizon_results["forecast_origin"].diff().iloc[1] == pd.Timedelta(hours=1)
    assert json.loads(forecast.applied_dispatch.iloc[0]["metadata_json"])[
        "applied_first_action_only"
    ]
    actuals = canonical.loc[canonical["forecast_origin"] == canonical["forecast_origin"].min()][
        [
            "timestamp_utc",
            "demand_forecast_mw",
            "solar_forecast_mw",
            "wind_forecast_mw",
            "renewable_forecast_mw",
            "net_load_before_battery_mw",
        ]
    ].drop_duplicates("timestamp_utc")
    all_times = canonical["timestamp_utc"].drop_duplicates()
    actuals = actuals.set_index("timestamp_utc").reindex(all_times).ffill().reset_index()
    oracle = rolling_horizon_simulation(
        canonical,
        spec,
        horizon=4,
        objective_mode="peak_shaving",
        mode="oracle",
        oracle_actuals=actuals,
    )
    assert oracle.mode == "oracle"
    assert oracle.horizon_results["oracle_non_deployable"].all()
    assert json.loads(oracle.applied_dispatch.iloc[0]["metadata_json"])["oracle_non_deployable"]


def test_rolling_horizon_rejects_missing_origins_and_oracle_coverage() -> None:
    canonical = canonical_inputs(origins=3)
    middle = pd.Timestamp("2026-01-01T01:00:00Z")
    missing = canonical.loc[canonical["forecast_origin"] != middle]
    spec = battery_spec(self_discharge_rate=0)
    with pytest.raises(BatteryOptimizationError, match="missing forecast origins"):
        rolling_horizon_simulation(missing, spec, horizon=4, objective_mode="peak_shaving")
    with pytest.raises(BatteryOptimizationError, match="requires explicitly"):
        rolling_horizon_simulation(
            canonical.iloc[:4],
            spec,
            horizon=4,
            objective_mode="peak_shaving",
            mode="oracle",
        )
    with pytest.raises(BatteryOptimizationError, match="do not cover"):
        rolling_horizon_simulation(
            canonical.iloc[:4],
            spec,
            horizon=4,
            objective_mode="peak_shaving",
            mode="oracle",
            oracle_actuals=pd.DataFrame(
                {
                    "timestamp_utc": canonical.iloc[:1]["timestamp_utc"],
                    "demand_forecast_mw": [100],
                }
            ),
        )


def test_dispatch_pipeline_artifacts_storage_idempotency_filters_and_mlflow(tmp_path: Path) -> None:
    database = tmp_path / "grid.duckdb"
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    settings = Settings(
        DUCKDB_PATH=database,
        MLFLOW_TRACKING_URI=tracking_uri,
        MLFLOW_ARTIFACT_ROOT=tmp_path / "mlartifacts",
        MLFLOW_ENABLED=True,
        DISPATCH_HORIZON_HOURS=4,
        BATTERY_SELF_DISCHARGE_PER_HOUR=0,
        BATTERY_TERMINAL_SOC_MWH=250,
        _env_file=None,
    )
    result = run_dispatch_optimization(
        settings,
        region="PJM",
        battery_id="bess-1",
        forecast_origin="2026-01-01",
        horizon=4,
        objective_mode="peak_shaving",
        forecast_frame=target_forecasts(),
        artifact_root=tmp_path / "dispatch",
    )
    assert result.mlflow_run_id
    assert result.duckdb_rows == 4
    for filename in (
        "dispatch_schedule.parquet",
        "dispatch_summary.json",
        "battery_configuration.json",
        "objective_breakdown.json",
        "solver_diagnostics.json",
        "soc_trajectory.csv",
    ):
        assert (result.artifact_dir / filename).exists()
    rerun = run_dispatch_optimization(
        settings,
        region="PJM",
        battery_id="bess-1",
        forecast_origin="2026-01-01",
        horizon=4,
        objective_mode="peak_shaving",
        forecast_frame=target_forecasts(),
        artifact_root=tmp_path / "dispatch",
        mlflow_enabled=False,
    )
    assert rerun.optimization.dispatch_run_id == result.optimization.dispatch_run_id
    assert rerun.duckdb_rows == 4
    storage = BatteryDispatchStorage(database)
    runs = storage.read_dispatches(
        region="PJM",
        battery_id="bess-1",
        objective_mode="peak_shaving",
        solver_status="optimal",
        start="2025-12-31",
        end="2026-01-02",
    )
    assert len(runs) == 1
    assert str(runs.iloc[0]["forecast_origin"].tz) == "UTC"
    points = storage.read_points(result.optimization.dispatch_run_id)
    assert len(points) == 4
    assert json.loads(runs.iloc[0]["lineage_json"])["requested_model_alias"] == "champion"
    client = MlflowClient(tracking_uri=tracking_uri)
    artifacts = client.list_artifacts(str(result.mlflow_run_id), "battery_dispatch")
    assert {item.path.rsplit("/", 1)[-1] for item in artifacts}.issuperset(
        {"dispatch_schedule.parquet", "solver_diagnostics.json"}
    )


def test_backtest_pipeline_writes_comparison_and_idempotent_duckdb(tmp_path: Path) -> None:
    settings = Settings(
        DUCKDB_PATH=tmp_path / "grid.duckdb",
        MLFLOW_ENABLED=False,
        DISPATCH_HORIZON_HOURS=4,
        BATTERY_SELF_DISCHARGE_PER_HOUR=0,
        _env_file=None,
    )
    canonical = canonical_inputs(origins=2)
    actuals = (
        canonical[
            [
                "timestamp_utc",
                "demand_forecast_mw",
                "solar_forecast_mw",
                "wind_forecast_mw",
                "renewable_forecast_mw",
                "net_load_before_battery_mw",
            ]
        ]
        .drop_duplicates("timestamp_utc")
        .copy()
    )
    actuals["net_load_before_battery_mw"] += 5
    actuals["demand_forecast_mw"] += 5
    result = run_battery_backtest(
        settings,
        region="PJM",
        battery_id="bess-1",
        start_date="2026-01-01T00:00:00Z",
        end_date="2026-01-01T01:00:00Z",
        objective_mode="peak_shaving",
        horizon=4,
        forecast_inputs=canonical,
        historical_actuals=actuals,
        mlflow_enabled=False,
        artifact_root=tmp_path / "backtests",
    )
    assert set(result.strategy_comparison["strategy"]) == {
        "no_battery",
        "rule_based",
        "optimized_forecast_based",
    }
    assert result.duckdb_run_rows == 1
    assert result.duckdb_metric_rows > 1
    assert result.mlflow_run_id is None
    assert json.loads(result.rolling.applied_dispatch.iloc[0]["metadata_json"])[
        "evaluated_on_realized_observation"
    ]
    for filename in (
        "dispatch_points.parquet",
        "horizon_results.parquet",
        "strategy_comparison.csv",
        "overall_metrics.json",
        "daily_metrics.csv",
        "solver_failures.csv",
        "configuration.json",
    ):
        assert (result.artifact_dir / filename).exists()
    rerun = run_battery_backtest(
        settings,
        region="PJM",
        battery_id="bess-1",
        start_date="2026-01-01T00:00:00Z",
        end_date="2026-01-01T01:00:00Z",
        objective_mode="peak_shaving",
        horizon=4,
        forecast_inputs=canonical,
        historical_actuals=actuals,
        mlflow_enabled=False,
        artifact_root=tmp_path / "backtests",
    )
    assert rerun.backtest_run_id == result.backtest_run_id
    assert rerun.duckdb_run_rows == 1
