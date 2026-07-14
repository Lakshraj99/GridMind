"""Milestone 5 Typer help, success, listing, and failure paths."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from typer.testing import CliRunner

import gridmind.cli as cli_module
from gridmind.config import Settings
from gridmind.exceptions import BatteryOptimizationError

runner = CliRunner()


def test_battery_cli_help_and_invalid_objective() -> None:
    for command in ("optimize-dispatch", "battery-backtest", "dispatches"):
        result = runner.invoke(cli_module.app, [command, "--help"])
        assert result.exit_code == 0
        assert "Usage" in result.output
    invalid = runner.invoke(
        cli_module.app,
        [
            "optimize-dispatch",
            "--region",
            "PJM",
            "--battery-id",
            "b1",
            "--forecast-origin",
            "2026-01-01",
            "--objective",
            "profit_magic",
        ],
    )
    assert invalid.exit_code != 0


def test_optimize_dispatch_cli_success_and_missing_forecast_failure(
    tmp_path: Path, monkeypatch: object
) -> None:
    settings = Settings(DUCKDB_PATH=tmp_path / "grid.duckdb", _env_file=None)
    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)  # type: ignore[attr-defined]
    schedule = pd.DataFrame(
        {
            "forecast_origin": [pd.Timestamp("2026-01-01T00:00:00Z")],
            "soc_end_mwh": [250.0],
        }
    )
    fake = SimpleNamespace(
        optimization=SimpleNamespace(
            schedule=schedule,
            diagnostics=SimpleNamespace(status="optimal", objective_value=1.25),
        ),
        battery=SimpleNamespace(battery_id="b1", capacity_mwh=500),
        metrics={
            "original_peak_load_mw": 100,
            "optimized_peak_load_mw": 80,
            "absolute_peak_reduction_mw": 20,
            "charge_energy_mwh": 10,
            "discharge_energy_mwh": 10,
        },
        artifact_dir=tmp_path / "artifacts",
        duckdb_rows=24,
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        cli_module, "run_dispatch_optimization", lambda *_args, **_kwargs: fake
    )
    success = runner.invoke(
        cli_module.app,
        [
            "optimize-dispatch",
            "--region",
            "PJM",
            "--battery-id",
            "b1",
            "--forecast-origin",
            "2026-01-01",
            "--no-mlflow",
        ],
    )
    assert success.exit_code == 0
    assert "Solver status: optimal" in success.output
    assert "Decision-support simulation only" in success.output

    def fail(*_args: object, **_kwargs: object) -> None:
        raise BatteryOptimizationError("missing forecast horizon")

    monkeypatch.setattr(cli_module, "run_dispatch_optimization", fail)  # type: ignore[attr-defined]
    failed = runner.invoke(
        cli_module.app,
        [
            "optimize-dispatch",
            "--region",
            "PJM",
            "--battery-id",
            "b1",
            "--forecast-origin",
            "2026-01-01",
        ],
    )
    assert failed.exit_code == 1
    assert "missing forecast horizon" in failed.output


def test_backtest_and_dispatch_listing_cli_paths(tmp_path: Path, monkeypatch: object) -> None:
    settings = Settings(DUCKDB_PATH=tmp_path / "grid.duckdb", _env_file=None)
    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)  # type: ignore[attr-defined]
    comparison = pd.DataFrame(
        [
            {
                "strategy": "optimized_forecast_based",
                "absolute_peak_reduction_mw": 12.0,
                "energy_cost_savings": 50.0,
                "equivalent_full_cycles": 0.2,
                "soc_violations": 0.0,
                "power_violations": 0.0,
            }
        ]
    )
    fake = SimpleNamespace(
        rolling=SimpleNamespace(successful_optimizations=3, solver_failures=0),
        strategy_comparison=comparison,
        artifact_dir=tmp_path / "backtest",
        mlflow_run_id=None,
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        cli_module, "run_battery_backtest", lambda *_args, **_kwargs: fake
    )
    result = runner.invoke(
        cli_module.app,
        [
            "battery-backtest",
            "--region",
            "PJM",
            "--battery-id",
            "b1",
            "--start-date",
            "2026-01-01",
            "--end-date",
            "2026-01-02",
            "--objective",
            "peak_shaving",
            "--no-mlflow",
        ],
    )
    assert result.exit_code == 0
    assert "Evaluated horizons: 3" in result.output

    class FakeStorage:
        def __init__(self, _path: Path) -> None:
            pass

        def read_dispatches(self, **_kwargs: object) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "dispatch_run_id": ["run"],
                    "forecast_origin": [pd.Timestamp("2026-01-01", tz="UTC")],
                    "created_at_utc": [pd.Timestamp("2026-01-01", tz="UTC")],
                }
            )

    monkeypatch.setattr(cli_module, "BatteryDispatchStorage", FakeStorage)  # type: ignore[attr-defined]
    csv_path = tmp_path / "dispatches.csv"
    listing = runner.invoke(
        cli_module.app,
        ["dispatches", "--region", "PJM", "--csv", str(csv_path)],
    )
    assert listing.exit_code == 0
    assert csv_path.exists()
    assert "Z" in csv_path.read_text()
