"""HiGHS battery objective, solver status, and robust-mode tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

import gridmind.optimization.solver as solver_module
from gridmind.exceptions import BatteryOptimizationError
from gridmind.optimization.contracts import ObjectiveWeights, validate_dispatch_input
from gridmind.optimization.solver import optimize_battery_dispatch
from test_battery_physics import battery_spec, dispatch_input


@pytest.mark.parametrize(
    "mode",
    ["peak_shaving", "energy_arbitrage", "renewable_utilization", "balanced"],
)
def test_all_objective_modes_are_deterministic_physical_and_report_contributions(
    mode: str,
) -> None:
    source = dispatch_input()
    spec = battery_spec(self_discharge_rate=0, terminal_soc_target_mwh=50)
    first = optimize_battery_dispatch(source, spec, objective_mode=mode)  # type: ignore[arg-type]
    second = optimize_battery_dispatch(source, spec, objective_mode=mode)  # type: ignore[arg-type]
    assert first.diagnostics.status == "optimal"
    assert first.diagnostics.solver_name.endswith("HiGHS")
    assert first.diagnostics.constraint_validation_passed
    assert first.dispatch_run_id == second.dispatch_run_id
    pd.testing.assert_series_equal(first.schedule["charge_mw"], second.schedule["charge_mw"])
    assert not (
        (first.schedule["charge_mw"] > 1e-6) & (first.schedule["discharge_mw"] > 1e-6)
    ).any()
    assert first.schedule["soc_end_mwh"].iloc[-1] == pytest.approx(50)
    assert set(first.objective_breakdown).issuperset(
        {
            "peak_load_mw",
            "energy_cost",
            "renewable_aligned_charge_mwh",
            "degradation_cost",
            "terminal_soc_deviation_mwh",
        }
    )
    assert first.lineage["robust_mode"] is False


def test_peak_shaving_and_arbitrage_improve_their_named_metrics() -> None:
    source = dispatch_input()
    spec = battery_spec(self_discharge_rate=0, terminal_soc_target_mwh=50)
    peak = optimize_battery_dispatch(source, spec, objective_mode="peak_shaving")
    arbitrage = optimize_battery_dispatch(source, spec, objective_mode="energy_arbitrage")
    assert (
        peak.schedule["net_load_after_battery_mw"].max()
        < source["net_load_before_battery_mw"].max()
    )
    before_cost = float((source["net_load_before_battery_mw"] * source["energy_price"]).sum())
    assert arbitrage.objective_breakdown["energy_cost"] < before_cost


def test_robust_mode_preserves_original_values_and_raises_reserve() -> None:
    source = dispatch_input()
    spec = battery_spec(
        self_discharge_rate=0,
        terminal_soc_target_mwh=50,
        reserve_soc_mwh=10,
    )
    result = optimize_battery_dispatch(
        source,
        spec,
        objective_mode="peak_shaving",
        robust=True,
        demand_uplift_pct=0.03,
        renewable_reduction_pct=0.10,
        extra_reserve_pct=0.05,
    )
    metadata = json.loads(result.schedule.iloc[0]["metadata_json"])
    assert metadata["unadjusted_demand_forecast_mw"] == source.iloc[0]["demand_forecast_mw"]
    assert result.schedule.iloc[0]["demand_forecast_mw"] == pytest.approx(
        source.iloc[0]["demand_forecast_mw"] * 1.03
    )
    assert result.lineage["extra_reserve_pct"] == 0.05
    with pytest.raises(BatteryOptimizationError, match="non-negative"):
        optimize_battery_dispatch(source, spec, robust=True, demand_uplift_pct=-1)
    with pytest.raises(BatteryOptimizationError, match="exceeds"):
        optimize_battery_dispatch(source, spec, robust=True, extra_reserve_pct=1)


def test_incomplete_horizon_and_invalid_input_fail_clearly() -> None:
    source = dispatch_input()
    with pytest.raises(BatteryOptimizationError, match="missing columns"):
        validate_dispatch_input(source.drop(columns="energy_price"), horizon=4, step_hours=1)
    with pytest.raises(BatteryOptimizationError, match="requires 4"):
        validate_dispatch_input(source.iloc[:-1], horizon=4, step_hours=1)
    gap = source.copy()
    gap.loc[2, "timestamp_utc"] += pd.Timedelta(hours=1)
    with pytest.raises(BatteryOptimizationError, match="not complete"):
        validate_dispatch_input(gap, horizon=4, step_hours=1)
    bad = source.copy()
    bad.loc[0, "metadata_json"] = "bad"
    with pytest.raises(BatteryOptimizationError, match="metadata"):
        validate_dispatch_input(bad, horizon=4, step_hours=1)
    with pytest.raises(BatteryOptimizationError, match="positive"):
        validate_dispatch_input(source, horizon=4, step_hours=0)


def test_infeasible_timeout_and_solver_error_are_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    source = dispatch_input(periods=1)
    impossible = battery_spec(
        initial_soc_mwh=20,
        terminal_soc_target_mwh=100,
        max_charge_mw=1,
        self_discharge_rate=0,
    )
    with pytest.raises(BatteryOptimizationError, match="infeasible"):
        optimize_battery_dispatch(source, impossible)

    monkeypatch.setattr(
        solver_module,
        "milp",
        lambda *_args, **_kwargs: SimpleNamespace(status=1, x=None, fun=None, message="time limit"),
    )
    with pytest.raises(BatteryOptimizationError, match="timeout"):
        optimize_battery_dispatch(source, battery_spec(self_discharge_rate=0))

    def explode(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("solver unavailable")

    monkeypatch.setattr(solver_module, "milp", explode)
    with pytest.raises(BatteryOptimizationError, match="solver failed"):
        optimize_battery_dispatch(source, battery_spec(self_discharge_rate=0))


def test_energy_and_weight_contracts_require_valid_inputs() -> None:
    source = dispatch_input()
    source.loc[0, "energy_price"] = float("nan")
    with pytest.raises(BatteryOptimizationError, match="finite"):
        optimize_battery_dispatch(source, battery_spec(), objective_mode="energy_arbitrage")
    weights = ObjectiveWeights(peak=2, energy_cost=3, renewable_utilization=4)
    result = optimize_battery_dispatch(
        dispatch_input(),
        battery_spec(self_discharge_rate=0),
        objective_mode="balanced",
        weights=weights,
    )
    assert result.objective_breakdown["weighted_peak_contribution"] > 0


def test_solver_respects_variable_duration_and_daily_cycle_limit() -> None:
    source = dispatch_input(step=0.5)
    spec = battery_spec(
        self_discharge_rate=0,
        terminal_soc_target_mwh=50,
        maximum_daily_cycles=0.10,
    )
    result = optimize_battery_dispatch(
        source, spec, objective_mode="peak_shaving", duration_hours=0.5
    )
    assert result.objective_breakdown["throughput_mwh"] <= 20.0 + 1e-5
    assert result.schedule["soc_end_mwh"].iloc[-1] == pytest.approx(50)
