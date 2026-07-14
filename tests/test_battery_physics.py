"""Battery contract, physics, baseline, and settings tests."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from pydantic import ValidationError

from gridmind.config import Settings
from gridmind.exceptions import BatteryOptimizationError
from gridmind.optimization.baselines import no_battery_baseline, rule_based_baseline
from gridmind.optimization.battery import (
    battery_specification_from_settings,
    soc_transition,
    validate_dispatch_physics,
)
from gridmind.optimization.contracts import BatterySpecification, ObjectiveWeights
from gridmind.optimization.evaluation import compare_strategies, evaluate_dispatch


def battery_spec(**updates: object) -> BatterySpecification:
    values: dict[str, object] = {
        "battery_id": "pjm-test",
        "capacity_mwh": 100.0,
        "max_charge_mw": 25.0,
        "max_discharge_mw": 20.0,
        "min_soc_mwh": 10.0,
        "max_soc_mwh": 100.0,
        "initial_soc_mwh": 50.0,
        "terminal_soc_target_mwh": 50.0,
        "charge_efficiency": 0.90,
        "discharge_efficiency": 0.80,
        "self_discharge_rate": 0.01,
        "maximum_daily_cycles": 1.5,
        "degradation_cost_per_mwh": 2.0,
        "reserve_soc_mwh": 20.0,
        "metadata": {"test": True},
    }
    values.update(updates)
    return BatterySpecification(**values)  # type: ignore[arg-type]


def dispatch_input(periods: int = 4, *, step: float = 1.0) -> pd.DataFrame:
    origin = pd.Timestamp("2026-01-01T00:00:00Z")
    timestamps = pd.date_range(
        origin + pd.Timedelta(hours=step), periods=periods, freq=pd.Timedelta(hours=step)
    )
    net = [50.0, 60.0, 130.0, 120.0][:periods]
    return pd.DataFrame(
        {
            "region": "PJM",
            "forecast_origin": origin,
            "timestamp_utc": timestamps,
            "forecast_step": range(1, periods + 1),
            "demand_forecast_mw": [value + 20 for value in net],
            "solar_forecast_mw": 10.0,
            "wind_forecast_mw": 10.0,
            "renewable_forecast_mw": 20.0,
            "net_load_before_battery_mw": net,
            "energy_price": [20.0, 25.0, 120.0, 100.0][:periods],
            "metadata_json": json.dumps({"offline": True}),
        }
    )


def test_soc_transition_efficiency_self_discharge_and_variable_duration() -> None:
    spec = battery_spec()
    expected = 50 * 0.99**0.5 + 10 * 0.9 * 0.5 - 4 / 0.8 * 0.5
    assert soc_transition(50, 10, 4, spec, 0.5) == pytest.approx(expected)
    with pytest.raises(BatteryOptimizationError, match="duration"):
        soc_transition(50, 0, 0, spec, 0)


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"capacity_mwh": 0}, "positive"),
        ({"min_soc_mwh": 100}, "SOC bounds"),
        ({"initial_soc_mwh": 5}, "initial SOC"),
        ({"terminal_soc_target_mwh": 101}, "terminal SOC"),
        ({"reserve_soc_mwh": 5}, "reserve SOC"),
        ({"charge_efficiency": 0}, "efficiencies"),
        ({"self_discharge_rate": 1}, "self-discharge"),
        ({"maximum_daily_cycles": 0}, "Cycle limit"),
    ],
)
def test_battery_specification_rejects_invalid_physics(
    updates: dict[str, object], message: str
) -> None:
    with pytest.raises(BatteryOptimizationError, match=message):
        battery_spec(**updates)
    with pytest.raises(BatteryOptimizationError, match="identifier"):
        battery_spec(battery_id=" ")


def test_settings_build_spec_and_validate_battery_configuration() -> None:
    settings = Settings(_env_file=None)
    spec = battery_specification_from_settings(settings, "bess-1")
    assert spec.reserve_soc_mwh == 50
    assert spec.metadata["illustrative_unless_operator_supplied"] is True
    assert spec.with_initial_soc(300).initial_soc_mwh == 300
    with pytest.raises(ValidationError):
        Settings(BATTERY_MIN_SOC_MWH=500, BATTERY_MAX_SOC_MWH=500, _env_file=None)
    with pytest.raises(ValidationError):
        Settings(BATTERY_MAX_SOC_MWH=600, BATTERY_CAPACITY_MWH=500, _env_file=None)
    with pytest.raises(ValidationError):
        Settings(BATTERY_INITIAL_SOC_MWH=20, BATTERY_MIN_SOC_MWH=50, _env_file=None)
    with pytest.raises(ValidationError):
        Settings(BATTERY_RESERVE_SOC_PCT=0.9, BATTERY_MAX_SOC_MWH=400, _env_file=None)
    with pytest.raises(BatteryOptimizationError, match="weights"):
        ObjectiveWeights(peak=-1)


def test_baselines_respect_constraints_and_evaluation_is_comparable() -> None:
    spec = battery_spec(self_discharge_rate=0, terminal_soc_target_mwh=50)
    source = dispatch_input()
    no_battery = no_battery_baseline(source, spec)
    rule = rule_based_baseline(source, spec)
    assert (no_battery["net_battery_power_mw"] == 0).all()
    assert not ((rule["charge_mw"] > 0) & (rule["discharge_mw"] > 0)).any()
    assert set(rule["operating_mode"]).issubset({"charge", "discharge", "idle"})
    assert validate_dispatch_physics(rule, spec, duration_hours=1, terminal_required=False)["valid"]
    metrics = evaluate_dispatch(rule, spec)
    assert metrics["total_throughput_mwh"] > 0
    assert metrics["power_violations"] == 0
    comparison = compare_strategies({"rule": rule, "none": no_battery}, spec)
    assert set(comparison["strategy"]) == {"rule", "none"}


def test_post_solve_validation_rejects_power_soc_continuity_cycle_and_terminal() -> None:
    spec = battery_spec(self_discharge_rate=0, terminal_soc_target_mwh=50)
    schedule = no_battery_baseline(dispatch_input(), spec)
    broken = schedule.copy()
    broken.loc[0, "charge_mw"] = 30
    broken.loc[0, "discharge_mw"] = 1
    broken.loc[0, "soc_end_mwh"] = 200
    with pytest.raises(BatteryOptimizationError, match="violates"):
        validate_dispatch_physics(broken, spec, duration_hours=1)
    terminal = schedule.copy()
    terminal.loc[terminal.index[-1], "soc_end_mwh"] = 49
    with pytest.raises(BatteryOptimizationError, match="violates"):
        validate_dispatch_physics(terminal, spec, duration_hours=1)
    empty = schedule.iloc[0:0]
    with pytest.raises(BatteryOptimizationError, match="empty"):
        validate_dispatch_physics(empty, spec, duration_hours=1)
