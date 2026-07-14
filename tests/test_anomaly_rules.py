"""Deterministic data-quality rule coverage."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from gridmind.anomalies.rules import RuleConfig, RuleDetector
from gridmind.config import Settings


def test_anomaly_configuration_defaults_and_threshold_validation() -> None:
    settings = Settings(_env_file=None)
    assert settings.anomaly_detection_enabled is True
    assert settings.anomaly_lookback_hours == 720
    assert settings.anomaly_contamination == 0.01
    assert settings.anomaly_experiment_name == "gridmind-anomaly-detection"
    with pytest.raises(ValidationError):
        Settings(ANOMALY_CONTAMINATION=0.5, _env_file=None)
    with pytest.raises(ValidationError):
        Settings(RESIDUAL_ZSCORE_WARNING=4, RESIDUAL_ZSCORE_CRITICAL=3, _env_file=None)
    with pytest.raises(ValidationError):
        Settings(MISSING_HOUR_WARNING_COUNT=3, MISSING_HOUR_CRITICAL_COUNT=3, _env_file=None)


def test_rules_detect_timestamp_value_change_flatline_and_weather_anomalies() -> None:
    timestamps = pd.date_range("2026-01-01", periods=12, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "region": "PJM",
            "timestamp_utc": timestamps,
            "demand_mw": [100, 101, 102, 180, 100, -1, 103, 104, 105, 105, 105, 105],
            "relative_humidity_pct": [50] * 6 + [150] + [50] * 5,
        }
    )
    frame = frame.drop(index=1)
    frame = pd.concat([frame, frame.iloc[[2]]], ignore_index=True)
    frame = pd.concat([frame.iloc[4:], frame.iloc[:4]], ignore_index=True)
    weather = pd.DataFrame({"region": "PJM", "timestamp_utc": timestamps.delete([1, 7])})
    result = RuleDetector(
        RuleConfig(
            demand_change_threshold=0.20,
            flatline_hours=4,
            missing_warning_count=1,
            missing_critical_count=3,
            stale_after_hours=48,
        )
    ).detect(
        frame,
        target="demand_mw",
        weather=weather,
        now=pd.Timestamp("2026-01-01T12:00:00Z"),
    )
    types = set(result["anomaly_type"])
    assert {
        "missing_timestamp",
        "unexpected_frequency",
        "duplicate_timestamp",
        "non_monotonic_timestamp",
        "invalid_value",
        "demand_spike",
        "demand_drop",
        "flatline",
        "weather_grid_mismatch",
    }.issubset(types)
    assert result["anomaly_id"].is_unique
    assert str(result["timestamp_utc"].dt.tz) == "UTC"
    assert result["anomaly_score"].between(0, 100).all()
    assert not any("repair" in explanation for explanation in result["explanation"])


def test_rules_isolate_regions_and_detect_renewable_drop_and_staleness() -> None:
    timestamps = pd.date_range("2026-01-01", periods=8, freq="h", tz="UTC")
    frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "region": "PJM",
                    "timestamp_utc": timestamps.delete(3),
                    "solar_generation_mw": [100, 100, 100, 20, 5, 6, 7],
                }
            ),
            pd.DataFrame(
                {
                    "region": "MISO",
                    "timestamp_utc": timestamps,
                    "solar_generation_mw": range(100, 108),
                }
            ),
        ],
        ignore_index=True,
    )
    result = RuleDetector(
        RuleConfig(renewable_drop_threshold=0.30, flatline_hours=3, stale_after_hours=2)
    ).detect(
        frame,
        target="solar_generation_mw",
        now=pd.Timestamp("2026-01-02T00:00:00Z"),
    )
    missing = result.loc[result["anomaly_type"] == "missing_timestamp"]
    assert set(missing["region"]) == {"PJM"}
    assert "renewable_drop" in set(result["anomaly_type"])
    assert set(result.loc[result["anomaly_type"] == "stale_observation", "region"]) == {
        "PJM",
        "MISO",
    }


def test_rules_detect_demand_renewable_and_forecast_weather_coverage() -> None:
    timestamps = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    demand = pd.DataFrame(
        {"region": "PJM", "timestamp_utc": timestamps, "demand_mw": [100, 101, 102, 103]}
    )
    renewable = pd.DataFrame({"region": "PJM", "timestamp_utc": timestamps.delete(2)})
    weather = pd.DataFrame({"region": "PJM", "timestamp_utc": timestamps.delete(3)})
    detector = RuleDetector(RuleConfig(flatline_hours=5, stale_after_hours=24))
    coverage = detector.detect(
        demand,
        target="demand_mw",
        weather=weather,
        renewables=renewable,
        now=pd.Timestamp("2026-01-01T04:00:00Z"),
    )
    assert set(coverage["anomaly_type"]) == {"weather_grid_mismatch", "coverage_mismatch"}
    forecasts = pd.DataFrame({"region": "PJM", "target": "demand_mw", "timestamp_utc": timestamps})
    forecast_coverage = detector.detect_forecast_weather_coverage(
        forecasts, weather, target="demand_mw", now=pd.Timestamp("2026-01-01T04:00:00Z")
    )
    assert len(forecast_coverage) == 1
    assert "forecast horizon" in forecast_coverage.iloc[0]["explanation"]


def test_rule_detector_rejects_missing_contract_columns(tmp_path: Path) -> None:
    del tmp_path
    with pytest.raises(ValueError, match="missing columns"):
        RuleDetector().detect(pd.DataFrame({"region": ["PJM"]}), target="demand_mw")
