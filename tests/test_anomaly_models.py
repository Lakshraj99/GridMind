"""Residual, IsolationForest, ensemble, and severity tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridmind.anomalies.contracts import make_anomaly
from gridmind.anomalies.ensemble import combine_detector_events
from gridmind.anomalies.multivariate import IsolationForestConfig, MultivariateDetector
from gridmind.anomalies.residuals import (
    ResidualConfig,
    ResidualDetector,
    align_actuals_and_forecasts,
)
from gridmind.anomalies.severity import severity_from_score, severity_score
from gridmind.exceptions import AnomalyDetectionError


def _residual_data(target: str = "demand_mw") -> tuple[pd.DataFrame, pd.DataFrame]:
    timestamps = pd.date_range("2026-01-01", periods=40, freq="h", tz="UTC")
    residuals = np.resize(np.array([-2.0, -1.0, 1.0, 2.0]), len(timestamps))
    residuals[-1] = 50.0
    actual = pd.DataFrame({"region": "PJM", "timestamp_utc": timestamps, target: 100.0})
    forecasts = pd.DataFrame(
        {
            "region": "PJM",
            "target": target,
            "forecast_origin": timestamps - pd.Timedelta(hours=1),
            "timestamp_utc": timestamps,
            "forecast_step": 1,
            "predicted_value": 100.0 - residuals,
            "model_name": "model",
            "model_version": "1",
            "run_id": "run",
            "weather_mode": "realistic_forecast",
            "created_at_utc": timestamps,
        }
    )
    return actual, forecasts


def test_residual_alignment_latest_origin_version_and_prior_only_statistics() -> None:
    actual, forecasts = _residual_data()
    older = forecasts.copy()
    older["forecast_origin"] -= pd.Timedelta(hours=1)
    older["predicted_value"] = 0.0
    older["model_version"] = "2"
    aligned = align_actuals_and_forecasts(
        actual, pd.concat([older, forecasts]), target="demand_mw", model_version="1"
    )
    assert len(aligned) == len(actual)
    assert (aligned["model_version"] == "1").all()
    result = ResidualDetector(ResidualConfig(min_history=12, window=24)).detect(
        actual, forecasts, target="demand_mw"
    )
    assert result.insufficient_history_rows == 12
    assert result.scored_rows.loc[20, "rolling_residual_mean"] == pytest.approx(
        result.scored_rows.loc[:19, "residual"].tail(24).mean()
    )
    anomaly = result.anomalies.iloc[-1]
    assert anomaly["timestamp_utc"] == actual["timestamp_utc"].iloc[-1]
    assert anomaly["anomaly_type"] == "unexpected_demand_spike"
    assert anomaly["forecast_origin"] < anomaly["timestamp_utc"]
    summary = json.loads(anomaly["feature_summary"])
    assert summary["zscore"] > 4


def test_residual_detector_skips_missing_values_and_nighttime_solar() -> None:
    actual, forecasts = _residual_data("solar_generation_mw")
    actual["timestamp_utc"] += pd.Timedelta(hours=9)
    forecasts["timestamp_utc"] += pd.Timedelta(hours=9)
    forecasts["forecast_origin"] += pd.Timedelta(hours=9)
    actual.loc[0, "solar_generation_mw"] = np.nan
    forecasts.loc[0, "predicted_value"] = np.nan
    result = ResidualDetector(ResidualConfig(min_history=4)).detect(
        actual, forecasts, target="solar_generation_mw"
    )
    assert len(result.scored_rows) == 39
    assert result.anomalies.empty


def _multivariate_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_times = pd.date_range("2026-01-01", periods=80, freq="h", tz="UTC")
    score_times = pd.date_range(train_times[-1] + pd.Timedelta(hours=1), periods=12, freq="h")
    training_parts = []
    scoring_parts = []
    for offset, region in enumerate(("PJM", "MISO")):
        training_parts.append(
            pd.DataFrame(
                {
                    "region": region,
                    "timestamp_utc": train_times,
                    "demand_mw": 100 + offset + np.sin(np.arange(80) / 5),
                    "temperature_c": 20 + np.cos(np.arange(80) / 7),
                }
            )
        )
        score = pd.DataFrame(
            {
                "region": region,
                "timestamp_utc": score_times,
                "demand_mw": 100 + offset + np.sin(np.arange(12) / 5),
                "temperature_c": 20 + np.cos(np.arange(12) / 7),
            }
        )
        score.loc[score.index[-2:], ["demand_mw", "temperature_c"]] = [1000, -100]
        scoring_parts.append(score)
    scoring_parts[0].loc[0, "temperature_c"] = np.nan
    return pd.concat(training_parts, ignore_index=True), pd.concat(scoring_parts, ignore_index=True)


def test_isolation_forest_is_deterministic_region_isolated_and_serializable(
    tmp_path: Path,
) -> None:
    training, scoring = _multivariate_frames()
    config = IsolationForestConfig(
        contamination=0.05,
        random_seed=7,
        min_training_rows=60,
        n_estimators=30,
        score_quantile=0.95,
    )
    first = MultivariateDetector(("demand_mw", "temperature_c"), config).fit(training)
    result = first.score(scoring)
    second = (
        MultivariateDetector(("demand_mw", "temperature_c"), config).fit(training).score(scoring)
    )
    assert result.training_rows == 160
    assert result.excluded_rows == 1
    assert result.gap_count == 0
    assert not result.anomalies.empty
    pd.testing.assert_series_equal(
        result.scored_rows["isolation_decision"],
        second.scored_rows["isolation_decision"],
        check_names=False,
    )
    assert set(result.anomalies["region"]) == {"PJM", "MISO"}
    path = first.save(tmp_path / "detector.joblib")
    loaded = MultivariateDetector.load(path)
    assert loaded.feature_names == ("demand_mw", "temperature_c")
    assert json.loads(path.with_suffix(".schema.json").read_text())["regions"] == ["MISO", "PJM"]
    with pytest.raises(AnomalyDetectionError, match="follow its training"):
        loaded.score(training.tail(2))


def test_isolation_forest_validates_features_and_region_contract() -> None:
    training, scoring = _multivariate_frames()
    detector = MultivariateDetector(
        ("demand_mw",), IsolationForestConfig(min_training_rows=60, n_estimators=10)
    ).fit(training)
    with pytest.raises(AnomalyDetectionError, match="missing features"):
        detector.score(scoring.drop(columns="demand_mw"))
    unknown = scoring.loc[scoring["region"] == "PJM"].copy()
    unknown["region"] = "ERCOT"
    with pytest.raises(AnomalyDetectionError, match="No fitted"):
        detector.score(unknown)


def test_isolation_forest_target_thresholds_and_excessive_rate_warning() -> None:
    training, scoring = _multivariate_frames()
    lower = MultivariateDetector(
        ("demand_mw", "temperature_c"),
        IsolationForestConfig(
            target="demand_mw",
            min_training_rows=60,
            n_estimators=20,
            score_quantile=0.80,
            maximum_anomaly_rate=0.01,
        ),
    ).fit(training)
    higher = MultivariateDetector(
        ("demand_mw", "temperature_c"),
        IsolationForestConfig(
            target="net_load_mw", min_training_rows=60, n_estimators=20, score_quantile=0.99
        ),
    ).fit(training)
    assert all(
        lower.score_thresholds[region] < higher.score_thresholds[region]
        for region in ("PJM", "MISO")
    )
    result = lower.score(scoring)
    assert result.calibration["target"] == "demand_mw"
    assert result.calibration["evaluated_row_count"] == len(scoring) - 1
    assert result.calibration["flagged_row_count"] == len(result.anomalies)
    assert result.calibration_warning is not None
    assert "exceeding" in result.calibration_warning


def test_ensemble_contributions_critical_override_and_score_bounds() -> None:
    timestamp = pd.Timestamp("2026-01-01T00:00:00Z")
    events = pd.DataFrame(
        [
            make_anomaly(
                region="PJM",
                target="demand_mw",
                timestamp=timestamp,
                detector_name="rules",
                anomaly_type="invalid_value",
                anomaly_score=90,
                severity="critical",
                explanation="negative",
            ),
            make_anomaly(
                region="PJM",
                target="demand_mw",
                timestamp=timestamp,
                detector_name="residual",
                anomaly_type="unexpected_demand_drop",
                anomaly_score=50,
                severity="warning",
                explanation="residual",
            ),
        ]
    )
    ensemble = combine_detector_events(events)
    assert ensemble.iloc[0]["severity"] == "critical"
    metadata = json.loads(ensemble.iloc[0]["metadata_json"])
    assert metadata["critical_override"] is True
    assert len(metadata["contributions"]) == 2
    assert severity_from_score(severity_score(magnitude=0)) == "info"
    assert (
        severity_from_score(severity_score(magnitude=1, detector_count=3, duration_hours=6))
        == "critical"
    )
    assert 0 <= severity_score(magnitude=-4, completeness=0) <= 100
