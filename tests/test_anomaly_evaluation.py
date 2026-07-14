"""Synthetic injection and metric evaluation tests."""

from __future__ import annotations

import pandas as pd
import pytest

from gridmind.anomalies.contracts import make_anomaly
from gridmind.anomalies.evaluation import (
    INJECTION_TYPES,
    evaluate_detections,
    inject_synthetic_anomalies,
)


def _history(periods: int = 120) -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01", periods=periods, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "region": "PJM",
            "timestamp_utc": timestamps,
            "demand_mw": [100 + index % 24 for index in range(periods)],
            "solar_generation_mw": [
                max(0, 20 - abs(timestamp.hour - 12) * 3) for timestamp in timestamps
            ],
            "wind_generation_mw": [30 + index % 5 for index in range(periods)],
            "relative_humidity_pct": 50.0,
        }
    )


def test_all_injection_types_are_deterministic_and_leave_original_unchanged() -> None:
    original = _history()
    snapshot = original.copy(deep=True)
    first = inject_synthetic_anomalies(
        original, target="demand_mw", seed=9, anomaly_types=INJECTION_TYPES
    )
    second = inject_synthetic_anomalies(
        original, target="demand_mw", seed=9, anomaly_types=INJECTION_TYPES
    )
    pd.testing.assert_frame_equal(original, snapshot)
    pd.testing.assert_frame_equal(first.frame, second.frame)
    pd.testing.assert_frame_equal(first.labels, second.labels)
    assert set(first.labels["anomaly_type"]) == set(INJECTION_TYPES)
    assert first.labels["synthetic"].all()
    assert len(first.frame) == len(original) - 2
    with pytest.raises(ValueError, match="Unsupported"):
        inject_synthetic_anomalies(original, target="demand_mw", anomaly_types=("real_incident",))
    with pytest.raises(ValueError, match="enough history"):
        inject_synthetic_anomalies(original.head(10), target="demand_mw")


def test_evaluation_metrics_delay_false_positives_and_per_type() -> None:
    labels = pd.DataFrame(
        [
            {
                "injection_id": "one",
                "region": "PJM",
                "target": "demand_mw",
                "anomaly_type": "single_hour_demand_spike",
                "start_utc": pd.Timestamp("2025-01-01T05:00:00Z"),
                "end_utc": pd.Timestamp("2025-01-01T05:00:00Z"),
                "magnitude": 1.0,
                "expected_severity": "critical",
                "synthetic": True,
            },
            {
                "injection_id": "two",
                "region": "PJM",
                "target": "demand_mw",
                "anomaly_type": "gradual_drift",
                "start_utc": pd.Timestamp("2025-01-01T10:00:00Z"),
                "end_utc": pd.Timestamp("2025-01-01T12:00:00Z"),
                "magnitude": 0.5,
                "expected_severity": "warning",
                "synthetic": True,
            },
        ]
    )
    detected = pd.DataFrame(
        [
            make_anomaly(
                region="PJM",
                target="demand_mw",
                timestamp="2025-01-01T05:00:00Z",
                detector_name="rules",
                anomaly_type="demand_spike",
                anomaly_score=90,
                severity="critical",
                explanation="spike",
            ),
            make_anomaly(
                region="PJM",
                target="demand_mw",
                timestamp="2025-01-01T11:00:00Z",
                detector_name="rules",
                anomaly_type="demand_spike",
                anomaly_score=45,
                severity="warning",
                explanation="drift",
            ),
            make_anomaly(
                region="PJM",
                target="demand_mw",
                timestamp="2025-01-01T20:00:00Z",
                detector_name="rules",
                anomaly_type="flatline",
                anomaly_score=45,
                severity="warning",
                explanation="false positive",
            ),
        ]
    )
    result = evaluate_detections(
        detected,
        labels,
        evaluation_start="2025-01-01T00:00:00Z",
        evaluation_end="2025-01-02T00:00:00Z",
    )
    assert result.overall_metrics["precision"] == pytest.approx(2 / 3)
    assert result.overall_metrics["recall"] == 1.0
    assert result.overall_metrics["f1"] == pytest.approx(0.8)
    assert result.overall_metrics["false_positives_per_day"] == 1.0
    assert result.overall_metrics["mean_detection_delay_hours"] == 0.5
    assert result.overall_metrics["severity_accuracy"] == 1.0
    assert result.per_type_metrics["detected"].sum() == 2
