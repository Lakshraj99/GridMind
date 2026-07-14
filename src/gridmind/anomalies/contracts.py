"""Canonical, serializable anomaly event contract."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC
from typing import Any, Literal

import numpy as np
import pandas as pd

from gridmind.exceptions import AnomalyDetectionError
from gridmind.time_utils import format_utc_timestamp, to_utc_timestamp

Severity = Literal["info", "warning", "critical"]

SEVERITIES = ("info", "warning", "critical")
ANOMALY_TYPES = (
    "missing_timestamp",
    "duplicate_timestamp",
    "non_monotonic_timestamp",
    "unexpected_frequency",
    "invalid_value",
    "demand_spike",
    "demand_drop",
    "renewable_drop",
    "flatline",
    "forecast_residual",
    "unexpected_demand_spike",
    "unexpected_demand_drop",
    "solar_underproduction",
    "wind_generation_drop",
    "abnormal_net_load",
    "multivariate_outlier",
    "weather_grid_mismatch",
    "coverage_mismatch",
    "stale_observation",
)
ANOMALY_COLUMNS = [
    "anomaly_id",
    "region",
    "target",
    "timestamp_utc",
    "detector_name",
    "detector_version",
    "anomaly_type",
    "anomaly_score",
    "severity",
    "observed_value",
    "expected_value",
    "residual",
    "threshold",
    "feature_summary",
    "explanation",
    "forecast_origin",
    "model_name",
    "model_version",
    "run_id",
    "detected_at_utc",
    "metadata_json",
]


def empty_anomaly_frame() -> pd.DataFrame:
    """Return an empty frame with the canonical columns."""
    return pd.DataFrame(columns=ANOMALY_COLUMNS)


def deterministic_anomaly_id(
    region: str,
    target: str,
    timestamp: object,
    detector_name: str,
    anomaly_type: str,
    detector_version: str = "1",
) -> str:
    """Create a stable identifier for an anomaly's natural key."""
    material = "|".join(
        (
            region,
            target,
            format_utc_timestamp(timestamp),
            detector_name,
            detector_version,
            anomaly_type,
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


def make_anomaly(
    *,
    region: str,
    target: str,
    timestamp: object,
    detector_name: str,
    anomaly_type: str,
    anomaly_score: float,
    severity: Severity,
    explanation: str,
    observed_value: float | None = None,
    expected_value: float | None = None,
    residual: float | None = None,
    threshold: float | None = None,
    feature_summary: dict[str, Any] | None = None,
    forecast_origin: object | None = None,
    model_name: str = "",
    model_version: str = "",
    run_id: str = "",
    detector_version: str = "1",
    detected_at: object | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one JSON-safe canonical anomaly event."""
    timestamp_utc = to_utc_timestamp(timestamp)
    return {
        "anomaly_id": deterministic_anomaly_id(
            region, target, timestamp_utc, detector_name, anomaly_type, detector_version
        ),
        "region": region,
        "target": target,
        "timestamp_utc": timestamp_utc,
        "detector_name": detector_name,
        "detector_version": detector_version,
        "anomaly_type": anomaly_type,
        "anomaly_score": float(np.clip(anomaly_score, 0.0, 100.0)),
        "severity": severity,
        "observed_value": observed_value,
        "expected_value": expected_value,
        "residual": residual,
        "threshold": threshold,
        "feature_summary": json.dumps(feature_summary or {}, sort_keys=True, allow_nan=False),
        "explanation": explanation,
        "forecast_origin": to_utc_timestamp(forecast_origin)
        if forecast_origin is not None
        else pd.NaT,
        "model_name": model_name,
        "model_version": model_version,
        "run_id": run_id,
        "detected_at_utc": to_utc_timestamp(detected_at or pd.Timestamp.now(tz=UTC)),
        "metadata_json": json.dumps(metadata or {}, sort_keys=True, allow_nan=False),
    }


def validate_anomaly_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and canonicalize events before persistence or artifact writing."""
    missing = set(ANOMALY_COLUMNS).difference(frame.columns)
    if missing:
        raise AnomalyDetectionError(f"Anomaly events are missing columns: {sorted(missing)}")
    result = frame[ANOMALY_COLUMNS].copy()
    if result.empty:
        return result
    if not set(result["severity"]).issubset(SEVERITIES):
        raise AnomalyDetectionError("Anomaly events contain an unsupported severity.")
    if not set(result["anomaly_type"]).issubset(ANOMALY_TYPES):
        raise AnomalyDetectionError("Anomaly events contain an unsupported anomaly type.")
    for column in ("timestamp_utc", "forecast_origin", "detected_at_utc"):
        result[column] = pd.to_datetime(result[column], utc=True, errors="coerce")
    if result[["timestamp_utc", "detected_at_utc"]].isna().any().any():
        raise AnomalyDetectionError("Anomaly timestamps must be valid UTC instants.")
    scores = pd.to_numeric(result["anomaly_score"], errors="coerce")
    if scores.isna().any() or not scores.between(0, 100).all():
        raise AnomalyDetectionError("Anomaly scores must be finite values from 0 to 100.")
    for column in ("feature_summary", "metadata_json"):
        try:
            result[column].map(json.loads)
        except (TypeError, json.JSONDecodeError) as exc:
            raise AnomalyDetectionError(f"{column} must contain valid JSON.") from exc
    if result["anomaly_id"].duplicated().any():
        raise AnomalyDetectionError("Anomaly batch contains duplicate anomaly identifiers.")
    return result.sort_values(["timestamp_utc", "region", "target", "detector_name"]).reset_index(
        drop=True
    )
