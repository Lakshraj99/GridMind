"""Canonical alert and alert-history contracts."""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from gridmind.exceptions import AlertLifecycleError
from gridmind.time_utils import format_utc_timestamp

ALERT_STATUSES = ("open", "acknowledged", "resolved", "suppressed")
ALERT_COLUMNS = [
    "alert_id",
    "region",
    "target",
    "anomaly_type",
    "severity",
    "status",
    "first_seen_utc",
    "last_seen_utc",
    "occurrence_count",
    "latest_anomaly_id",
    "title",
    "summary",
    "acknowledged_at_utc",
    "resolved_at_utc",
    "created_at_utc",
    "updated_at_utc",
    "metadata_json",
]
HISTORY_COLUMNS = [
    "history_id",
    "alert_id",
    "status",
    "severity",
    "changed_at_utc",
    "change_reason",
    "anomaly_id",
    "metadata_json",
]
MEANINGFUL_ALERT_COLUMNS = [
    "status",
    "severity",
    "first_seen_utc",
    "last_seen_utc",
    "occurrence_count",
    "latest_anomaly_id",
    "title",
    "summary",
    "acknowledged_at_utc",
    "resolved_at_utc",
    "metadata_json",
]


def deterministic_alert_id(region: str, target: str, anomaly_type: str, first_seen: object) -> str:
    material = f"{region}|{target}|{anomaly_type}|{format_utc_timestamp(first_seen)}"
    return hashlib.sha256(material.encode()).hexdigest()[:32]


def normalized_alert_state(alert: dict[str, object]) -> dict[str, object]:
    """Return only business-state fields in stable JSON-comparable form."""
    normalized: dict[str, object] = {}
    timestamp_columns = {
        "first_seen_utc",
        "last_seen_utc",
        "acknowledged_at_utc",
        "resolved_at_utc",
    }
    for column in MEANINGFUL_ALERT_COLUMNS:
        value = alert.get(column)
        if column in timestamp_columns:
            normalized[column] = (
                format_utc_timestamp(value)
                if not pd.Series([value], dtype="object").isna().iloc[0]
                else None
            )
        elif column == "occurrence_count":
            normalized[column] = int(str(value)) if value is not None else 0
        elif column == "metadata_json":
            normalized[column] = json.dumps(
                json.loads(str(value or "{}")), sort_keys=True, separators=(",", ":")
            )
        else:
            normalized[column] = value
    return normalized


def alert_state_fingerprint(alert: dict[str, object] | None) -> str:
    """Hash normalized persisted state, intentionally excluding ``updated_at_utc``."""
    if alert is None:
        return "none"
    payload = json.dumps(normalized_alert_state(alert), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_alert_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(ALERT_COLUMNS).difference(frame.columns)
    if missing:
        raise AlertLifecycleError(f"Alerts are missing columns: {sorted(missing)}")
    result = frame[ALERT_COLUMNS].copy()
    if result.empty:
        return result
    if not set(result["status"]).issubset(ALERT_STATUSES):
        raise AlertLifecycleError("Alert contains an unsupported status.")
    if not set(result["severity"]).issubset({"info", "warning", "critical"}):
        raise AlertLifecycleError("Alert contains an unsupported severity.")
    for column in (
        "first_seen_utc",
        "last_seen_utc",
        "acknowledged_at_utc",
        "resolved_at_utc",
        "created_at_utc",
        "updated_at_utc",
    ):
        result[column] = pd.to_datetime(result[column], utc=True, errors="coerce")
    if (
        result[["first_seen_utc", "last_seen_utc", "created_at_utc", "updated_at_utc"]]
        .isna()
        .any()
        .any()
    ):
        raise AlertLifecycleError("Required alert timestamps must be valid UTC instants.")
    if (pd.to_numeric(result["occurrence_count"], errors="coerce") < 1).any():
        raise AlertLifecycleError("Alert occurrence count must be positive.")
    try:
        result["metadata_json"].map(json.loads)
    except (TypeError, json.JSONDecodeError) as exc:
        raise AlertLifecycleError("Alert metadata must contain valid JSON.") from exc
    return result.sort_values(["last_seen_utc", "alert_id"]).reset_index(drop=True)
