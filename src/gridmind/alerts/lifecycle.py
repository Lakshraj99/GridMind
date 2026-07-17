"""Idempotent alert deduplication, escalation, and state transitions."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC
from typing import Any

import pandas as pd

from gridmind.alerts.contracts import (
    ALERT_COLUMNS,
    alert_state_fingerprint,
    deterministic_alert_id,
)
from gridmind.alerts.storage import AlertStorage
from gridmind.anomalies.contracts import validate_anomaly_frame
from gridmind.exceptions import AlertLifecycleError
from gridmind.time_utils import to_utc_timestamp

SEVERITY_RANK = {"info": 0, "warning": 1, "critical": 2}
COUNT_KEYS = (
    "opened",
    "updated",
    "unchanged",
    "acknowledged",
    "resolved",
    "suppressed",
    "auto_resolved",
)


def empty_lifecycle_counts() -> dict[str, int]:
    return {key: 0 for key in COUNT_KEYS}


class AlertManager:
    def __init__(
        self, storage: AlertStorage, *, dedup_hours: int = 6, auto_resolve_hours: int = 24
    ) -> None:
        self.storage = storage
        self.dedup_window = pd.Timedelta(hours=dedup_hours)
        self.auto_resolve_window = pd.Timedelta(hours=auto_resolve_hours)

    def process(self, anomalies: pd.DataFrame) -> dict[str, int]:
        """Persist only committed business-state transitions for a batch."""
        events = validate_anomaly_frame(anomalies)
        existing = self.storage.read_alerts()
        records: list[dict[str, Any]] = [
            {str(key): value for key, value in record.items()}
            for record in existing.to_dict(orient="records")
        ]
        alerts = {str(record["alert_id"]): record for record in records}
        prior_history = self.storage.read_history()
        represented_anomalies = (
            set(prior_history["anomaly_id"].dropna().astype(str))
            if not prior_history.empty
            else set()
        )
        counts = empty_lifecycle_counts()
        changed: dict[str, dict[str, Any]] = {}
        history: list[dict[str, object]] = []
        for anomaly in events.itertuples(index=False):
            anomaly_id = str(anomaly.anomaly_id)
            if anomaly_id in represented_anomalies:
                counts["unchanged"] += 1
                continue
            match = self._dedup_match(alerts, anomaly)
            if match is None:
                proposed = self._new_alert(anomaly)
                alert_id = str(proposed["alert_id"])
                previous = alerts.get(alert_id)
                if previous is not None and self._same_state(previous, proposed):
                    counts["unchanged"] += 1
                    represented_anomalies.add(anomaly_id)
                    continue
                alerts[alert_id] = proposed
                changed[alert_id] = proposed
                action = "opened" if previous is None else "reopened"
                history.append(self._history(previous, proposed, action, anomaly_id))
                represented_anomalies.add(anomaly_id)
                counts["opened" if previous is None else "updated"] += 1
                continue
            previous = match.copy()
            proposed = self._proposed_occurrence(previous, anomaly)
            if self._same_state(previous, proposed):
                counts["unchanged"] += 1
                represented_anomalies.add(anomaly_id)
                continue
            alert_id = str(proposed["alert_id"])
            alerts[alert_id] = proposed
            changed[alert_id] = proposed
            history.append(self._history(previous, proposed, "occurrence", anomaly_id))
            represented_anomalies.add(anomaly_id)
            counts["updated"] += 1
        if changed:
            self.storage.upsert_alerts(pd.DataFrame(changed.values())[ALERT_COLUMNS])
        if history:
            self.storage.append_history(pd.DataFrame(history))
        counts["total"] = len(alerts)
        return counts

    def update_status(
        self, alert_id: str, status: str, *, now: object | None = None, reason: str | None = None
    ) -> pd.Series:
        if status not in {"open", "acknowledged", "resolved", "suppressed"}:
            raise AlertLifecycleError(f"Unsupported alert status '{status}'.")
        alerts = self.storage.read_alerts()
        selected = alerts.loc[alerts["alert_id"] == alert_id]
        if selected.empty:
            raise AlertLifecycleError(f"Alert '{alert_id}' was not found.")
        timestamp = to_utc_timestamp(now or pd.Timestamp.now(tz=UTC))
        previous = {str(key): value for key, value in selected.iloc[0].to_dict().items()}
        if str(previous["status"]) == status:
            return pd.Series(previous)
        proposed = previous.copy()
        proposed["status"] = status
        if status == "acknowledged":
            proposed["acknowledged_at_utc"] = timestamp
        elif status == "resolved":
            proposed["resolved_at_utc"] = timestamp
        elif status == "open":
            proposed["resolved_at_utc"] = pd.NaT
        if self._same_state(previous, proposed):
            return pd.Series(previous)
        proposed["updated_at_utc"] = timestamp
        action = (
            reason
            or {
                "acknowledged": "acknowledged",
                "resolved": "resolved",
                "suppressed": "suppressed",
                "open": "reopened",
            }[status]
        )
        self.storage.upsert_alerts(pd.DataFrame([proposed])[ALERT_COLUMNS])
        self.storage.append_history(
            pd.DataFrame(
                [
                    self._history(
                        previous,
                        proposed,
                        action,
                        str(proposed["latest_anomaly_id"]),
                    )
                ]
            )
        )
        return pd.Series(proposed)

    def auto_resolve(self, *, now: object | None = None) -> int:
        timestamp = to_utc_timestamp(now or pd.Timestamp.now(tz=UTC))
        active = self.storage.read_alerts()
        eligible = active.loc[
            active["status"].isin(["open", "acknowledged"])
            & ((timestamp - active["last_seen_utc"]) >= self.auto_resolve_window)
        ]
        resolved = 0
        for alert_id in eligible["alert_id"]:
            current = self.storage.read_alerts()
            before = current.loc[current["alert_id"] == alert_id].iloc[0]
            after = self.update_status(
                str(alert_id), "resolved", now=timestamp, reason="auto_resolved"
            )
            resolved += int(str(before["status"]) != str(after["status"]))
        return resolved

    def _dedup_match(
        self, alerts: dict[str, dict[str, Any]], anomaly: Any
    ) -> dict[str, Any] | None:
        candidates = [
            alert
            for alert in alerts.values()
            if alert["region"] == anomaly.region
            and alert["target"] == anomaly.target
            and alert["anomaly_type"] == anomaly.anomaly_type
            and alert["status"] in {"open", "acknowledged"}
            and pd.Timedelta(0)
            <= anomaly.timestamp_utc - alert["last_seen_utc"]
            <= self.dedup_window
        ]
        return max(candidates, key=lambda item: item["last_seen_utc"]) if candidates else None

    @staticmethod
    def _proposed_occurrence(previous: dict[str, Any], anomaly: Any) -> dict[str, Any]:
        proposed = previous.copy()
        proposed["last_seen_utc"] = anomaly.timestamp_utc
        proposed["latest_anomaly_id"] = anomaly.anomaly_id
        proposed["occurrence_count"] = int(previous["occurrence_count"]) + 1
        if SEVERITY_RANK[str(anomaly.severity)] > SEVERITY_RANK[str(previous["severity"])]:
            proposed["severity"] = anomaly.severity
            proposed["title"] = (
                f"{anomaly.severity.title()} "
                f"{anomaly.anomaly_type.replace('_', ' ')} in {anomaly.region}"
            )
        proposed["summary"] = anomaly.explanation
        proposed["updated_at_utc"] = anomaly.detected_at_utc
        return proposed

    @staticmethod
    def _new_alert(anomaly: Any) -> dict[str, Any]:
        alert_id = deterministic_alert_id(
            str(anomaly.region),
            str(anomaly.target),
            str(anomaly.anomaly_type),
            anomaly.timestamp_utc,
        )
        return {
            "alert_id": alert_id,
            "region": anomaly.region,
            "target": anomaly.target,
            "anomaly_type": anomaly.anomaly_type,
            "severity": anomaly.severity,
            "status": "open",
            "first_seen_utc": anomaly.timestamp_utc,
            "last_seen_utc": anomaly.timestamp_utc,
            "occurrence_count": 1,
            "latest_anomaly_id": anomaly.anomaly_id,
            "title": (
                f"{anomaly.severity.title()} "
                f"{anomaly.anomaly_type.replace('_', ' ')} in {anomaly.region}"
            ),
            "summary": anomaly.explanation,
            "acknowledged_at_utc": pd.NaT,
            "resolved_at_utc": pd.NaT,
            "created_at_utc": anomaly.detected_at_utc,
            "updated_at_utc": anomaly.detected_at_utc,
            "metadata_json": json.dumps({"detector": anomaly.detector_name}, sort_keys=True),
        }

    @staticmethod
    def _same_state(previous: dict[str, Any], proposed: dict[str, Any]) -> bool:
        return alert_state_fingerprint(previous) == alert_state_fingerprint(proposed)

    @staticmethod
    def _history(
        previous: dict[str, Any] | None,
        proposed: dict[str, Any],
        action: str,
        anomaly_id: str,
    ) -> dict[str, object]:
        previous_fingerprint = alert_state_fingerprint(previous)
        new_fingerprint = alert_state_fingerprint(proposed)
        material = f"{proposed['alert_id']}|{action}|{previous_fingerprint}|{new_fingerprint}"
        return {
            "history_id": hashlib.sha256(material.encode()).hexdigest()[:32],
            "alert_id": proposed["alert_id"],
            "status": proposed["status"],
            "severity": proposed["severity"],
            "changed_at_utc": to_utc_timestamp(proposed["updated_at_utc"]),
            "change_reason": action,
            "anomaly_id": anomaly_id,
            "metadata_json": json.dumps(
                {
                    "previous_state_fingerprint": previous_fingerprint,
                    "new_state_fingerprint": new_fingerprint,
                },
                sort_keys=True,
            ),
        }
