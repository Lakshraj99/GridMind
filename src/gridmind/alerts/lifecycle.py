"""Idempotent alert deduplication, escalation, and state transitions."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC
from typing import Any

import pandas as pd

from gridmind.alerts.contracts import ALERT_COLUMNS, deterministic_alert_id
from gridmind.alerts.storage import AlertStorage
from gridmind.anomalies.contracts import validate_anomaly_frame
from gridmind.exceptions import AlertLifecycleError
from gridmind.time_utils import to_utc_timestamp

SEVERITY_RANK = {"info": 0, "warning": 1, "critical": 2}


class AlertManager:
    def __init__(
        self, storage: AlertStorage, *, dedup_hours: int = 6, auto_resolve_hours: int = 24
    ) -> None:
        self.storage = storage
        self.dedup_window = pd.Timedelta(hours=dedup_hours)
        self.auto_resolve_window = pd.Timedelta(hours=auto_resolve_hours)

    def process(self, anomalies: pd.DataFrame) -> dict[str, int]:
        events = validate_anomaly_frame(anomalies)
        existing = self.storage.read_alerts()
        existing_records: list[dict[str, Any]] = [
            {str(key): value for key, value in record.items()}
            for record in existing.to_dict(orient="records")
        ]
        alerts = {str(record["alert_id"]): record for record in existing_records}
        opened = updated = 0
        history: list[dict[str, object]] = []
        for anomaly in events.itertuples(index=False):
            match = self._dedup_match(alerts, anomaly)
            if match is None:
                record = self._new_alert(anomaly)
                alerts[str(record["alert_id"])] = record
                history.append(self._history(record, "opened", str(anomaly.anomaly_id)))
                opened += 1
                continue
            if str(match["latest_anomaly_id"]) == str(anomaly.anomaly_id):
                continue
            match["last_seen_utc"] = anomaly.timestamp_utc
            match["latest_anomaly_id"] = anomaly.anomaly_id
            match["occurrence_count"] = int(match["occurrence_count"]) + 1
            if SEVERITY_RANK[str(anomaly.severity)] > SEVERITY_RANK[str(match["severity"])]:
                match["severity"] = anomaly.severity
            match["updated_at_utc"] = anomaly.detected_at_utc
            match["summary"] = anomaly.explanation
            history.append(self._history(match, "occurrence", str(anomaly.anomaly_id)))
            updated += 1
        if alerts:
            self.storage.upsert_alerts(pd.DataFrame(alerts.values())[ALERT_COLUMNS])
        if history:
            self.storage.append_history(pd.DataFrame(history))
        return {"opened": opened, "updated": updated, "total": len(alerts)}

    def update_status(
        self, alert_id: str, status: str, *, now: object | None = None, reason: str = "manual"
    ) -> pd.Series:
        if status not in {"open", "acknowledged", "resolved", "suppressed"}:
            raise AlertLifecycleError(f"Unsupported alert status '{status}'.")
        alerts = self.storage.read_alerts()
        selected = alerts.loc[alerts["alert_id"] == alert_id]
        if selected.empty:
            raise AlertLifecycleError(f"Alert '{alert_id}' was not found.")
        timestamp = to_utc_timestamp(now or pd.Timestamp.now(tz=UTC))
        row = selected.iloc[0].copy()
        row["status"] = status
        row["updated_at_utc"] = timestamp
        if status == "acknowledged":
            row["acknowledged_at_utc"] = timestamp
        if status == "resolved":
            row["resolved_at_utc"] = timestamp
        self.storage.upsert_alerts(pd.DataFrame([row])[ALERT_COLUMNS])
        self.storage.append_history(
            pd.DataFrame(
                [
                    self._history(
                        {str(key): value for key, value in row.to_dict().items()},
                        reason,
                        str(row["latest_anomaly_id"]),
                    )
                ]
            )
        )
        return row

    def auto_resolve(self, *, now: object | None = None) -> int:
        timestamp = to_utc_timestamp(now or pd.Timestamp.now(tz=UTC))
        active = self.storage.read_alerts()
        eligible = active.loc[
            active["status"].isin(["open", "acknowledged"])
            & ((timestamp - active["last_seen_utc"]) >= self.auto_resolve_window)
        ]
        for alert_id in eligible["alert_id"]:
            self.update_status(str(alert_id), "resolved", now=timestamp, reason="healthy_period")
        return len(eligible)

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
            and abs(anomaly.timestamp_utc - alert["last_seen_utc"]) <= self.dedup_window
        ]
        return max(candidates, key=lambda item: item["last_seen_utc"]) if candidates else None

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
    def _history(alert: dict[str, Any], reason: str, anomaly_id: str) -> dict[str, object]:
        changed = to_utc_timestamp(alert["updated_at_utc"])
        material = f"{alert['alert_id']}|{reason}|{anomaly_id}|{changed.isoformat()}"
        return {
            "history_id": hashlib.sha256(material.encode()).hexdigest()[:32],
            "alert_id": alert["alert_id"],
            "status": alert["status"],
            "severity": alert["severity"],
            "changed_at_utc": changed,
            "change_reason": reason,
            "anomaly_id": anomaly_id,
            "metadata_json": "{}",
        }
