"""Anomaly and alert persistence plus lifecycle regressions."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from gridmind.alerts.lifecycle import AlertManager, empty_lifecycle_counts
from gridmind.alerts.storage import AlertStorage
from gridmind.anomalies.contracts import make_anomaly
from gridmind.anomalies.storage import AnomalyStorage
from gridmind.exceptions import AlertLifecycleError


def _event(timestamp: str, *, severity: str = "warning") -> dict[str, object]:
    return make_anomaly(  # type: ignore[arg-type]
        region="PJM",
        target="demand_mw",
        timestamp=pd.Timestamp(timestamp),
        detector_name="rules",
        anomaly_type="demand_spike",
        anomaly_score=80 if severity == "critical" else 45,
        severity=severity,
        observed_value=150,
        expected_value=100,
        residual=50,
        explanation="Demand changed abruptly.",
        detected_at=pd.Timestamp(timestamp),
        metadata={"source": "test"},
    )


def test_anomaly_storage_is_utc_json_idempotent_and_filterable(tmp_path: Path) -> None:
    storage = AnomalyStorage(tmp_path / "grid.duckdb")
    events = pd.DataFrame(
        [
            _event("2026-01-01T00:00:00+00:00"),
            make_anomaly(
                region="MISO",
                target="wind_generation_mw",
                timestamp="2026-01-01T06:30:00+05:30",
                detector_name="isolation_forest",
                anomaly_type="multivariate_outlier",
                anomaly_score=20,
                severity="info",
                explanation="joint deviation",
                metadata={"decision": -0.1},
            ),
        ]
    )
    assert storage.upsert(events) == 2
    assert storage.upsert(events) == 2
    selected = storage.read(region="MISO", severity="info", detector="isolation_forest")
    assert len(selected) == 1
    assert selected.iloc[0]["timestamp_utc"] == pd.Timestamp("2026-01-01T01:00:00Z")
    assert json.loads(selected.iloc[0]["metadata_json"])["decision"] == -0.1
    assert storage.read(start="2026-01-01T00:30:00Z", end="2026-01-01T01:30:00Z").shape[0] == 1


def test_alert_dedup_escalation_acknowledgement_auto_resolution_and_history(
    tmp_path: Path,
) -> None:
    storage = AlertStorage(tmp_path / "grid.duckdb")
    manager = AlertManager(storage, dedup_hours=6, auto_resolve_hours=24)
    first = pd.DataFrame([_event("2026-01-01T00:00:00Z")])
    second = pd.DataFrame([_event("2026-01-01T02:00:00Z", severity="critical")])
    assert manager.process(first) == {**empty_lifecycle_counts(), "opened": 1, "total": 1}
    history_after_open = len(storage.read_history())
    assert manager.process(second) == {**empty_lifecycle_counts(), "updated": 1, "total": 1}
    assert len(storage.read_history()) == history_after_open + 1
    alert_count = storage.count()
    history_count = len(storage.read_history())
    replay = manager.process(pd.concat([first, second], ignore_index=True))
    assert replay == {**empty_lifecycle_counts(), "unchanged": 2, "total": 1}
    assert storage.count() == alert_count
    assert len(storage.read_history()) == history_count
    alert = storage.read_alerts().iloc[0]
    assert alert["occurrence_count"] == 2
    assert alert["severity"] == "critical"
    assert str(alert["first_seen_utc"].tz) == "UTC"
    manager.update_status(str(alert["alert_id"]), "acknowledged", now="2026-01-01T03:00:00Z")
    acknowledged = storage.read_alerts(status="acknowledged")
    assert len(acknowledged) == 1
    assert pd.notna(acknowledged.iloc[0]["acknowledged_at_utc"])
    assert manager.auto_resolve(now="2026-01-02T03:00:00Z") == 1
    resolved = storage.read_alerts(status="resolved")
    assert len(resolved) == 1
    history = storage.read_history(str(alert["alert_id"]))
    assert list(history["change_reason"]) == [
        "opened",
        "occurrence",
        "acknowledged",
        "auto_resolved",
    ]
    assert history["history_id"].is_unique


def test_alert_updated_at_only_and_duplicate_history_are_idempotent(tmp_path: Path) -> None:
    storage = AlertStorage(tmp_path / "grid.duckdb")
    manager = AlertManager(storage)
    manager.process(pd.DataFrame([_event("2026-01-01T00:00:00Z")]))
    alert = storage.read_alerts().iloc[0]
    history = storage.read_history()
    manager.update_status(str(alert["alert_id"]), "open", now="2026-01-01T06:00:00Z")
    assert len(storage.read_history()) == len(history)
    assert storage.read_alerts().iloc[0]["updated_at_utc"] == alert["updated_at_utc"]
    assert storage.append_history(history) == len(history)
    assert len(storage.read_history()) == len(history)


def test_acknowledgement_and_manual_resolution_each_create_one_history_row(
    tmp_path: Path,
) -> None:
    storage = AlertStorage(tmp_path / "grid.duckdb")
    manager = AlertManager(storage)
    manager.process(pd.DataFrame([_event("2026-01-01T00:00:00Z")]))
    alert_id = str(storage.read_alerts().iloc[0]["alert_id"])
    before = len(storage.read_history())
    manager.update_status(alert_id, "acknowledged", now="2026-01-01T01:00:00Z")
    assert len(storage.read_history()) == before + 1
    manager.update_status(alert_id, "resolved", now="2026-01-01T02:00:00Z")
    assert len(storage.read_history()) == before + 2
    assert list(storage.read_history()["change_reason"][-2:]) == ["acknowledged", "resolved"]


def test_alert_new_window_manual_resolution_suppression_and_invalid_transition(
    tmp_path: Path,
) -> None:
    storage = AlertStorage(tmp_path / "grid.duckdb")
    manager = AlertManager(storage, dedup_hours=1, auto_resolve_hours=24)
    manager.process(pd.DataFrame([_event("2026-01-01T00:00:00Z"), _event("2026-01-01T03:00:00Z")]))
    alerts = storage.read_alerts(region="PJM", target="demand_mw")
    assert len(alerts) == 2
    alert_id = str(alerts.iloc[0]["alert_id"])
    manager.update_status(alert_id, "suppressed", now="2026-01-01T04:00:00Z")
    assert len(storage.read_alerts(status="suppressed", severity="warning")) == 1
    manager.update_status(alert_id, "resolved", now="2026-01-01T05:00:00Z")
    with pytest.raises(AlertLifecycleError, match="Unsupported"):
        manager.update_status(alert_id, "deleted")
    with pytest.raises(AlertLifecycleError, match="not found"):
        manager.update_status("missing", "resolved")
