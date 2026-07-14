"""Application boundary for alert listing and lifecycle transitions."""

from __future__ import annotations

import pandas as pd

from gridmind.alerts.lifecycle import AlertManager
from gridmind.alerts.storage import AlertStorage
from gridmind.config import Settings


def list_alerts(
    settings: Settings,
    *,
    region: str | None = None,
    target: str | None = None,
    status: str | None = None,
    severity: str | None = None,
) -> pd.DataFrame:
    return AlertStorage(settings.duckdb_path).read_alerts(
        region=region, target=target, status=status, severity=severity
    )


def update_alert_status(settings: Settings, *, alert_id: str, status: str) -> pd.Series:
    manager = AlertManager(
        AlertStorage(settings.duckdb_path),
        dedup_hours=settings.alert_dedup_window_hours,
        auto_resolve_hours=settings.alert_auto_resolve_hours,
    )
    return manager.update_status(alert_id, status)
