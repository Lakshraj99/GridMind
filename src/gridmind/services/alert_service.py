"""Alert read and controlled lifecycle-write service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gridmind.alerts.lifecycle import AlertManager
from gridmind.alerts.storage import AlertStorage
from gridmind.config import Settings
from gridmind.exceptions import ResourceNotFoundError
from gridmind.services.common import DuckDBReadService, Page, TTLCache, frame_records, where_clause


class AlertService(DuckDBReadService):
    table = "grid_alerts"

    def __init__(self, path: Path | str, settings: Settings, *, cache_ttl: float = 30) -> None:
        super().__init__(path)
        self.settings = settings
        self.cache = TTLCache(cache_ttl)

    def _manager(self) -> AlertManager:
        return AlertManager(
            AlertStorage(self.path),
            dedup_hours=self.settings.alert_dedup_window_hours,
            auto_resolve_hours=self.settings.alert_auto_resolve_hours,
        )

    def list(self, *, limit: int, offset: int, **filters: object) -> Page:
        self.require_table(self.table)
        where, parameters = where_clause(
            [
                ("region =", filters.get("region")),
                ("target =", filters.get("target")),
                ("severity =", filters.get("severity")),
                ("status =", filters.get("status")),
                ("last_seen_utc >=", filters.get("start_time")),
                ("last_seen_utc <=", filters.get("end_time")),
            ]
        )
        count = self.query(f"SELECT COUNT(*) AS total FROM {self.table}{where}", parameters)
        frame = self.query(
            f"SELECT * FROM {self.table}{where} ORDER BY last_seen_utc DESC LIMIT ? OFFSET ?",
            [*parameters, limit, offset],
        )
        return Page(frame_records(frame), int(count.iloc[0]["total"]), limit, offset)

    def get(self, alert_id: str) -> dict[str, Any]:
        records = frame_records(
            self.query(f"SELECT * FROM {self.table} WHERE alert_id = ?", [alert_id])
        )
        if not records:
            raise ResourceNotFoundError(f"Alert '{alert_id}' was not found.")
        history = self.query(
            "SELECT * FROM alert_history WHERE alert_id = ? ORDER BY changed_at_utc", [alert_id]
        )
        records[0]["history"] = frame_records(history)
        return records[0]

    def update(self, alert_id: str, status: str) -> dict[str, Any]:
        self._manager().update_status(alert_id, status)
        self.cache.clear()
        return self.get(alert_id)
