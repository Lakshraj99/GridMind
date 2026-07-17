"""Anomaly query, detail, and calibration summary service."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from gridmind.exceptions import ResourceNotFoundError
from gridmind.services.common import DuckDBReadService, Page, TTLCache, frame_records, where_clause


class AnomalyService(DuckDBReadService):
    table = "anomaly_events"

    def __init__(
        self,
        path: Path | str,
        *,
        cache_ttl: float = 30,
        maximum_rate: float = 0.10,
        metrics: Any | None = None,
    ) -> None:
        super().__init__(path, metrics=metrics)
        self.cache = TTLCache(cache_ttl)
        self.maximum_rate = maximum_rate

    def list(self, *, limit: int, offset: int, **filters: object) -> Page:
        self.require_table(self.table)
        where, parameters = where_clause(
            [
                ("region =", filters.get("region")),
                ("target =", filters.get("target")),
                ("severity =", filters.get("severity")),
                ("detector_name =", filters.get("detector")),
                ("anomaly_type =", filters.get("anomaly_type")),
                ("timestamp_utc >=", filters.get("start_time")),
                ("timestamp_utc <=", filters.get("end_time")),
            ]
        )
        count = self.query(f"SELECT COUNT(*) AS total FROM {self.table}{where}", parameters)
        frame = self.query(
            f"SELECT * FROM {self.table}{where} "
            "ORDER BY timestamp_utc DESC, anomaly_id LIMIT ? OFFSET ?",
            [*parameters, limit, offset],
        )
        items = frame_records(frame)
        for item in items:
            item["review_status"] = "detection_requiring_human_review"
        return Page(items, int(count.iloc[0]["total"]), limit, offset)

    def get(self, anomaly_id: str) -> dict[str, Any]:
        self.require_table(self.table)
        records = frame_records(
            self.query(f"SELECT * FROM {self.table} WHERE anomaly_id = ?", [anomaly_id])
        )
        if not records:
            raise ResourceNotFoundError(f"Anomaly '{anomaly_id}' was not found.")
        records[0]["review_status"] = "detection_requiring_human_review"
        return records[0]

    def summary(self) -> dict[str, Any]:
        self.require_table(self.table)

        def build() -> dict[str, Any]:
            groups = self.query(
                f"SELECT target, severity, anomaly_type, detector_name, "
                f"CAST(timestamp_utc AS DATE) AS day, COUNT(*) AS count FROM {self.table} "
                "GROUP BY ALL ORDER BY day"
            )
            records = frame_records(groups)
            detector_count = len({row["detector_name"] for row in records})
            daily_rates = [
                {
                    "day": row["day"],
                    "target": row["target"],
                    "detector": row["detector_name"],
                    "event_count": row["count"],
                    "hourly_opportunity_rate": float(row["count"]) / 24.0,
                }
                for row in records
            ]
            agreement = self.query(
                f"SELECT COUNT(*) AS timestamp_groups, "
                "SUM(CASE WHEN detector_count > 1 THEN 1 ELSE 0 END) AS agreed_groups "
                f"FROM (SELECT region, target, timestamp_utc, "
                f"COUNT(DISTINCT detector_name) AS detector_count FROM {self.table} GROUP BY ALL)"
            ).iloc[0]
            warnings = (
                ["Daily anomaly rate exceeds the configured calibration maximum."]
                if any(
                    float(row["hourly_opportunity_rate"]) > self.maximum_rate for row in daily_rates
                )
                else []
            )
            if not records:
                warnings.append("No anomaly detections are available for calibration review.")
            return {
                "groups": records,
                "count_by_target": _counts(records, "target"),
                "count_by_severity": _counts(records, "severity"),
                "count_by_anomaly_type": _counts(records, "anomaly_type"),
                "daily_anomaly_rates": daily_rates,
                "detector_agreement": {
                    "distinct_detectors": detector_count,
                    "timestamp_groups": int(agreement["timestamp_groups"] or 0),
                    "agreed_groups": int(agreement["agreed_groups"] or 0),
                },
                "calibration_warnings": warnings,
                "disclaimer": (
                    "Anomalies are detections requiring human review, not confirmed incidents."
                ),
            }

        value, hit = self.cache.get_or_create("summary", build)
        if self.metrics is not None:
            self.metrics.cache.labels("hit" if hit else "miss").inc()
        return cast(dict[str, Any], value)


def _counts(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in records:
        key = str(row[field])
        result[key] = result.get(key, 0) + int(row["count"])
    return result
