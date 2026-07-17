"""Forecast query and lineage service."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd

from gridmind.exceptions import ResourceNotFoundError
from gridmind.services.common import DuckDBReadService, Page, TTLCache, frame_records, where_clause


class ForecastService(DuckDBReadService):
    table = "target_forecasts"

    def __init__(
        self, path: Path | str, *, cache_ttl: float = 30, metrics: Any | None = None
    ) -> None:
        super().__init__(path, metrics=metrics)
        self.cache = TTLCache(cache_ttl)

    def list(self, *, limit: int, offset: int, **filters: object) -> Page:
        self.require_table(self.table)
        where, parameters = where_clause(
            [
                ("region =", filters.get("region")),
                ("target =", filters.get("target")),
                ("forecast_origin =", filters.get("forecast_origin")),
                ("timestamp_utc >=", filters.get("start_time")),
                ("timestamp_utc <=", filters.get("end_time")),
                ("model_name =", filters.get("model_name")),
                ("model_version =", filters.get("model_version")),
                ("weather_mode =", filters.get("weather_mode")),
            ]
        )
        count = self.query(f"SELECT COUNT(*) AS total FROM {self.table}{where}", parameters)
        frame = self.query(
            f"SELECT * FROM {self.table}{where} "
            "ORDER BY forecast_origin DESC, timestamp_utc LIMIT ? OFFSET ?",
            [*parameters, limit, offset],
        )
        records = frame_records(frame)
        alias = filters.get("model_alias")
        for record in records:
            record["lineage"] = {
                key: record.get(key) for key in ("model_name", "model_version", "run_id")
            }
            if alias is not None:
                record["lineage"]["requested_alias"] = alias
        return Page(records, int(count.iloc[0]["total"]), limit, offset)

    def latest(self, *, region: str, target: str, horizon: int, model_alias: str) -> dict[str, Any]:
        """Return the newest model series containing an exact contiguous UTC horizon."""
        self.require_table(self.table)
        candidates = self.query(
            f"SELECT forecast_origin, model_name, model_version, run_id, weather_mode "
            f"FROM {self.table} WHERE region = ? AND target = ? GROUP BY ALL "
            "HAVING COUNT(DISTINCT timestamp_utc) >= ? ORDER BY forecast_origin DESC",
            [region, target, horizon],
        )
        for candidate in candidates.to_dict(orient="records"):
            frame = self.query(
                f"SELECT * FROM {self.table} WHERE region = ? AND target = ? "
                "AND forecast_origin = ? AND model_name = ? AND model_version = ? "
                "AND run_id = ? AND weather_mode = ? ORDER BY timestamp_utc LIMIT ?",
                [
                    region,
                    target,
                    candidate["forecast_origin"],
                    candidate["model_name"],
                    candidate["model_version"],
                    candidate["run_id"],
                    candidate["weather_mode"],
                    horizon,
                ],
            )
            origin = pd.to_datetime(candidate["forecast_origin"], utc=True)
            expected = pd.date_range(
                origin + pd.Timedelta(hours=1), periods=horizon, freq="h", tz="UTC"
            )
            actual = pd.DatetimeIndex(pd.to_datetime(frame["timestamp_utc"], utc=True))
            if list(actual) != list(expected):
                continue
            lineage = {
                "model_name": candidate["model_name"],
                "model_version": candidate["model_version"],
                "run_id": candidate["run_id"],
                "requested_alias": model_alias,
            }
            items = frame_records(frame)
            for item in items:
                item["lineage"] = lineage
            return Page(items, horizon, horizon, 0).as_dict() | {"lineage": lineage}
        raise ResourceNotFoundError(
            "No complete contiguous forecast horizon was found for the requested filters."
        )

    def summary(self) -> dict[str, Any]:
        self.require_table(self.table)

        def build() -> dict[str, Any]:
            frame = self.query(
                f"SELECT region, target, COUNT(*) AS row_count, "
                f"MIN(timestamp_utc) AS start_time, MAX(timestamp_utc) AS end_time, "
                f"MAX(forecast_origin) AS latest_forecast_origin, "
                f"LIST(DISTINCT model_version) AS model_versions FROM {self.table} "
                "GROUP BY region, target ORDER BY region, target"
            )
            groups = frame_records(frame)
            return {
                "available_regions": sorted({str(row["region"]) for row in groups}),
                "available_targets": sorted({str(row["target"]) for row in groups}),
                "latest_forecast_origin": max(
                    (str(row["latest_forecast_origin"]) for row in groups), default=None
                ),
                "forecast_row_counts": groups,
                "model_versions": sorted(
                    {
                        str(version)
                        for row in groups
                        for version in (row.get("model_versions") or [])
                    }
                ),
                "date_range": {
                    "start": min((str(row["start_time"]) for row in groups), default=None),
                    "end": max((str(row["end_time"]) for row in groups), default=None),
                },
                "total_rows": int(frame["row_count"].sum()),
            }

        value, hit = self.cache.get_or_create("summary", build)
        if self.metrics is not None:
            self.metrics.cache.labels("hit" if hit else "miss").inc()
        return cast(dict[str, Any], value)
