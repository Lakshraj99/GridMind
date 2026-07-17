"""Battery dispatch run, point, physics, and lineage query service."""

from __future__ import annotations

from typing import Any

from gridmind.exceptions import ResourceNotFoundError
from gridmind.services.common import (
    DuckDBReadService,
    Page,
    decode_json_fields,
    frame_records,
    where_clause,
)

DISCLAIMER = "Simulated decision support only; this does not control a physical battery."


class DispatchService(DuckDBReadService):
    run_table = "battery_dispatch_runs"
    point_table = "battery_dispatch_points"

    def list(self, *, limit: int, offset: int, **filters: object) -> Page:
        self.require_table(self.run_table)
        where, parameters = where_clause(
            [
                ("region =", filters.get("region")),
                ("battery_id =", filters.get("battery_id")),
                ("objective_mode =", filters.get("objective_mode")),
                ("solver_status =", filters.get("solver_status")),
                ("forecast_origin =", filters.get("forecast_origin")),
                ("forecast_origin >=", filters.get("start_time")),
                ("forecast_origin <=", filters.get("end_time")),
            ]
        )
        count = self.query(f"SELECT COUNT(*) AS total FROM {self.run_table}{where}", parameters)
        frame = self.query(
            f"SELECT * FROM {self.run_table}{where} ORDER BY forecast_origin DESC LIMIT ? OFFSET ?",
            [*parameters, limit, offset],
        )
        items = [decode_json_fields(item) for item in frame_records(frame)]
        for item in items:
            item.pop("artifact_path", None)
            item["disclaimer"] = DISCLAIMER
        return Page(items, int(count.iloc[0]["total"]), limit, offset)

    def get(self, run_id: str) -> dict[str, Any]:
        self.require_table(self.run_table)
        records = frame_records(
            self.query(f"SELECT * FROM {self.run_table} WHERE dispatch_run_id = ?", [run_id])
        )
        if not records:
            raise ResourceNotFoundError(f"Dispatch run '{run_id}' was not found.")
        result = decode_json_fields(records[0])
        result.pop("artifact_path", None)
        result["disclaimer"] = DISCLAIMER
        return result

    def points(self, run_id: str, *, limit: int, offset: int) -> Page:
        self.get(run_id)
        count = self.query(
            f"SELECT COUNT(*) AS total FROM {self.point_table} WHERE dispatch_run_id = ?", [run_id]
        )
        frame = self.query(
            f"SELECT * FROM {self.point_table} WHERE dispatch_run_id = ? "
            "ORDER BY timestamp_utc LIMIT ? OFFSET ?",
            [run_id, limit, offset],
        )
        return Page(frame_records(frame), int(count.iloc[0]["total"]), limit, offset)

    def summary(self, run_id: str) -> dict[str, Any]:
        run = self.get(run_id)
        points = (
            self.query(
                "SELECT MIN(soc_start_mwh) AS minimum_soc_mwh, "
                "MAX(soc_end_mwh) AS maximum_soc_mwh, "
                f"SUM(charge_mw + discharge_mw) AS total_throughput_mwh FROM {self.point_table} "
                "WHERE dispatch_run_id = ?",
                [run_id],
            )
            .iloc[0]
            .to_dict()
        )
        configuration = run.get("configuration", {})
        battery = configuration.get("battery", configuration)
        capacity = float(battery.get("capacity_mwh", 0) or 0)
        step_hours = float(configuration.get("step_hours", 1) or 1)
        throughput = float(points.get("total_throughput_mwh", 0) or 0) * step_hours
        points["total_throughput_mwh"] = throughput
        points["equivalent_cycles"] = throughput / (2 * capacity) if capacity else None
        return {
            "dispatch_run_id": run_id,
            "peak_reduction_mw": float(run["peak_before_mw"]) - float(run["peak_after_mw"]),
            "objective": run.get("objective_breakdown", {}),
            "objective_value": run.get("objective_value"),
            "physics": points,
            "constraint_validation_passed": run.get("constraint_validation_passed"),
            "lineage": run.get("lineage", {}),
            "battery_specification": battery,
            "disclaimer": DISCLAIMER,
        }
