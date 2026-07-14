"""Map EIA fuel codes into explicit solar and wind hourly targets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from gridmind.renewables.schemas import RENEWABLE_COLUMNS, validate_renewable_data

FUEL_TARGETS = {
    "SUN": "solar_generation_mw",
    "SOLAR": "solar_generation_mw",
    "SOLAR GENERATION": "solar_generation_mw",
    "WND": "wind_generation_mw",
    "WIND": "wind_generation_mw",
    "WIND GENERATION": "wind_generation_mw",
}


@dataclass(frozen=True)
class RenewableProcessingResult:
    valid: pd.DataFrame
    quarantine: pd.DataFrame
    report: dict[str, Any]


def _fuel_identifier(record: dict[str, Any]) -> str:
    # ``fueltype`` is the EIA fuel-type route's code field. Retain ``type`` as
    # a compatibility fallback for previously persisted fixtures/pages.
    code = str(record.get("fueltype", record.get("type", ""))).strip().upper()
    name = str(record.get("type-name", "")).strip().upper()
    return code if code in FUEL_TARGETS else name


def process_renewable_records(
    records: list[dict[str, Any]], *, ingestion_timestamp: pd.Timestamp | None = None
) -> RenewableProcessingResult:
    """Pivot solar/wind without treating absent measurements as zero."""
    rows = [
        {
            "timestamp_utc": pd.to_datetime(
                str(record.get("period", "")), utc=True, errors="coerce"
            ),
            "region": record.get("respondent"),
            "target": FUEL_TARGETS.get(_fuel_identifier(record)),
            "value": pd.to_numeric(str(record.get("value", "nan")), errors="coerce"),
        }
        for record in records
    ]
    raw = pd.DataFrame(rows).dropna(subset=["timestamp_utc", "region", "target"])
    conflicts = raw.groupby(["region", "timestamp_utc", "target"])["value"].nunique(dropna=False)
    conflicting = conflicts[conflicts > 1]
    if not conflicting.empty:
        from gridmind.exceptions import RenewableDataError

        raise RenewableDataError("Conflicting duplicate renewable measurements were found.")
    raw = raw.drop_duplicates(["region", "timestamp_utc", "target"], keep="last")
    pivoted = raw.pivot(
        index=["timestamp_utc", "region"], columns="target", values="value"
    ).reset_index()
    pivoted.columns.name = None
    for column in ("solar_generation_mw", "wind_generation_mw"):
        if column not in pivoted:
            pivoted[column] = float("nan")
        pivoted[column] = pivoted[column].astype("float64")
    invalid = (pivoted["solar_generation_mw"].notna() & (pivoted["solar_generation_mw"] < 0)) | (
        pivoted["wind_generation_mw"].notna() & (pivoted["wind_generation_mw"] < 0)
    )
    quarantine = pivoted.loc[invalid].copy()
    valid = pivoted.loc[~invalid].copy()
    valid["total_renewable_generation_mw"] = valid[
        ["solar_generation_mw", "wind_generation_mw"]
    ].sum(axis=1, min_count=2)
    stamp = ingestion_timestamp or pd.Timestamp(datetime.now(UTC))
    valid["ingestion_timestamp_utc"] = stamp
    valid = validate_renewable_data(valid[RENEWABLE_COLUMNS])
    gap_count = 0
    for _region, group in valid.groupby("region", observed=True):
        differences = group.sort_values("timestamp_utc")["timestamp_utc"].diff().dropna()
        gap_count += sum(max(int(value / pd.Timedelta(hours=1)) - 1, 0) for value in differences)
    report = {
        "raw_measurement_rows": len(records),
        "hourly_rows": len(pivoted),
        "quarantined_rows": len(quarantine),
        "retained_rows": len(valid),
        "missing_solar_rows": int(valid["solar_generation_mw"].isna().sum()),
        "missing_wind_rows": int(valid["wind_generation_mw"].isna().sum()),
        "missing_total_rows": int(valid["total_renewable_generation_mw"].isna().sum()),
        "renewable_target_gap_count": int(gap_count),
    }
    return RenewableProcessingResult(valid, quarantine, report)
