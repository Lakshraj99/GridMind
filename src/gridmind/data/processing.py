"""Normalize EIA records into GridMind's canonical data contract."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from gridmind.data.schemas import validate_processed_data, validate_raw_data
from gridmind.exceptions import ConflictingDuplicateError, DataValidationError, MissingDemandError
from gridmind.time_utils import format_utc_timestamp, inclusive_hourly_range

MissingDemandPolicy = Literal["error", "drop"]

CANONICAL_COLUMNS = [
    "timestamp_utc",
    "region",
    "demand_mw",
    "forecast_demand_mw",
    "net_generation_mw",
    "total_interchange_mw",
    "ingestion_timestamp_utc",
]

MEASUREMENT_COLUMNS = {
    "D": "demand_mw",
    "DEMAND": "demand_mw",
    "DF": "forecast_demand_mw",
    "DEMAND FORECAST": "forecast_demand_mw",
    "NG": "net_generation_mw",
    "NET GENERATION": "net_generation_mw",
    "TI": "total_interchange_mw",
    "TOTAL INTERCHANGE": "total_interchange_mw",
}


def _measurement_identifier(item: dict[str, Any]) -> Any:
    """Prefer a recognized EIA type code, falling back to its documented name."""
    type_code = item.get("type")
    type_name = item.get("type-name")
    normalized_code = str(type_code).strip().upper() if type_code is not None else ""
    if normalized_code in MEASUREMENT_COLUMNS:
        return normalized_code
    normalized_name = str(type_name).strip().upper() if type_name is not None else ""
    return normalized_name or type_code


def normalize_eia_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Select source fields from EIA records into a stable raw representation."""
    rows = [
        {
            "timestamp": item.get("period"),
            "region": item.get("respondent"),
            "measurement_type": _measurement_identifier(item),
            "value": item.get("value"),
        }
        for item in records
    ]
    frame = pd.DataFrame(rows, columns=["timestamp", "region", "measurement_type", "value"])
    return validate_raw_data(frame)


def process_eia_records(
    records: list[dict[str, Any]],
    *,
    ingestion_timestamp: pd.Timestamp | None = None,
    missing_demand_policy: MissingDemandPolicy = "error",
) -> pd.DataFrame:
    """Parse EIA measurements and apply an explicit missing-demand policy."""
    frame = prepare_eia_records(records, ingestion_timestamp=ingestion_timestamp)
    missing_count = int(frame["demand_mw"].isna().sum())
    if missing_count and missing_demand_policy == "error":
        raise MissingDemandError(
            f"Found {missing_count} hourly observations with missing actual demand. "
            "Rerun ingestion with --missing-demand-policy drop to quarantine and exclude them."
        )
    if missing_demand_policy == "drop":
        frame = frame.loc[frame["demand_mw"].notna()].reset_index(drop=True)
    elif missing_demand_policy != "error":
        raise ValueError("missing_demand_policy must be 'error' or 'drop'.")
    return validate_processed_data(frame)


def prepare_eia_records(
    records: list[dict[str, Any]], *, ingestion_timestamp: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Parse, deduplicate, and pivot EIA measurements before target policy enforcement."""
    raw = normalize_eia_records(records).drop_duplicates().copy()
    try:
        raw["timestamp_utc"] = pd.to_datetime(raw["timestamp"], utc=True, errors="raise")
    except (ValueError, TypeError) as exc:
        raise DataValidationError(f"Invalid value in timestamp column: {exc}") from exc
    raw["region"] = raw["region"].astype("string")
    raw["measurement_type"] = raw["measurement_type"].astype("string").str.upper()
    raw["canonical_measurement"] = raw["measurement_type"].map(MEASUREMENT_COLUMNS)
    raw = raw.loc[raw["canonical_measurement"].notna()].copy()
    try:
        raw["numeric_value"] = pd.to_numeric(raw["value"], errors="raise").astype("float64")
    except (ValueError, TypeError) as exc:
        raise DataValidationError(f"Invalid numeric value in EIA 'value' column: {exc}") from exc

    key = ["region", "timestamp_utc", "canonical_measurement"]
    conflicts = raw.groupby(key, dropna=False)["numeric_value"].nunique(dropna=False)
    conflicting = conflicts[conflicts > 1]
    if not conflicting.empty:
        details = [tuple(str(part) for part in index) for index in conflicting.index[:5]]
        raise ConflictingDuplicateError(
            f"Conflicting duplicate measurements for region/timestamp/type: {details}"
        )
    raw = raw.drop_duplicates(key, keep="last")
    pivoted = raw.pivot(
        index=["timestamp_utc", "region"],
        columns="canonical_measurement",
        values="numeric_value",
    ).reset_index()
    pivoted.columns.name = None
    for column in CANONICAL_COLUMNS[2:6]:
        if column not in pivoted:
            pivoted[column] = float("nan")
        pivoted[column] = pivoted[column].astype("float64")
    timestamp = ingestion_timestamp or pd.Timestamp.now(tz="UTC")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    pivoted["ingestion_timestamp_utc"] = timestamp
    return pivoted[CANONICAL_COLUMNS].sort_values(["region", "timestamp_utc"], ignore_index=True)


def generate_quality_report(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize completeness, uniqueness, signs, and hourly continuity."""
    row_count = len(frame)
    duplicate_count = int(frame.duplicated(["region", "timestamp_utc"]).sum())
    missing_counts = {column: int(value) for column, value in frame.isna().sum().items()}
    missing_percent = {
        column: (float(value) / row_count * 100.0 if row_count else 0.0)
        for column, value in missing_counts.items()
    }
    numeric_measurements = [
        "demand_mw",
        "forecast_demand_mw",
        "net_generation_mw",
        "total_interchange_mw",
    ]
    negative_count = int((frame[numeric_measurements] < 0).sum().sum()) if row_count else 0
    timestamp_gap_count = 0
    if row_count:
        ordered = frame.sort_values(["region", "timestamp_utc"])
        differences = ordered.groupby("region")["timestamp_utc"].diff().dropna()
        timestamp_gap_count = int(
            sum(max(int(delta / pd.Timedelta(hours=1)) - 1, 0) for delta in differences)
        )
    return {
        "row_count": row_count,
        "date_range": {
            "start": format_utc_timestamp(frame["timestamp_utc"].min()) if row_count else None,
            "end": format_utc_timestamp(frame["timestamp_utc"].max()) if row_count else None,
        },
        "region_count": int(frame["region"].nunique()) if row_count else 0,
        "regions": sorted(str(value) for value in frame["region"].dropna().unique()),
        "duplicate_count": duplicate_count,
        "missing_value_count_by_column": missing_counts,
        "missing_percentage_by_column": missing_percent,
        "negative_value_count": negative_count,
        "timestamp_gap_count": timestamp_gap_count,
        "expected_hourly_frequency": timestamp_gap_count == 0 and duplicate_count == 0,
    }


def build_ingestion_reconciliation(
    records: list[dict[str, Any]],
    pivoted: pd.DataFrame,
    retained: pd.DataFrame,
    *,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    """Reconcile raw measurement records through retained canonical hourly rows."""
    raw = normalize_eia_records(records)
    unique = raw.drop_duplicates().copy()
    exact_duplicates = len(raw) - len(unique)
    unique["measurement_type"] = unique["measurement_type"].astype("string").str.upper()
    unique["canonical_measurement"] = unique["measurement_type"].map(MEASUREMENT_COLUMNS)
    mapped = unique.loc[unique["canonical_measurement"].notna()].copy()
    mapped["numeric_value"] = pd.to_numeric(mapped["value"], errors="coerce")
    conflict_key = ["region", "timestamp", "canonical_measurement"]
    conflict_counts = mapped.groupby(conflict_key, dropna=False)["numeric_value"].nunique(
        dropna=False
    )
    conflicting_duplicates = int((conflict_counts > 1).sum())

    expected = inclusive_hourly_range(start_date, end_date)
    observed = pd.DatetimeIndex(pivoted["timestamp_utc"].drop_duplicates())
    missing_source_timestamps = expected.difference(observed)
    unexpected_timestamps = observed.difference(expected)
    missing_demand_rows = int(pivoted["demand_mw"].isna().sum())
    demand = pd.to_numeric(pivoted["demand_mw"], errors="coerce")
    other_invalid_mask = demand.notna() & (~np.isfinite(demand) | (demand < 0))
    other_invalid_rows = int(other_invalid_mask.sum())

    # The reconciliation population includes exact source duplicates before
    # their explicit deduction. ``materialized_pivoted_timestamp_rows`` is the
    # actual number of unique timestamp rows produced by the pivot.
    reconciliation_pivoted_rows = len(pivoted) + exact_duplicates
    unexplained = (
        reconciliation_pivoted_rows
        - exact_duplicates
        - missing_demand_rows
        - other_invalid_rows
        - len(retained)
    )
    return {
        "raw_measurement_rows": len(raw),
        "unique_source_records": len(unique),
        "expected_hourly_timestamps": len(expected),
        "pivoted_timestamp_rows": reconciliation_pivoted_rows,
        "materialized_pivoted_timestamp_rows": len(pivoted),
        "exact_duplicates_removed": exact_duplicates,
        "conflicting_duplicates": conflicting_duplicates,
        "missing_demand_rows": missing_demand_rows,
        "other_invalid_rows": other_invalid_rows,
        "retained_rows": len(retained),
        "unexplained_difference": unexplained,
        "missing_source_timestamp_rows": len(missing_source_timestamps),
        "missing_source_timestamps": [
            format_utc_timestamp(value) for value in missing_source_timestamps
        ],
        "unexpected_timestamp_rows": len(unexpected_timestamps),
        "unexpected_timestamps": [format_utc_timestamp(value) for value in unexpected_timestamps],
    }
