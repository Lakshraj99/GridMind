"""EIA ingestion pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from gridmind.config import Settings
from gridmind.data.eia_client import EIAClient
from gridmind.data.processing import (
    build_ingestion_reconciliation,
    generate_quality_report,
    prepare_eia_records,
)
from gridmind.data.schemas import validate_processed_data
from gridmind.data.storage import (
    DuckDBStorage,
    write_json_report,
    write_processed_parquet,
    write_quarantine_parquet,
    write_raw_response,
)
from gridmind.exceptions import ConfigurationError, DataValidationError, MissingDemandError
from gridmind.time_utils import format_utc_timestamp


@dataclass(frozen=True)
class IngestionResult:
    """Paths and counts produced by a successful ingestion."""

    rows: int
    raw_path: Path
    processed_path: Path
    quality_report_path: Path
    duckdb_rows: int
    quarantined_rows: int = 0
    quarantine_path: Path | None = None


def run_ingestion(
    settings: Settings,
    *,
    region: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    missing_demand_policy: Literal["error", "drop"] | str | None = None,
    client: EIAClient | None = None,
) -> IngestionResult:
    """Fetch, validate, report, and persist one region's EIA data."""
    selected_region = region or settings.grid_region
    selected_start = start_date or settings.data_start_date
    selected_end = end_date or settings.data_end_date
    selected_policy = missing_demand_policy or settings.missing_demand_policy
    if selected_policy not in {"error", "drop"}:
        raise ConfigurationError("MISSING_DEMAND_POLICY must be either 'error' or 'drop'.")
    if not selected_start or not selected_end:
        raise ConfigurationError(
            "DATA_START_DATE and DATA_END_DATE (or CLI options) are required for ingestion."
        )

    owns_client = client is None
    active_client = client or EIAClient(
        api_key=settings.require_eia_api_key(), base_url=settings.eia_base_url
    )
    try:
        fetched = active_client.fetch_hourly_data(selected_region, selected_start, selected_end)
    finally:
        if owns_client:
            active_client.close()

    raw_path = write_raw_response(
        fetched.pages,
        settings.data_dir / "raw",
        secrets=active_client.redaction_secrets,
    )
    candidate = prepare_eia_records(fetched.records)
    missing_rows = candidate.loc[candidate["demand_mw"].isna()].copy()
    missing_timestamps = sorted(
        format_utc_timestamp(timestamp)
        for timestamp in missing_rows["timestamp_utc"].drop_duplicates()
    )
    report_frame = candidate
    retained_rows = 0 if selected_policy == "error" and not missing_rows.empty else len(candidate)
    quarantine_path: Path | None = None

    if not missing_rows.empty and selected_policy == "drop":
        quarantine_path = write_quarantine_parquet(missing_rows, settings.data_dir / "quarantine")
        report_frame = candidate.loc[candidate["demand_mw"].notna()].reset_index(drop=True)
        retained_rows = len(report_frame)

    quality_report = generate_quality_report(report_frame)
    reconciliation_retained = report_frame
    if selected_policy == "error" and not missing_rows.empty:
        reconciliation_retained = candidate.loc[candidate["demand_mw"].notna()].reset_index(
            drop=True
        )
    reconciliation = build_ingestion_reconciliation(
        fetched.records,
        candidate,
        reconciliation_retained,
        start_date=selected_start,
        end_date=selected_end,
    )
    quality_report.update(
        {
            "missing_demand_count_before_filtering": len(missing_rows),
            "quarantined_row_count": len(missing_rows) if selected_policy == "drop" else 0,
            "rows_retained": retained_rows,
            "missing_demand_timestamps": missing_timestamps,
            "missing_demand_policy": selected_policy,
            "resulting_hourly_sequence_contains_gaps": quality_report["timestamp_gap_count"] > 0,
            "reconciliation": reconciliation,
        }
    )
    report_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    report_path = write_json_report(
        quality_report,
        settings.data_quality_dir / f"data_quality_report_{report_stamp}.json",
    )
    if reconciliation["unexplained_difference"] != 0:
        raise DataValidationError(
            "Ingestion row reconciliation failed with an unexplained difference of "
            f"{reconciliation['unexplained_difference']}. Quality report: {report_path}."
        )
    if not missing_rows.empty and selected_policy == "error":
        raise MissingDemandError(
            f"Found {len(missing_rows)} hourly observations with missing actual demand. "
            f"Quality report: {report_path}. Rerun with --missing-demand-policy drop "
            "to quarantine and exclude them."
        )

    try:
        frame = validate_processed_data(report_frame)
    except DataValidationError:
        if quarantine_path is not None:
            quarantine_path.unlink(missing_ok=True)
        raise
    processed_path = write_processed_parquet(frame, settings.data_dir / "processed")
    row_count = DuckDBStorage(settings.duckdb_path).upsert(frame)
    return IngestionResult(
        rows=len(frame),
        raw_path=raw_path,
        processed_path=processed_path,
        quality_report_path=report_path,
        duckdb_rows=row_count,
        quarantined_rows=len(missing_rows) if selected_policy == "drop" else 0,
        quarantine_path=quarantine_path,
    )
