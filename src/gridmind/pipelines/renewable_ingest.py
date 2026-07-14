"""EIA renewable generation ingestion and complete-case net-load view."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd

from gridmind.config import Settings
from gridmind.data.eia_client import EIAClient
from gridmind.data.storage import DuckDBStorage, write_json_report, write_raw_response
from gridmind.renewables.processing import process_renewable_records
from gridmind.renewables.storage import (
    RenewableStorage,
    create_net_load_view,
    write_renewable_parquet,
)
from gridmind.renewables.targets import compute_net_load


@dataclass(frozen=True)
class RenewableIngestionResult:
    rows: int
    quarantined_rows: int
    processed_path: Path
    quarantine_path: Path | None
    report_path: Path
    duckdb_rows: int


def run_renewable_ingestion(
    settings: Settings,
    *,
    region: str,
    start_date: str,
    end_date: str,
    client: EIAClient | None = None,
) -> RenewableIngestionResult:
    active = client or EIAClient(
        settings.require_eia_api_key(),
        settings.eia_base_url,
        timeout=settings.weather_request_timeout_seconds,
    )
    owns = client is None
    try:
        fetched = active.fetch_renewable_data(region, start_date, end_date)
    finally:
        if owns:
            active.close()
    write_raw_response(
        fetched.pages,
        settings.data_dir / "raw" / "renewables",
        secrets=active.redaction_secrets,
    )
    result = process_renewable_records(fetched.records)
    processed = write_renewable_parquet(
        result.valid, settings.data_dir / "renewables" / "processed"
    )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    quarantine_path: Path | None = None
    if not result.quarantine.empty:
        directory = settings.data_dir / "quarantine"
        directory.mkdir(parents=True, exist_ok=True)
        quarantine_path = directory / f"invalid_renewable_{stamp}.parquet"
        result.quarantine.to_parquet(quarantine_path, index=False)
    count = RenewableStorage(settings.duckdb_path).upsert(result.valid)
    create_net_load_view(settings.duckdb_path)
    start_utc = pd.Timestamp(start_date, tz="UTC").to_pydatetime()
    end_utc = (
        pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    ).to_pydatetime()
    try:
        demand = DuckDBStorage(settings.duckdb_path).read_data(
            regions=[region], start_date=start_utc, end_date=end_utc
        )
    except (duckdb.Error, OSError):
        demand = pd.DataFrame(columns=["region", "timestamp_utc", "demand_mw"])
    _net_load, overlap_report = compute_net_load(demand, result.valid)
    quality_report = {
        **result.report,
        "region": region,
        "start_date": start_date,
        "end_date": end_date,
        "demand_renewable_overlap": overlap_report,
        "quarantine_artifact": str(quarantine_path) if quarantine_path else None,
    }
    report_path = write_json_report(
        quality_report,
        settings.data_quality_dir / f"renewable_coverage_{stamp}.json",
    )
    return RenewableIngestionResult(
        len(result.valid), len(result.quarantine), processed, quarantine_path, report_path, count
    )
