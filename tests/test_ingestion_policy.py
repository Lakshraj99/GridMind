"""Missing-demand ingestion policy and quarantine integration tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest

import gridmind.pipelines.ingest as ingest_module
from gridmind.config import Settings
from gridmind.data.eia_client import EIAClient
from gridmind.data.schemas import validate_processed_data
from gridmind.data.storage import DuckDBStorage, read_processed_parquet
from gridmind.exceptions import DataValidationError, MissingDemandError
from gridmind.pipelines.ingest import run_ingestion


def _payload_with_missing_middle_demand(secret: str) -> dict[str, Any]:
    records: list[dict[str, str]] = []
    measurements = (
        ("D", "Demand", 1000),
        ("DF", "Demand Forecast", 1010),
        ("NG", "Net Generation", 980),
        ("TI", "Total Interchange", 20),
    )
    for hour in range(3):
        for type_code, type_name, base_value in measurements:
            if hour == 1 and type_code == "D":
                continue
            records.append(
                {
                    "period": f"2024-01-01T0{hour}",
                    "respondent": "PJM",
                    "type": type_code,
                    "type-name": type_name,
                    "value": str(base_value + hour * 20),
                }
            )
    return {
        "response": {"total": str(len(records)), "data": records},
        "request": {
            "command": "/v2/electricity/rto/region-data/data/",
            "params": {"api_key": secret},
        },
    }


def _client(payload: dict[str, Any], secret: str) -> EIAClient:
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    return EIAClient(secret, client=httpx.Client(transport=transport))


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        DATA_DIR=tmp_path / "data",
        DATA_QUALITY_DIR=tmp_path / "artifacts" / "data_quality",
        DUCKDB_PATH=tmp_path / "grid.duckdb",
        MLFLOW_ENABLED=False,
        _env_file=None,
    )


def test_default_error_policy_reports_before_failing_without_processed_data(
    tmp_path: Path,
) -> None:
    secret = "top-secret-eia-key"
    payload = _payload_with_missing_middle_demand(secret)
    settings = _settings(tmp_path)

    with pytest.raises(MissingDemandError, match="--missing-demand-policy drop") as raised:
        run_ingestion(
            settings,
            region="PJM",
            start_date="2024-01-01",
            end_date="2024-01-02",
            client=_client(payload, secret),
        )

    reports = list(settings.data_quality_dir.glob("data_quality_report_*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["missing_demand_policy"] == "error"
    assert report["missing_demand_count_before_filtering"] == 1
    assert report["quarantined_row_count"] == 0
    assert report["rows_retained"] == 0
    assert report["missing_demand_timestamps"] == ["2024-01-01T01:00:00Z"]
    reconciliation = report["reconciliation"]
    assert reconciliation["unexplained_difference"] == 0
    assert reconciliation["pivoted_timestamp_rows"] == 3
    assert reconciliation["missing_demand_rows"] == 1
    assert reconciliation["retained_rows"] == 2
    assert not list((settings.data_dir / "processed").glob("**/*.parquet"))
    assert not settings.duckdb_path.exists()
    assert secret not in str(raised.value)
    raw_text = next((settings.data_dir / "raw").glob("*.json")).read_text(encoding="utf-8")
    assert secret not in raw_text
    assert "[REDACTED]" in raw_text


def test_drop_policy_quarantines_only_missing_rows_and_preserves_gap(
    tmp_path: Path,
) -> None:
    secret = "drop-policy-secret"
    payload = _payload_with_missing_middle_demand(secret)
    settings = _settings(tmp_path)
    result = run_ingestion(
        settings,
        region="PJM",
        start_date="2024-01-01",
        end_date="2024-01-02",
        missing_demand_policy="drop",
        client=_client(payload, secret),
    )

    assert result.rows == 2
    assert result.quarantined_rows == 1
    assert result.quarantine_path is not None and result.quarantine_path.exists()
    quarantine = pd.read_parquet(result.quarantine_path)
    assert quarantine["timestamp_utc"].dt.tz is not None
    assert quarantine.loc[0, "timestamp_utc"].isoformat() == "2024-01-01T01:00:00+00:00"
    assert quarantine.loc[0, "region"] == "PJM"
    assert pd.isna(quarantine.loc[0, "demand_mw"])
    assert quarantine.loc[0, "forecast_demand_mw"] == 1030.0
    assert quarantine.loc[0, "net_generation_mw"] == 1000.0
    assert quarantine.loc[0, "total_interchange_mw"] == 40.0
    assert pd.notna(quarantine.loc[0, "ingestion_timestamp_utc"])

    retained = read_processed_parquet(result.processed_path)
    validate_processed_data(retained)
    assert retained["demand_mw"].tolist() == [1000.0, 1040.0]
    assert retained["timestamp_utc"].dt.hour.tolist() == [0, 2]
    assert retained["demand_mw"].notna().all()

    report = json.loads(result.quality_report_path.read_text(encoding="utf-8"))
    assert result.quality_report_path.parent == settings.data_quality_dir
    assert settings.data_dir / "processed" not in result.quality_report_path.parents
    assert not list((settings.data_dir / "processed").rglob("*.json"))
    assert report["missing_demand_policy"] == "drop"
    assert report["missing_demand_count_before_filtering"] == 1
    assert report["quarantined_row_count"] == 1
    assert report["rows_retained"] == 2
    assert report["missing_demand_timestamps"] == ["2024-01-01T01:00:00Z"]
    assert report["timestamp_gap_count"] == 1
    assert report["resulting_hourly_sequence_contains_gaps"] is True
    reconciliation = report["reconciliation"]
    assert reconciliation["unexplained_difference"] == 0
    assert (
        reconciliation["pivoted_timestamp_rows"]
        - reconciliation["exact_duplicates_removed"]
        - reconciliation["missing_demand_rows"]
        - reconciliation["other_invalid_rows"]
        == reconciliation["retained_rows"]
    )

    stored = DuckDBStorage(settings.duckdb_path).read_region("PJM", "2024-01-01", "2024-01-02")
    assert len(stored) == 2
    assert stored["demand_mw"].notna().all()
    assert DuckDBStorage(settings.duckdb_path).inspect()["missing_demand_count"] == 0


def test_nonzero_reconciliation_difference_fails_before_persistence(
    tmp_path: Path,
    eia_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    original = ingest_module.build_ingestion_reconciliation

    def unexplained(*args: Any, **kwargs: Any) -> dict[str, Any]:
        report = original(*args, **kwargs)
        report["unexplained_difference"] = 1
        return report

    monkeypatch.setattr(ingest_module, "build_ingestion_reconciliation", unexplained)
    with pytest.raises(DataValidationError, match="unexplained difference of 1"):
        run_ingestion(
            settings,
            region="PJM",
            start_date="2024-01-01",
            end_date="2024-01-01T01:00:00Z",
            client=_client(eia_payload, "test"),
        )
    assert not list((settings.data_dir / "processed").rglob("*.parquet"))
    assert not settings.duckdb_path.exists()
