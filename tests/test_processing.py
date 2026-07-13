"""Tests for canonical processing and data-quality reporting."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from gridmind.data.processing import generate_quality_report, process_eia_records
from gridmind.exceptions import ConflictingDuplicateError, DataValidationError


def test_processes_fixture_to_canonical_frame(eia_payload: dict[str, Any]) -> None:
    records = eia_payload["response"]["data"]
    records.append(records[0].copy())
    result = process_eia_records(records, ingestion_timestamp=pd.Timestamp("2024-02-01", tz="UTC"))
    assert list(result.columns) == [
        "timestamp_utc",
        "region",
        "demand_mw",
        "forecast_demand_mw",
        "net_generation_mw",
        "total_interchange_mw",
        "ingestion_timestamp_utc",
    ]
    assert len(result) == 2
    assert str(result["timestamp_utc"].dtype) == "datetime64[ns, UTC]"
    assert result.loc[0, "demand_mw"] == 1000.0
    assert result.loc[0, "forecast_demand_mw"] == 1010.0
    assert result.loc[0, "net_generation_mw"] == 980.0
    assert result.loc[0, "total_interchange_mw"] == 20.0


def test_measurement_mapping_falls_back_to_documented_type_names() -> None:
    records = [
        {
            "period": "2024-01-01T00",
            "respondent": "PJM",
            "type": f"unknown-{index}",
            "type-name": type_name,
            "value": value,
        }
        for index, (type_name, value) in enumerate(
            (
                ("Demand", "1000"),
                ("Demand Forecast", "1010"),
                ("Net Generation", "980"),
                ("Total Interchange", "20"),
            )
        )
    ]
    result = process_eia_records(records)
    assert result.loc[0, "demand_mw"] == 1000.0
    assert result.loc[0, "forecast_demand_mw"] == 1010.0
    assert result.loc[0, "net_generation_mw"] == 980.0
    assert result.loc[0, "total_interchange_mw"] == 20.0


def test_detects_conflicting_duplicates(eia_payload: dict[str, Any]) -> None:
    records = eia_payload["response"]["data"]
    conflict = records[0].copy()
    conflict["value"] = "9999"
    with pytest.raises(ConflictingDuplicateError, match="Conflicting duplicate"):
        process_eia_records([*records, conflict])


def test_rejects_bad_timestamp_and_numeric_value(eia_payload: dict[str, Any]) -> None:
    record = eia_payload["response"]["data"][0].copy()
    record["period"] = "not-a-date"
    with pytest.raises(DataValidationError, match="timestamp"):
        process_eia_records([record])
    record["period"] = "2024-01-01T00"
    record["value"] = "not-a-number"
    with pytest.raises(DataValidationError, match="numeric"):
        process_eia_records([record])


def test_quality_report_counts_gaps_and_missing(hourly_frame: pd.DataFrame) -> None:
    frame = hourly_frame.iloc[:5].drop(index=2).reset_index(drop=True)
    report = generate_quality_report(frame)
    assert report["row_count"] == 4
    assert report["timestamp_gap_count"] == 1
    assert report["missing_value_count_by_column"]["forecast_demand_mw"] == 4
    assert report["expected_hourly_frequency"] is False
