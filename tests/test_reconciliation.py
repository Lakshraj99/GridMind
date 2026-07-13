"""Source-to-retained ingestion row reconciliation tests."""

from __future__ import annotations

from typing import Any

from gridmind.data.processing import (
    build_ingestion_reconciliation,
    prepare_eia_records,
)


def test_reconciliation_accounts_for_exact_source_duplicates(
    eia_payload: dict[str, Any],
) -> None:
    records = list(eia_payload["response"]["data"])
    records.append(records[0].copy())
    pivoted = prepare_eia_records(records)
    report = build_ingestion_reconciliation(
        records,
        pivoted,
        pivoted,
        start_date="2024-01-01T00:00:00Z",
        end_date="2024-01-01T01:00:00Z",
    )
    assert report["raw_measurement_rows"] == 9
    assert report["unique_source_records"] == 8
    assert report["exact_duplicates_removed"] == 1
    assert report["materialized_pivoted_timestamp_rows"] == 2
    assert report["pivoted_timestamp_rows"] == 3
    assert report["unexplained_difference"] == 0


def test_reconciliation_identifies_a_fully_absent_source_hour(
    eia_payload: dict[str, Any],
) -> None:
    original = eia_payload["response"]["data"]
    first = [record.copy() for record in original if record["period"].endswith("T00")]
    third = [
        {**record, "period": "2024-01-01T02"}
        for record in original
        if record["period"].endswith("T01")
    ]
    records = [*first, *third]
    pivoted = prepare_eia_records(records)
    report = build_ingestion_reconciliation(
        records,
        pivoted,
        pivoted,
        start_date="2024-01-01T00:00:00Z",
        end_date="2024-01-01T02:00:00Z",
    )
    assert report["expected_hourly_timestamps"] == 3
    assert report["materialized_pivoted_timestamp_rows"] == 2
    assert report["missing_source_timestamp_rows"] == 1
    assert report["missing_source_timestamps"] == ["2024-01-01T01:00:00Z"]
    assert report["unexplained_difference"] == 0


def test_reconciliation_exposes_conflicts_and_unexplained_row_loss(
    eia_payload: dict[str, Any],
) -> None:
    records = list(eia_payload["response"]["data"])
    pivoted = prepare_eia_records(records)
    conflict = {**records[0], "value": "9999"}
    report = build_ingestion_reconciliation(
        [*records, conflict],
        pivoted,
        pivoted.iloc[:-1],
        start_date="2024-01-01T00:00:00Z",
        end_date="2024-01-01T01:00:00Z",
    )
    assert report["conflicting_duplicates"] == 1
    assert report["unexplained_difference"] == 1
