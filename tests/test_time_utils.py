"""Host-timezone-independent UTC storage, report, and display tests."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd

from gridmind.data.processing import generate_quality_report
from gridmind.data.storage import DuckDBStorage
from gridmind.time_utils import format_utc_timestamp


def test_utc_formatter_normalizes_offsets_and_uses_z() -> None:
    assert format_utc_timestamp("2023-01-01T05:30:00+05:30") == "2023-01-01T00:00:00Z"
    assert format_utc_timestamp(pd.Timestamp("2025-12-31T23:00:00Z")) == ("2025-12-31T23:00:00Z")


def test_quality_report_uses_canonical_utc_z(hourly_frame: pd.DataFrame) -> None:
    report = generate_quality_report(hourly_frame.iloc[:2])
    assert report["date_range"] == {
        "start": "2024-01-01T00:00:00Z",
        "end": "2024-01-01T01:00:00Z",
    }


def test_duckdb_inspection_is_independent_of_host_timezone(
    tmp_path: Path, hourly_frame: pd.DataFrame
) -> None:
    previous = os.environ.get("TZ")
    try:
        os.environ["TZ"] = "Asia/Kolkata"
        time.tzset()
        storage = DuckDBStorage(tmp_path / "grid.duckdb")
        storage.upsert(hourly_frame.iloc[:2])
        summary = storage.inspect()
    finally:
        if previous is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous
        time.tzset()
    assert summary["date_range"] == {
        "start": "2024-01-01T00:00:00Z",
        "end": "2024-01-01T01:00:00Z",
    }
