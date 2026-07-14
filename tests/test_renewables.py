"""Renewable mapping, quarantine, net load, and persistence tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gridmind.exceptions import RenewableDataError, TargetForecastError
from gridmind.renewables.processing import process_renewable_records
from gridmind.renewables.storage import RenewableStorage, write_renewable_parquet
from gridmind.renewables.targets import (
    component_net_load,
    compute_net_load,
    get_target_definition,
)


def _records() -> list[dict[str, object]]:
    return [
        {
            "period": "2024-01-01T00",
            "respondent": "PJM",
            "fueltype": "SUN",
            "type-name": "Solar",
            "value": 10,
        },
        {
            "period": "2024-01-01T00",
            "respondent": "PJM",
            "fueltype": "WND",
            "type-name": "Wind",
            "value": 20,
        },
        {
            "period": "2024-01-01T01",
            "respondent": "PJM",
            "type": "unknown",
            "type-name": "Solar generation",
            "value": 11,
        },
        {"period": "2024-01-01T02", "respondent": "PJM", "type": "WND", "value": -1},
    ]


def test_renewable_mapping_missing_values_and_quarantine() -> None:
    result = process_renewable_records(
        _records(), ingestion_timestamp=pd.Timestamp("2024-02-01", tz="UTC")
    )
    assert len(result.valid) == 2
    assert len(result.quarantine) == 1
    complete = result.valid.iloc[0]
    assert complete["solar_generation_mw"] == 10
    assert complete["wind_generation_mw"] == 20
    assert complete["total_renewable_generation_mw"] == 30
    missing = result.valid.iloc[1]
    assert missing["solar_generation_mw"] == 11
    assert pd.isna(missing["wind_generation_mw"])
    assert pd.isna(missing["total_renewable_generation_mw"])
    assert result.report["missing_total_rows"] == 1


def test_conflicting_renewable_duplicates_fail() -> None:
    records = _records()[:2]
    records.append({**records[0], "value": 99})
    with pytest.raises(RenewableDataError, match="Conflicting"):
        process_renewable_records(records)


def test_renewable_duckdb_and_parquet_are_idempotent(tmp_path: Path) -> None:
    frame = process_renewable_records(_records()).valid
    storage = RenewableStorage(tmp_path / "grid.duckdb")
    assert storage.upsert(frame) == 2
    assert storage.upsert(frame) == 2
    assert len(storage.read("PJM")) == 2
    path = write_renewable_parquet(frame, tmp_path / "renewables")
    write_renewable_parquet(frame, tmp_path / "renewables")
    assert len(pd.read_parquet(path)) == 2


def test_net_load_is_complete_case_and_can_be_negative() -> None:
    timestamps = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    demand = pd.DataFrame(
        {"region": "PJM", "timestamp_utc": timestamps, "demand_mw": [100.0, 10.0, 50.0]}
    )
    renewable = pd.DataFrame(
        {
            "region": "PJM",
            "timestamp_utc": timestamps[:2],
            "total_renewable_generation_mw": [20.0, 30.0],
        }
    )
    net, report = compute_net_load(demand, renewable)
    assert net["net_load_mw"].tolist() == [80.0, -20.0]
    assert report["net_load_available_rows"] == 2
    assert report["missing_overlap_rows"] == 1
    index = pd.RangeIndex(2)
    assert component_net_load(
        pd.Series([100.0, 90.0], index=index),
        pd.Series([10.0, 20.0], index=index),
        pd.Series([30.0, 40.0], index=index),
    ).tolist() == [60.0, 30.0]
    with pytest.raises(TargetForecastError, match="identical"):
        component_net_load(pd.Series([1.0]), pd.Series([1.0], index=[2]), pd.Series([1.0]))
    assert get_target_definition("net_load_mw").nonnegative is False
    assert (
        get_target_definition("total_renewable_generation_mw").registry_setting
        == "total_renewable_model_name"
    )
    with pytest.raises(TargetForecastError, match="Unsupported"):
        get_target_definition("coal")
