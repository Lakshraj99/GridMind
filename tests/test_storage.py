"""Persistence tests using isolated temporary paths."""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from gridmind.data.processing import CANONICAL_COLUMNS
from gridmind.data.storage import (
    DuckDBStorage,
    empty_canonical_dataframe,
    read_processed_parquet,
    write_processed_parquet,
    write_raw_response,
)
from gridmind.exceptions import StorageError


def test_duckdb_upsert_is_idempotent(tmp_path: Path, hourly_frame: pd.DataFrame) -> None:
    storage = DuckDBStorage(tmp_path / "test.duckdb")
    subset = hourly_frame.iloc[:3].copy()
    assert storage.upsert(subset) == 3
    subset.loc[0, "demand_mw"] = 777.0
    assert storage.upsert(subset) == 3
    read = storage.read_region("PJM", "2024-01-01", "2024-01-02")
    assert len(read) == 3
    assert read.loc[0, "demand_mw"] == 777.0
    assert storage.inspect()["regions"] == ["PJM"]


def test_partitioned_parquet_round_trip(tmp_path: Path, hourly_frame: pd.DataFrame) -> None:
    directory = write_processed_parquet(hourly_frame.iloc[:48], tmp_path / "processed")
    update = hourly_frame.iloc[24:25].copy()
    update.loc[24, "demand_mw"] = 555.0
    write_processed_parquet(update, directory)
    result = read_processed_parquet(directory)
    assert len(result) == 48
    assert list(result.columns) == list(hourly_frame.columns)
    assert result.loc[24, "demand_mw"] == 555.0
    assert not result.duplicated(["region", "timestamp_utc"]).any()
    parquet_file = next(directory.rglob("*.parquet"))
    stored_columns = pd.read_parquet(parquet_file).columns
    assert "region" not in stored_columns
    assert "year" not in stored_columns
    assert "month" not in stored_columns


def test_reader_ignores_unrelated_json_and_opens_only_parquet(
    tmp_path: Path,
    hourly_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = write_processed_parquet(hourly_frame.iloc[:6], tmp_path / "processed")
    (directory / "data_quality_report.json").write_text('{"not": "parquet"}', encoding="utf-8")
    original = pd.read_parquet
    opened: list[Path] = []

    def read_spy(path: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        opened.append(Path(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", read_spy)
    result = read_processed_parquet(directory)
    assert len(result) == 6
    assert opened and all(path.suffix == ".parquet" for path in opened)


def test_empty_processed_directory_returns_stable_canonical_frame(tmp_path: Path) -> None:
    directory = tmp_path / "processed"
    directory.mkdir()
    (directory / "unrelated.json").write_text("{}", encoding="utf-8")
    result = read_processed_parquet(directory)
    expected = empty_canonical_dataframe()
    assert list(result.columns) == CANONICAL_COLUMNS
    assert result.empty
    assert result.dtypes.astype(str).to_dict() == expected.dtypes.astype(str).to_dict()


def test_repeated_writes_are_idempotent(tmp_path: Path, hourly_frame: pd.DataFrame) -> None:
    directory = tmp_path / "processed"
    subset = hourly_frame.iloc[:12].copy()
    write_processed_parquet(subset, directory)
    write_processed_parquet(subset, directory)
    result = read_processed_parquet(directory)
    assert len(result) == 12
    assert not result.duplicated(["region", "timestamp_utc"]).any()


def test_failed_write_preserves_existing_valid_dataset(
    tmp_path: Path,
    hourly_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = write_processed_parquet(hourly_frame.iloc[:8], tmp_path / "processed")
    before = read_processed_parquet(directory)

    def fail_write(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("simulated disk failure")

    with monkeypatch.context() as context:
        context.setattr(pd.DataFrame, "to_parquet", fail_write)
        with pytest.raises(StorageError, match="previously valid data was left unchanged"):
            write_processed_parquet(hourly_frame.iloc[8:10], directory)

    after = read_processed_parquet(directory)
    pd.testing.assert_frame_equal(after, before)


def test_invalid_parquet_is_wrapped_in_concise_storage_error(tmp_path: Path) -> None:
    directory = tmp_path / "processed"
    directory.mkdir()
    (directory / "broken.parquet").write_text("not parquet", encoding="utf-8")
    with pytest.raises(StorageError, match="Could not read processed Parquet data") as raised:
        read_processed_parquet(directory)
    assert "ArrowInvalid" not in str(raised.value)


def test_hive_partition_conflict_is_rejected(tmp_path: Path, hourly_frame: pd.DataFrame) -> None:
    directory = tmp_path / "processed"
    partition = directory / "region=PJM" / "year=2024" / "month=1"
    partition.mkdir(parents=True)
    conflicting = hourly_frame.iloc[:1].copy()
    conflicting["region"] = "MISO"
    conflicting.to_parquet(partition / "part.parquet", index=False)
    with pytest.raises(StorageError, match="conflicts with its Hive partition"):
        read_processed_parquet(directory)


def test_raw_response_uses_timestamped_json(tmp_path: Path) -> None:
    path = write_raw_response([{"response": {"data": []}}], tmp_path)
    assert path.name.startswith("eia_region_")
    assert path.suffix == ".json"
