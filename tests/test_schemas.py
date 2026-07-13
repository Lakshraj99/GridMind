"""Canonical schema constraint tests."""

import pandas as pd
import pytest

from gridmind.data.schemas import validate_processed_data
from gridmind.exceptions import DataValidationError


def test_schema_accepts_canonical_data(hourly_frame: pd.DataFrame) -> None:
    assert len(validate_processed_data(hourly_frame)) == 240


def test_schema_rejects_negative_demand(hourly_frame: pd.DataFrame) -> None:
    hourly_frame.loc[0, "demand_mw"] = -1.0
    with pytest.raises(DataValidationError, match="demand_mw"):
        validate_processed_data(hourly_frame)


def test_schema_rejects_duplicates_and_unsorted_data(hourly_frame: pd.DataFrame) -> None:
    duplicate = pd.concat([hourly_frame, hourly_frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataValidationError, match="unique"):
        validate_processed_data(duplicate)
    shuffled = hourly_frame.iloc[::-1]
    with pytest.raises(DataValidationError, match="sorted"):
        validate_processed_data(shuffled)
