"""Leakage, ordering, gaps, calendar, lag, and rolling feature tests."""

import math

import pandas as pd
import pytest

from gridmind.features.builder import FeatureBuilder
from gridmind.features.contracts import FeatureSpecification


@pytest.fixture()
def small_specification() -> FeatureSpecification:
    return FeatureSpecification.create(lags=(1, 2, 24), rolling_windows=(3, 24))


def test_calendar_cyclical_and_feature_order(
    hourly_frame: pd.DataFrame, small_specification: FeatureSpecification
) -> None:
    result = FeatureBuilder(small_specification).build_training(hourly_frame.iloc[:48])
    first = result.frame.iloc[0]
    assert first["hour"] == 0.0
    assert first["day_of_week"] == 1.0
    assert first["hour_sin"] == pytest.approx(0.0)
    assert first["hour_cos"] == pytest.approx(1.0)
    assert list(result.frame[list(small_specification.feature_names)].columns) == list(
        small_specification.feature_names
    )
    assert small_specification.required_history == 24


def test_lags_and_rolling_are_shifted(
    hourly_frame: pd.DataFrame, small_specification: FeatureSpecification
) -> None:
    source = hourly_frame.iloc[:50].copy()
    result = FeatureBuilder(small_specification).build_training(source)
    row = result.frame.loc[result.frame["timestamp_utc"] == source.loc[24, "timestamp_utc"]].iloc[0]
    assert row["demand_lag_1"] == source.loc[23, "demand_mw"]
    assert row["demand_lag_24"] == source.loc[0, "demand_mw"]
    assert row["demand_rolling_mean_3"] == pytest.approx(source.loc[21:23, "demand_mw"].mean())
    assert row["demand_rolling_max_24"] == source.loc[0:23, "demand_mw"].max()


def test_regions_are_isolated_and_future_perturbation_is_causal(
    ml_hourly_frame: pd.DataFrame, small_specification: FeatureSpecification
) -> None:
    source = ml_hourly_frame.groupby("region", observed=True).head(60).copy()
    builder = FeatureBuilder(small_specification)
    original = builder.build_training(source).frame
    cutoff = pd.Timestamp("2024-01-02 12:00", tz="UTC")
    modified = source.copy()
    modified.loc[modified["timestamp_utc"] > cutoff, "demand_mw"] += 99999.0
    rebuilt = builder.build_training(modified).frame
    columns = list(small_specification.feature_names)
    pd.testing.assert_frame_equal(
        original.loc[original["timestamp_utc"] <= cutoff, columns].reset_index(drop=True),
        rebuilt.loc[rebuilt["timestamp_utc"] <= cutoff, columns].reset_index(drop=True),
    )
    timestamp = pd.Timestamp("2024-01-02 00:00", tz="UTC")
    pjm = original.loc[
        (original["region"] == "PJM") & (original["timestamp_utc"] == timestamp)
    ].iloc[0]
    miso = original.loc[
        (original["region"] == "MISO") & (original["timestamp_utc"] == timestamp)
    ].iloc[0]
    assert miso["demand_lag_1"] - pjm["demand_lag_1"] == 300.0


def test_gaps_and_missing_targets_are_reported(
    hourly_frame: pd.DataFrame, small_specification: FeatureSpecification
) -> None:
    source = hourly_frame.iloc[:80].drop(index=40).copy()
    source.loc[55, "demand_mw"] = float("nan")
    result = FeatureBuilder(small_specification).build_training(source)
    assert result.report.timestamp_gap_count == 1
    assert result.report.gap_affected_rows > 1
    assert result.report.missing_target_rows == 1
    assert result.report.insufficient_contiguous_history_rows > 1
    assert result.report.removed_rows > small_specification.required_history
    assert result.frame["demand_mw"].notna().all()


def test_future_row_uses_only_prior_history(
    hourly_frame: pd.DataFrame, small_specification: FeatureSpecification
) -> None:
    history = hourly_frame.iloc[:48]
    timestamp = history["timestamp_utc"].max() + pd.Timedelta(hours=1)
    row = FeatureBuilder(small_specification).build_future_row(
        history, region="PJM", timestamp=timestamp
    )
    assert row.loc[0, "demand_lag_1"] == history["demand_mw"].iloc[-1]
    assert math.isfinite(row.loc[0, "demand_rolling_std_24"])


def test_feature_cache_is_content_scoped(
    hourly_frame: pd.DataFrame, small_specification: FeatureSpecification
) -> None:
    builder = FeatureBuilder(small_specification, cache_enabled=True)
    first = builder.build_training(hourly_frame.iloc[:50])
    second = builder.build_training(hourly_frame.iloc[:50].copy())
    assert first is second
    changed = hourly_frame.iloc[:50].copy()
    changed.loc[49, "demand_mw"] += 1.0
    third = builder.build_training(changed)
    assert third is not first
