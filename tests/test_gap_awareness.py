"""Contiguous-segment, gap-safe feature, window, and recursion tests."""

from __future__ import annotations

import pandas as pd
import pytest

from gridmind.continuity import detect_contiguous_segments, select_gap_aware_windows
from gridmind.exceptions import InsufficientHistoryError
from gridmind.features.builder import FeatureBuilder
from gridmind.features.contracts import FeatureSpecification
from gridmind.models.lightgbm_model import LightGBMGlobalForecaster
from gridmind.training.datasets import reserve_final_evaluation_history
from gridmind.training.evaluator import evaluate_model


def _frame(
    periods: int,
    *,
    regions: tuple[str, ...] = ("PJM",),
    missing: dict[str, set[int]] | None = None,
) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=periods, freq="h", tz="UTC")
    rows = []
    missing = missing or {}
    for offset, region in enumerate(regions):
        for index, timestamp in enumerate(timestamps):
            if index in missing.get(region, set()):
                continue
            rows.append(
                {
                    "region": region,
                    "timestamp_utc": timestamp,
                    "demand_mw": 1000.0 + offset * 100.0 + index,
                }
            )
    return pd.DataFrame(rows)


def test_contiguous_segment_detection_reports_gap_metadata() -> None:
    result = detect_contiguous_segments(_frame(6, missing={"PJM": {2, 3}}))
    assert result.frame["region_segment_id"].unique().tolist() == ["PJM::0001", "PJM::0002"]
    assert result.segments["row_count"].tolist() == [2, 2]
    assert result.segments["duration"].tolist() == [pd.Timedelta(hours=2)] * 2
    assert result.segments["missing_expected_hours_before"].tolist() == [0, 2]
    summary = result.summary()
    assert summary["timestamp_gap_count"] == 2
    assert summary["gap_event_count"] == 1
    assert summary["contiguous_segment_count"] == 2
    assert summary["segments"][1]["segment_start"] == "2024-01-01T04:00:00Z"
    assert summary["segments"][1]["gap_before_segment"] == "0 days 03:00:00"


def test_regions_receive_independent_stable_segments() -> None:
    result = detect_contiguous_segments(
        _frame(8, regions=("PJM", "MISO"), missing={"PJM": {2}, "MISO": {5}})
    )
    assert set(result.frame["region_segment_id"].unique()) == {
        "MISO::0001",
        "MISO::0002",
        "PJM::0001",
        "PJM::0002",
    }
    by_region = result.segments.groupby("region")["missing_expected_hours_before"].sum()
    assert by_region.to_dict() == {"MISO": 1, "PJM": 1}


def test_lags_and_rollings_restart_after_gap() -> None:
    specification = FeatureSpecification.create(lags=(1, 3), rolling_windows=(3,))
    source = _frame(10, missing={"PJM": {5}})
    result = FeatureBuilder(specification).build_training(source)
    after_gap = result.frame.loc[
        result.frame["timestamp_utc"] > pd.Timestamp("2024-01-01T05:00:00Z")
    ]
    assert after_gap["timestamp_utc"].tolist() == [pd.Timestamp("2024-01-01T09:00:00Z")]
    row = after_gap.iloc[0]
    assert row["demand_lag_1"] == 1008.0
    assert row["demand_lag_3"] == 1006.0
    assert row["demand_rolling_mean_3"] == pytest.approx((1006 + 1007 + 1008) / 3)
    assert result.report.gap_affected_rows == 3


def test_gap_aware_selection_rejects_horizon_and_history_gaps() -> None:
    selection = select_gap_aware_windows(
        _frame(106, missing={"PJM": {100}}),
        horizon=2,
        windows=1,
        step_size=2,
        required_history=10,
    )
    assert selection.origins == (pd.Timestamp("2024-01-05T01:00:00Z"),)
    reasons = [reason for item in selection.candidates for reason in item["rejection_reasons"]]
    assert any("insufficient_contiguous_history" in reason for reason in reasons)
    assert any("incomplete_forecast_horizon" in reason for reason in reasons)
    assert selection.windows[0].segments_by_region == {"PJM": "PJM::0001"}


def test_selection_searches_backward_and_returns_exact_requested_count() -> None:
    selection = select_gap_aware_windows(
        _frame(201, missing={"PJM": {185}}),
        horizon=4,
        windows=3,
        step_size=4,
        required_history=24,
    )
    assert selection.origins == tuple(
        pd.to_datetime(
            ["2024-01-08T04:00:00Z", "2024-01-08T08:00:00Z", "2024-01-08T12:00:00Z"],
            utc=True,
        )
    )
    assert len(selection.windows) == selection.requested_windows == 3
    assert len(selection.candidates) > 3


def test_insufficient_windows_error_is_actionable() -> None:
    with pytest.raises(InsufficientHistoryError) as exc_info:
        select_gap_aware_windows(_frame(31), horizon=4, windows=3, step_size=4, required_history=24)
    message = str(exc_info.value)
    for detail in (
        "requested windows=3",
        "available valid windows=1",
        "horizon=4",
        "required history=24",
        "missing expected hours=0",
        "most recent contiguous segment=PJM::0001",
    ):
        assert detail in message


def test_tuning_history_and_final_windows_are_separate_with_gaps() -> None:
    source = _frame(500, missing={"PJM": {300}})
    final = select_gap_aware_windows(source, horizon=4, windows=2, step_size=4, required_history=24)
    older, _reserved, boundary = reserve_final_evaluation_history(
        source,
        horizon=4,
        windows=2,
        step_size=4,
        required_history=24,
        selection=final,
    )
    tuning = select_gap_aware_windows(older, horizon=4, windows=2, step_size=4, required_history=24)
    assert max(window.validation_timestamps[-1] for window in tuning.windows) < boundary
    assert boundary == min(window.validation_timestamps[0] for window in final.windows)


def test_recursive_prediction_refuses_gap_inside_required_context() -> None:
    specification = FeatureSpecification.create(lags=(1, 24), rolling_windows=(3, 24))
    complete = _frame(80)
    model = LightGBMGlobalForecaster(
        specification=specification,
        n_jobs=1,
        params={"n_estimators": 10},
    ).fit(complete)
    invalid_history = complete.drop(index=70).reset_index(drop=True)
    with pytest.raises(InsufficientHistoryError, match="cannot cross a timestamp gap"):
        model.predict(invalid_history, horizon=1)


def test_future_perturbation_remains_causal_with_gaps() -> None:
    specification = FeatureSpecification.create(lags=(1, 24), rolling_windows=(3, 24))
    source = _frame(100, missing={"PJM": {40}})
    cutoff = pd.Timestamp("2024-01-04T00:00:00Z")
    original = FeatureBuilder(specification).build_training(source).frame
    modified = source.copy()
    modified.loc[modified["timestamp_utc"] > cutoff, "demand_mw"] += 100_000.0
    rebuilt = FeatureBuilder(specification).build_training(modified).frame
    columns = ["timestamp_utc", *specification.feature_names]
    pd.testing.assert_frame_equal(
        original.loc[original["timestamp_utc"] <= cutoff, columns].reset_index(drop=True),
        rebuilt.loc[rebuilt["timestamp_utc"] <= cutoff, columns].reset_index(drop=True),
    )


def test_gapped_dataset_trains_on_valid_windows() -> None:
    specification = FeatureSpecification.create(lags=(1, 24), rolling_windows=(3, 24))
    source = _frame(180, missing={"PJM": {150, 151}})
    result = evaluate_model(
        source,
        lambda: LightGBMGlobalForecaster(
            specification=specification,
            n_jobs=1,
            params={"n_estimators": 10},
        ),
        horizon=4,
        windows=2,
        step_size=4,
    )
    assert result.window_selection is not None
    assert len(result.window_selection.windows) == 2
    assert len(result.predictions) == 8


def test_fully_contiguous_window_origins_retain_previous_tail_behavior() -> None:
    selection = select_gap_aware_windows(
        _frame(100), horizon=4, windows=2, step_size=4, required_history=24
    )
    assert selection.origins == tuple(
        pd.to_datetime(["2024-01-04T19:00:00Z", "2024-01-04T23:00:00Z"], utc=True)
    )
    assert len(selection.candidates) == 2
