"""Chronology and leakage tests for rolling-origin validation."""

import pandas as pd

from gridmind.forecasting.baselines import LastValueForecaster
from gridmind.forecasting.validation import rolling_origin_evaluate


class RecordingForecaster(LastValueForecaster):
    """Record the maximum timestamp and demand visible at each fit origin."""

    name = "recording"

    def __init__(self) -> None:
        self.max_timestamps: list[pd.Timestamp] = []
        self.max_demands: list[float] = []

    def predict(self, history: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
        self.max_timestamps.append(history["timestamp_utc"].max())
        self.max_demands.append(float(history["demand_mw"].max()))
        return super().predict(history, horizon)


def test_rolling_splits_are_chronological(hourly_frame: pd.DataFrame) -> None:
    model = RecordingForecaster()
    predictions = rolling_origin_evaluate(hourly_frame, model, horizon=12, windows=3, step_size=12)
    for origin, (_, window) in zip(
        model.max_timestamps, predictions.groupby("validation_window"), strict=True
    ):
        assert origin < window["timestamp_utc"].min()
        assert (window["forecast_origin"] == origin).all()


def test_validation_targets_are_not_visible_to_model(hourly_frame: pd.DataFrame) -> None:
    frame = hourly_frame.copy()
    frame.loc[216:, "demand_mw"] = 999999.0
    model = RecordingForecaster()
    predictions = rolling_origin_evaluate(frame, model, horizon=24, windows=1)
    assert model.max_demands == [1230.0]
    assert predictions["actual_demand_mw"].min() == 999999.0
    assert predictions["predicted_demand_mw"].max() < 999999.0
