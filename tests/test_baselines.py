"""Tests for deterministic baseline forecast calculations."""

import pandas as pd
import pytest

from gridmind.exceptions import InsufficientHistoryError
from gridmind.forecasting.baselines import (
    AveragedSeasonalNaiveForecaster,
    LastValueForecaster,
    SeasonalNaiveForecaster,
)


def test_last_value_forecast(hourly_frame: pd.DataFrame) -> None:
    history = hourly_frame.iloc[:10]
    result = LastValueForecaster().predict(history, horizon=3)
    assert result["predicted_demand_mw"].tolist() == [1090.0, 1090.0, 1090.0]
    assert result["forecast_origin"].max() < result["timestamp_utc"].min()


def test_daily_seasonal_naive(hourly_frame: pd.DataFrame) -> None:
    history = hourly_frame.iloc[:48]
    result = SeasonalNaiveForecaster(24).predict(history, horizon=24)
    assert result["predicted_demand_mw"].tolist() == hourly_frame.iloc[24:48]["demand_mw"].tolist()


def test_weekly_and_average_baselines(hourly_frame: pd.DataFrame) -> None:
    history = hourly_frame.iloc[:192]
    weekly = SeasonalNaiveForecaster(168).predict(history, horizon=2)
    average = AveragedSeasonalNaiveForecaster().predict(history, horizon=2)
    assert weekly["predicted_demand_mw"].tolist() == [1000.0, 1010.0]
    assert average["predicted_demand_mw"].tolist() == [1000.0, 1010.0]


def test_insufficient_or_gapped_history(hourly_frame: pd.DataFrame) -> None:
    with pytest.raises(InsufficientHistoryError, match="168"):
        SeasonalNaiveForecaster(168).predict(hourly_frame.iloc[:100])
    gapped = hourly_frame.iloc[:48].drop(index=24)
    with pytest.raises(InsufficientHistoryError, match="required historical timestamp"):
        SeasonalNaiveForecaster(24).predict(gapped, horizon=1)


def test_forecast_requires_one_region(hourly_frame: pd.DataFrame) -> None:
    frame = hourly_frame.iloc[:2].copy()
    frame.loc[1, "region"] = "MISO"
    with pytest.raises(ValueError, match="one region"):
        LastValueForecaster().predict(frame)
