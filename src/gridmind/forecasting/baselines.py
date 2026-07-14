"""Leakage-safe deterministic baseline demand forecasters."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from gridmind.exceptions import InsufficientHistoryError
from gridmind.time_utils import to_utc_timestamp

FORECAST_COLUMNS = [
    "timestamp_utc",
    "region",
    "actual_demand_mw",
    "predicted_demand_mw",
    "model_name",
    "forecast_origin",
]


class BaselineForecaster(ABC):
    """Shared interface for deterministic univariate demand baselines."""

    name: str

    @abstractmethod
    def predict(self, history: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
        """Forecast future hourly demand using only observations in history."""

    @staticmethod
    def _prepare(history: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, str, pd.Timestamp]:
        if horizon <= 0:
            raise ValueError("Forecast horizon must be positive.")
        if history.empty:
            raise InsufficientHistoryError("At least one historical observation is required.")
        ordered = history.sort_values("timestamp_utc").copy()
        ordered["timestamp_utc"] = pd.to_datetime(
            ordered["timestamp_utc"], utc=True, errors="raise"
        )
        if ordered["region"].nunique() != 1:
            raise ValueError("A forecast history must contain exactly one region.")
        if ordered.duplicated("timestamp_utc").any():
            raise ValueError("Forecast history contains duplicate timestamps.")
        origin = to_utc_timestamp(ordered["timestamp_utc"].iloc[-1])
        return ordered, str(ordered["region"].iloc[0]), origin

    def _result(
        self,
        timestamps: pd.DatetimeIndex,
        region: str,
        values: list[float],
        origin: pd.Timestamp,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp_utc": timestamps,
                "region": region,
                "actual_demand_mw": pd.Series([float("nan")] * len(values), dtype="float64"),
                "predicted_demand_mw": values,
                "model_name": self.name,
                "forecast_origin": origin,
            }
        )[FORECAST_COLUMNS]


class LastValueForecaster(BaselineForecaster):
    """Repeat the latest observed demand for every forecast hour."""

    name = "last_value"

    def predict(self, history: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
        ordered, region, origin = self._prepare(history, horizon)
        timestamps = pd.date_range(
            origin + pd.Timedelta(hours=1), periods=horizon, freq="h", tz="UTC"
        )
        value = float(ordered["demand_mw"].iloc[-1])
        return self._result(timestamps, region, [value] * horizon, origin)


class SeasonalNaiveForecaster(BaselineForecaster):
    """Repeat demand observed at the same hour one season ago."""

    def __init__(self, lag: int) -> None:
        if lag <= 0:
            raise ValueError("Seasonal lag must be positive.")
        self.lag = lag
        self.name = f"seasonal_naive_{lag}h"

    def predict(self, history: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
        ordered, region, origin = self._prepare(history, horizon)
        if len(ordered) < self.lag:
            raise InsufficientHistoryError(
                f"Model {self.name} requires at least {self.lag} hourly observations; "
                f"received {len(ordered)}."
            )
        timestamps = pd.date_range(
            origin + pd.Timedelta(hours=1), periods=horizon, freq="h", tz="UTC"
        )
        known = {
            to_utc_timestamp(timestamp): float(value)
            for timestamp, value in zip(ordered["timestamp_utc"], ordered["demand_mw"], strict=True)
        }
        predictions: list[float] = []
        for timestamp in timestamps:
            source = timestamp - pd.Timedelta(hours=self.lag)
            if source not in known:
                raise InsufficientHistoryError(
                    f"Model {self.name} cannot forecast {timestamp.isoformat()}; "
                    f"required historical timestamp {source.isoformat()} is missing."
                )
            prediction = known[source]
            predictions.append(prediction)
            known[timestamp] = prediction
        return self._result(timestamps, region, predictions, origin)


class AveragedSeasonalNaiveForecaster(BaselineForecaster):
    """Average the 24-hour and 168-hour seasonal-naive predictions."""

    name = "seasonal_naive_average_24h_168h"

    def predict(self, history: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
        daily = SeasonalNaiveForecaster(24).predict(history, horizon)
        weekly = SeasonalNaiveForecaster(168).predict(history, horizon)
        result = daily.copy()
        result["predicted_demand_mw"] = (
            daily["predicted_demand_mw"] + weekly["predicted_demand_mw"]
        ) / 2.0
        result["model_name"] = self.name
        return result


def all_baseline_models() -> list[BaselineForecaster]:
    """Return every baseline included in the first milestone."""
    return [
        LastValueForecaster(),
        SeasonalNaiveForecaster(24),
        SeasonalNaiveForecaster(168),
        AveragedSeasonalNaiveForecaster(),
    ]
