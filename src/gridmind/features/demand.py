"""Past-only demand lag and rolling feature calculations."""

from __future__ import annotations

import pandas as pd

from gridmind.features.contracts import ROLLING_STATISTICS, FeatureSpecification


def add_demand_features(frame: pd.DataFrame, specification: FeatureSpecification) -> None:
    """Add lags and shifted rolling statistics to an hourly single-region frame in place."""
    demand = frame[specification.target_name]
    for lag in specification.lags:
        frame[f"demand_lag_{lag}"] = demand.shift(lag)
    shifted = demand.shift(1)
    for window in specification.rolling_windows:
        rolling = shifted.rolling(window=window, min_periods=window)
        aggregations = {
            "mean": rolling.mean(),
            "std": rolling.std(),
            "min": rolling.min(),
            "max": rolling.max(),
        }
        for statistic in ROLLING_STATISTICS:
            frame[f"demand_rolling_{statistic}_{window}"] = aggregations[statistic]
