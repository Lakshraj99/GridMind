"""Dataset adapters and chronological split helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
from mlforecast import MLForecast

from gridmind.continuity import WindowSelection, select_gap_aware_windows
from gridmind.features.contracts import FeatureSpecification


def to_mlforecast_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Adapt GridMind names to MLForecast's explicit global forecasting contract."""
    adapted = frame[["region", "timestamp_utc", "demand_mw"]].rename(
        columns={"region": "unique_id", "timestamp_utc": "ds", "demand_mw": "y"}
    )
    adapted["ds"] = pd.to_datetime(adapted["ds"], utc=True)
    return adapted.sort_values(["unique_id", "ds"], ignore_index=True)


def create_mlforecast_engine(
    models: Mapping[str, Any], specification: FeatureSpecification
) -> MLForecast:
    """Create an explicit MLForecast adapter for compatible recursive workflows.

    GridMind's primary wrappers retain the project-native feature contract so gaps,
    removal counts, and exact feature order stay auditable. This adapter provides the
    same hourly frequency and lag definitions when MLForecast interoperability is useful.
    """
    return MLForecast(
        models=dict(models),
        freq=specification.frequency,
        lags=list(specification.lags),
        date_features=["hour", "dayofweek", "day", "week", "month", "quarter"],
    )


def common_hourly_timestamps(frame: pd.DataFrame) -> pd.DatetimeIndex:
    """Return timestamps with an actual demand observation for every region."""
    pivot = frame.pivot(index="timestamp_utc", columns="region", values="demand_mw")
    return pd.DatetimeIndex(pivot.dropna().index).sort_values()


def final_evaluation_start(
    frame: pd.DataFrame,
    *,
    horizon: int,
    windows: int,
    step_size: int,
    required_history: int = 1,
    selection: WindowSelection | None = None,
) -> pd.Timestamp:
    """Return the earliest timestamp reserved for untouched final evaluation."""
    selected = selection or select_gap_aware_windows(
        frame,
        horizon=horizon,
        windows=windows,
        step_size=step_size,
        required_history=required_history,
    )
    return min(window.validation_timestamps[0] for window in selected.windows)


def reserve_final_evaluation_history(
    frame: pd.DataFrame,
    *,
    horizon: int,
    windows: int,
    step_size: int,
    required_history: int = 1,
    selection: WindowSelection | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Separate older tuning history from untouched final windows."""
    start = final_evaluation_start(
        frame,
        horizon=horizon,
        windows=windows,
        step_size=step_size,
        required_history=required_history,
        selection=selection,
    )
    older = frame.loc[frame["timestamp_utc"] < start].copy()
    final = frame.loc[frame["timestamp_utc"] >= start].copy()
    return older, final, start
