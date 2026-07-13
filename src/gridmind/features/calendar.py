"""Calendar features that are known before a forecast timestamp occurs."""

from __future__ import annotations

import math

import pandas as pd


def add_calendar_features(frame: pd.DataFrame, timestamp_column: str = "timestamp_utc") -> None:
    """Add deterministic UTC calendar and cyclical features in place."""
    timestamps = frame[timestamp_column].dt
    frame["hour"] = timestamps.hour.astype("float64")
    frame["day_of_week"] = timestamps.dayofweek.astype("float64")
    frame["day_of_month"] = timestamps.day.astype("float64")
    frame["week_of_year"] = timestamps.isocalendar().week.astype("float64")
    frame["month"] = timestamps.month.astype("float64")
    frame["quarter"] = timestamps.quarter.astype("float64")
    frame["is_weekend"] = (timestamps.dayofweek >= 5).astype("float64")
    frame["hour_sin"] = (2 * math.pi * frame["hour"] / 24).map(math.sin)
    frame["hour_cos"] = (2 * math.pi * frame["hour"] / 24).map(math.cos)
    frame["day_of_week_sin"] = (2 * math.pi * frame["day_of_week"] / 7).map(math.sin)
    frame["day_of_week_cos"] = (2 * math.pi * frame["day_of_week"] / 7).map(math.cos)
