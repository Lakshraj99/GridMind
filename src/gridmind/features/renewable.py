"""Target-specific solar and wind derived feature helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_solar_weather_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["daylight_indicator"] = (result["shortwave_radiation_wm2"] > 0).astype(float)
    result["hour_sin_solar"] = np.sin(2 * np.pi * result["timestamp_utc"].dt.hour / 24)
    result["hour_cos_solar"] = np.cos(2 * np.pi * result["timestamp_utc"].dt.hour / 24)
    result["month_sin_solar"] = np.sin(2 * np.pi * result["timestamp_utc"].dt.month / 12)
    result["month_cos_solar"] = np.cos(2 * np.pi * result["timestamp_utc"].dt.month / 12)
    return result


def add_wind_weather_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["wind_speed_squared"] = result["wind_speed_10m_kph"] ** 2
    result["season"] = ((result["timestamp_utc"].dt.month - 1) // 3).astype(float)
    return result
