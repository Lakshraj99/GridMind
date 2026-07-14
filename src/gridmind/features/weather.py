"""Leakage-labelled weather features for target forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from gridmind.continuity import detect_contiguous_segments
from gridmind.exceptions import FeatureEngineeringError
from gridmind.renewables.targets import WeatherMode

WEATHER_BASE_FEATURES = (
    "temperature_c",
    "apparent_temperature_c",
    "relative_humidity_pct",
    "precipitation_mm",
    "cloud_cover_pct",
    "wind_speed_10m_kph",
    "shortwave_radiation_wm2",
    "direct_radiation_wm2",
    "diffuse_radiation_wm2",
)


@dataclass(frozen=True)
class WeatherFeatureResult:
    frame: pd.DataFrame
    feature_names: tuple[str, ...]
    report: dict[str, Any]


def simulate_forecast_weather(historical: pd.DataFrame, *, lag_hours: int = 24) -> pd.DataFrame:
    """Create a persistence weather forecast using only observations from t-lag."""
    source = historical.loc[historical["weather_data_type"] == "historical"].copy()
    outputs: list[pd.DataFrame] = []
    for _segment, group in detect_contiguous_segments(source).frame.groupby(
        "region_segment_id", observed=True
    ):
        simulated = group.copy()
        for column in (
            *WEATHER_BASE_FEATURES,
            "wind_direction_10m_deg",
            "wind_direction_sin",
            "wind_direction_cos",
            "temperature_min_c",
            "temperature_max_c",
            "temperature_spread_c",
            "wind_speed_spread_kph",
        ):
            if column in simulated:
                simulated[column] = simulated[column].shift(lag_hours)
        simulated["weather_data_type"] = "forecast"
        outputs.append(simulated.drop(columns="region_segment_id"))
    return pd.concat(outputs, ignore_index=True).dropna(subset=list(WEATHER_BASE_FEATURES))


def build_weather_features(
    weather: pd.DataFrame,
    *,
    mode: WeatherMode = "realistic_forecast",
    lags: tuple[int, ...] = (1, 3, 6, 12, 24),
    rolling_windows: tuple[int, ...] = (3, 6, 12, 24),
) -> WeatherFeatureResult:
    """Build contemporaneous forecast/oracle fields and gap-isolated history features."""
    required_type = "forecast" if mode == "realistic_forecast" else "historical"
    selected = weather.loc[weather["weather_data_type"] == required_type].copy()
    if selected.empty:
        raise FeatureEngineeringError(
            f"Weather mode {mode} requires weather_data_type={required_type}; none is available."
        )
    if set(selected["weather_data_type"].unique()) != {required_type}:
        raise FeatureEngineeringError("Weather experiment modes cannot be mixed.")
    continuity = detect_contiguous_segments(selected)
    outputs: list[pd.DataFrame] = []
    lag_names: list[str] = []
    rolling_names: list[str] = []
    for _segment, group in continuity.frame.groupby("region_segment_id", observed=True):
        featured = group.copy().reset_index(drop=True)
        for column in WEATHER_BASE_FEATURES:
            for lag in lags:
                name = f"{column}_lag_{lag}"
                featured[name] = featured[column].shift(lag)
                lag_names.append(name)
            for window in rolling_windows:
                name = f"{column}_rolling_mean_{window}"
                featured[name] = (
                    featured[column].shift(1).rolling(window, min_periods=window).mean()
                )
                rolling_names.append(name)
        outputs.append(featured)
    result = pd.concat(outputs, ignore_index=True)
    result["temperature_squared"] = result["temperature_c"] ** 2
    result["cooling_degree"] = (result["temperature_c"] - 18.0).clip(lower=0)
    result["heating_degree"] = (18.0 - result["temperature_c"]).clip(lower=0)
    hour = result["timestamp_utc"].dt.hour.astype(float)
    month = result["timestamp_utc"].dt.month.astype(float)
    weekend = (result["timestamp_utc"].dt.dayofweek >= 5).astype(float)
    result["temperature_hour_interaction"] = result["temperature_c"] * hour
    result["temperature_weekend_interaction"] = result["temperature_c"] * weekend
    result["apparent_temperature_difference"] = (
        result["apparent_temperature_c"] - result["temperature_c"]
    )
    result["humidity_temperature_interaction"] = (
        result["relative_humidity_pct"] * result["temperature_c"]
    )
    result["wind_speed_squared"] = result["wind_speed_10m_kph"] ** 2
    result["month_sin"] = np.sin(2 * np.pi * month / 12)
    result["month_cos"] = np.cos(2 * np.pi * month / 12)
    result["season"] = ((month - 1) // 3).astype(float)
    result["solar_radiation_daylight"] = (result["shortwave_radiation_wm2"] > 0).astype(float)
    if "wind_direction_sin" not in result:
        radians = np.deg2rad(result["wind_direction_10m_deg"])
        result["wind_direction_sin"] = np.sin(radians)
        result["wind_direction_cos"] = np.cos(radians)
    derived = (
        "temperature_squared",
        "cooling_degree",
        "heating_degree",
        "temperature_hour_interaction",
        "temperature_weekend_interaction",
        "apparent_temperature_difference",
        "humidity_temperature_interaction",
        "wind_speed_squared",
        "month_sin",
        "month_cos",
        "season",
        "solar_radiation_daylight",
        "wind_direction_sin",
        "wind_direction_cos",
    )
    names = tuple(dict.fromkeys((*WEATHER_BASE_FEATURES, *derived, *lag_names, *rolling_names)))
    invalid = result[list(names)].isna().any(axis=1)
    report = {
        "weather_mode": mode,
        "source_rows": len(selected),
        "output_rows": int((~invalid).sum()),
        "excluded_rows": int(invalid.sum()),
        "timestamp_gap_count": int(continuity.segments["missing_expected_hours_before"].sum()),
    }
    return WeatherFeatureResult(
        result.drop(columns="region_segment_id").loc[~invalid].reset_index(drop=True),
        names,
        report,
    )
