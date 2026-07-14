"""Target definitions and demand/renewable net-load alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from gridmind.exceptions import TargetForecastError

SUPPORTED_TARGETS = (
    "demand_mw",
    "solar_generation_mw",
    "wind_generation_mw",
    "total_renewable_generation_mw",
    "net_load_mw",
)


@dataclass(frozen=True)
class TargetDefinition:
    name: str
    source_table: str
    nonnegative: bool
    registry_setting: str
    weather_required: bool


TARGET_DEFINITIONS = {
    "demand_mw": TargetDefinition(
        "demand_mw", "hourly_grid_data", True, "demand_weather_model_name", True
    ),
    "solar_generation_mw": TargetDefinition(
        "solar_generation_mw", "hourly_renewable_generation", True, "solar_model_name", True
    ),
    "wind_generation_mw": TargetDefinition(
        "wind_generation_mw", "hourly_renewable_generation", True, "wind_model_name", True
    ),
    "total_renewable_generation_mw": TargetDefinition(
        "total_renewable_generation_mw",
        "hourly_renewable_generation",
        True,
        "total_renewable_model_name",
        True,
    ),
    "net_load_mw": TargetDefinition(
        "net_load_mw", "target_net_load", False, "net_load_model_name", True
    ),
}


def get_target_definition(target: str) -> TargetDefinition:
    try:
        return TARGET_DEFINITIONS[target]
    except KeyError as exc:
        raise TargetForecastError(
            f"Unsupported target '{target}'. Choose from {list(SUPPORTED_TARGETS)}."
        ) from exc


def compute_net_load(
    demand: pd.DataFrame, renewable: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute demand minus total renewables only on complete timestamp overlap."""
    left = demand[["region", "timestamp_utc", "demand_mw"]].copy()
    right = renewable[["region", "timestamp_utc", "total_renewable_generation_mw"]].copy()
    merged = left.merge(right, on=["region", "timestamp_utc"], how="outer", indicator=True)
    complete = merged[
        (merged["_merge"] == "both")
        & merged["demand_mw"].notna()
        & merged["total_renewable_generation_mw"].notna()
    ].copy()
    complete["net_load_mw"] = complete["demand_mw"] - complete["total_renewable_generation_mw"]
    if not np.isfinite(complete["net_load_mw"].to_numpy(dtype=float)).all():
        raise TargetForecastError("Net load contains non-finite values.")
    report = {
        "demand_rows": len(left),
        "renewable_rows": len(right),
        "overlap_rows": int((merged["_merge"] == "both").sum()),
        "net_load_available_rows": len(complete),
        "missing_overlap_rows": len(merged) - len(complete),
    }
    return complete[
        [
            "region",
            "timestamp_utc",
            "demand_mw",
            "total_renewable_generation_mw",
            "net_load_mw",
        ]
    ].sort_values(["region", "timestamp_utc"], ignore_index=True), report


def component_net_load(
    demand_predictions: pd.Series,
    solar_predictions: pd.Series,
    wind_predictions: pd.Series,
) -> pd.Series:
    """Combine aligned component forecasts without imposing a sign constraint."""
    if not (
        demand_predictions.index.equals(solar_predictions.index)
        and demand_predictions.index.equals(wind_predictions.index)
    ):
        raise TargetForecastError("Component net-load predictions must use identical windows.")
    return demand_predictions - solar_predictions - wind_predictions


WeatherMode = Literal["historical_oracle", "realistic_forecast"]
