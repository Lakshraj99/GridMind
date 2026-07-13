"""Small shared helpers for recursive global tree forecasting."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC
from typing import Any

import numpy as np
import pandas as pd

from gridmind.exceptions import InsufficientHistoryError, ModelTrainingError
from gridmind.features.builder import FeatureBuilder

PREDICTION_COLUMNS = [
    "region",
    "timestamp_utc",
    "forecast_origin",
    "predicted_demand_mw",
    "model_name",
    "model_version",
    "run_id",
    "created_at_utc",
]


def recursive_predict(
    *,
    history: pd.DataFrame,
    horizon: int,
    builder: FeatureBuilder,
    model_name: str,
    model_version: str,
    run_id: str,
    predict_one: Callable[[pd.DataFrame], Any],
) -> pd.DataFrame:
    """Generate multi-region recursive forecasts from history only."""
    if horizon <= 0:
        raise ValueError("Forecast horizon must be positive.")
    working = history[["region", "timestamp_utc", "demand_mw"]].copy()
    working["timestamp_utc"] = pd.to_datetime(working["timestamp_utc"], utc=True)
    regions = sorted(str(value) for value in working["region"].unique())
    origins = working.groupby("region", observed=True)["timestamp_utc"].max()
    if len(origins) == 0 or origins.nunique() != 1:
        raise InsufficientHistoryError(
            "All regions must have history through the same forecast origin."
        )
    origin = pd.Timestamp(origins.iloc[0])
    created_at = pd.Timestamp.now(tz=UTC)
    rows: list[dict[str, Any]] = []
    for step in range(1, horizon + 1):
        timestamp = origin + pd.Timedelta(hours=step)
        step_predictions: list[tuple[str, float]] = []
        for region in regions:
            feature_row = builder.build_future_row(working, region=region, timestamp=timestamp)
            raw_prediction = predict_one(feature_row)
            prediction = float(np.asarray(raw_prediction).reshape(-1)[0])
            if not np.isfinite(prediction):
                raise ModelTrainingError(
                    f"Model {model_name} produced a non-finite prediction for {region}."
                )
            step_predictions.append((region, prediction))
            rows.append(
                {
                    "region": region,
                    "timestamp_utc": timestamp,
                    "forecast_origin": origin,
                    "predicted_demand_mw": prediction,
                    "model_name": model_name,
                    "model_version": model_version,
                    "run_id": run_id,
                    "created_at_utc": created_at,
                }
            )
        appended = pd.DataFrame(
            [
                {"region": region, "timestamp_utc": timestamp, "demand_mw": prediction}
                for region, prediction in step_predictions
            ]
        )
        working = pd.concat([working, appended], ignore_index=True)
    return pd.DataFrame(rows)[PREDICTION_COLUMNS]
