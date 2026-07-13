"""Rolling ML evaluation chronology and metric slicing tests."""

import pandas as pd
import pytest

from gridmind.exceptions import PredictionValidationError
from gridmind.features.contracts import FeatureSpecification
from gridmind.forecasting.baselines import SeasonalNaiveForecaster
from gridmind.models.lightgbm_model import LightGBMGlobalForecaster
from gridmind.training.evaluator import (
    evaluate_baseline,
    evaluate_model,
    relative_improvement,
    summarize_predictions,
)


def test_rolling_ml_evaluation_has_all_slices(ml_hourly_frame: pd.DataFrame) -> None:
    frame = ml_hourly_frame.groupby("region", observed=True).head(120).copy()
    specification = FeatureSpecification.create(lags=(1, 24), rolling_windows=(3, 24))
    result = evaluate_model(
        frame,
        lambda: LightGBMGlobalForecaster(
            specification=specification,
            n_jobs=1,
            params={"n_estimators": 15},
        ),
        horizon=4,
        windows=2,
        step_size=4,
    )
    assert len(result.predictions) == 16
    assert result.predictions["forecast_step"].min() == 1
    assert result.predictions["forecast_step"].max() == 4
    assert len(result.window_metrics) == 2
    assert len(result.horizon_metrics) == 4
    assert set(result.region_metrics["region"]) == {"MISO", "PJM"}
    assert (result.predictions["forecast_origin"] < result.predictions["timestamp_utc"]).all()
    assert result.training_seconds > 0


def test_baseline_comparison_and_negative_improvement(
    ml_hourly_frame: pd.DataFrame,
) -> None:
    frame = ml_hourly_frame.groupby("region", observed=True).head(100).copy()
    result = evaluate_baseline(
        frame,
        lambda: SeasonalNaiveForecaster(24),
        horizon=4,
        windows=2,
        step_size=4,
    )
    assert result.overall_metrics["wape"] >= 0
    assert relative_improvement(2.0, 1.0) == -1.0
    assert relative_improvement(0.5, 1.0) == 0.5


@pytest.mark.parametrize("value", [-1.0, float("inf")])
def test_invalid_predictions_are_rejected(value: float) -> None:
    frame = pd.DataFrame(
        {
            "actual_demand_mw": [1.0, 2.0],
            "predicted_demand_mw": [value, 2.0],
            "validation_window": [1, 1],
            "forecast_step": [1, 2],
            "region": ["PJM", "PJM"],
        }
    )
    with pytest.raises(PredictionValidationError):
        summarize_predictions(frame, training_seconds=0.0, prediction_seconds=0.0)
