"""Tests for baseline evaluation metrics."""

import math

import pandas as pd
import pytest

from gridmind.forecasting.metrics import calculate_metrics, evaluate_predictions


def test_metric_calculations() -> None:
    metrics = calculate_metrics(
        pd.Series([10.0, 20.0, 30.0]),
        pd.Series([12.0, 18.0, 33.0]),
        mase_scale=5.0,
    )
    assert metrics["mae"] == pytest.approx(7 / 3)
    assert metrics["rmse"] == pytest.approx(math.sqrt(17 / 3))
    assert metrics["wape"] == pytest.approx(7 / 60)
    assert metrics["mase"] == pytest.approx(7 / 15)
    assert metrics["bias"] == pytest.approx(1.0)


def test_zero_denominators_are_safe() -> None:
    metrics = calculate_metrics(pd.Series([0.0, 0.0]), pd.Series([1.0, -1.0]))
    assert math.isnan(metrics["wape"])
    assert math.isnan(metrics["mase"])


def test_evaluates_every_window() -> None:
    predictions = pd.DataFrame(
        {
            "actual_demand_mw": [1.0, 2.0, 3.0, 4.0],
            "predicted_demand_mw": [1.0, 1.0, 3.0, 3.0],
            "validation_window": [1, 1, 2, 2],
        }
    )
    report = evaluate_predictions(predictions)
    assert report["overall"]["mae"] == 0.5
    assert len(report["by_window"]) == 2
