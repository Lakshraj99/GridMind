"""Optuna tuning separation, artifacts, failure handling, and MLForecast adapter tests."""

from pathlib import Path

import pandas as pd
import pytest

from gridmind.exceptions import InsufficientHistoryError
from gridmind.features.contracts import FeatureSpecification
from gridmind.training.datasets import (
    create_mlforecast_engine,
    reserve_final_evaluation_history,
    to_mlforecast_frame,
)
from gridmind.training.tuning import tune_model


def test_mlforecast_adapter_uses_explicit_contract(ml_hourly_frame: pd.DataFrame) -> None:
    specification = FeatureSpecification.create(lags=(1, 24), rolling_windows=(3, 24))
    adapted = to_mlforecast_frame(ml_hourly_frame.iloc[:10])
    engine = create_mlforecast_engine({}, specification)
    assert list(adapted.columns) == ["unique_id", "ds", "y"]
    assert engine.ts.freq == "h"
    assert engine.ts.lags == [1, 24]


def test_tuning_reserves_final_windows_and_writes_artifacts(
    tmp_path: Path, ml_hourly_frame: pd.DataFrame
) -> None:
    older, final, boundary = reserve_final_evaluation_history(
        ml_hourly_frame, horizon=2, windows=1, step_size=2
    )
    assert older["timestamp_utc"].max() < boundary
    assert final["timestamp_utc"].min() == boundary
    result = tune_model(
        ml_hourly_frame,
        "lightgbm",
        final_horizon=2,
        final_windows=1,
        step_size=2,
        tuning_windows=1,
        trials=2,
        timeout_seconds=None,
        random_seed=9,
        n_jobs=1,
        output_dir=tmp_path / "tuning",
    )
    assert result.best_params
    assert result.tuning_end < result.final_evaluation_start
    assert len(result.study.trials) == 2
    assert all(path.exists() for path in result.artifact_paths)


def test_failed_trials_are_reported(tmp_path: Path, hourly_frame: pd.DataFrame) -> None:
    with pytest.raises(InsufficientHistoryError, match="required history=336"):
        tune_model(
            hourly_frame.iloc[:40],
            "lightgbm",
            final_horizon=2,
            final_windows=1,
            step_size=2,
            tuning_windows=1,
            trials=1,
            timeout_seconds=None,
            random_seed=1,
            n_jobs=1,
            output_dir=tmp_path,
        )
