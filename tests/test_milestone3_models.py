"""Weather features, target models/evaluation, serialization, and forecast storage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from gridmind.data.target_storage import TargetForecastStorage, validate_target_forecasts
from gridmind.exceptions import (
    FeatureEngineeringError,
    InsufficientHistoryError,
    TargetForecastError,
)
from gridmind.features.renewable import add_solar_weather_features, add_wind_weather_features
from gridmind.features.weather import build_weather_features, simulate_forecast_weather
from gridmind.models.serialization import load_model_bundle
from gridmind.models.target_factory import TargetForecaster, create_target_model
from gridmind.training.multi_target import (
    evaluate_target_model,
    target_specific_metrics,
    tune_target_model,
)


def _regional_weather(
    periods: int = 100,
    *,
    data_type: str = "forecast",
    regions: tuple[str, ...] = ("PJM",),
) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=periods, freq="h", tz="UTC")
    rows: list[dict[str, Any]] = []
    for region_index, region in enumerate(regions):
        for index, timestamp in enumerate(timestamps):
            rows.append(
                {
                    "region": region,
                    "timestamp_utc": timestamp,
                    "weather_data_type": data_type,
                    "temperature_c": 10.0 + region_index + index / 100,
                    "apparent_temperature_c": 9.0 + index / 100,
                    "relative_humidity_pct": 50.0,
                    "precipitation_mm": 0.0,
                    "cloud_cover_pct": 20.0,
                    "wind_speed_10m_kph": 15.0,
                    "wind_direction_10m_deg": 90.0,
                    "wind_direction_sin": 1.0,
                    "wind_direction_cos": 0.0,
                    "shortwave_radiation_wm2": 100.0 if 6 <= timestamp.hour <= 18 else 0.0,
                    "direct_radiation_wm2": 60.0,
                    "diffuse_radiation_wm2": 40.0,
                    "temperature_min_c": 9.0,
                    "temperature_max_c": 11.0,
                    "temperature_spread_c": 2.0,
                    "wind_speed_spread_kph": 1.0,
                    "ingestion_timestamp_utc": pd.Timestamp("2024-02-01", tz="UTC"),
                    "data_source": "fixture",
                }
            )
    return pd.DataFrame(rows)


def test_weather_features_modes_lags_gaps_and_multi_region() -> None:
    weather = _regional_weather(40, regions=("PJM", "MISO"))
    weather = weather.drop(
        weather.index[
            (weather["region"] == "PJM")
            & (weather["timestamp_utc"] == pd.Timestamp("2024-01-01T20:00:00Z"))
        ]
    )
    result = build_weather_features(weather, lags=(1,), rolling_windows=(3,))
    assert result.report["weather_mode"] == "realistic_forecast"
    assert result.report["timestamp_gap_count"] == 1
    assert "temperature_squared" in result.feature_names
    assert {"month_sin", "month_cos", "season", "wind_speed_squared"}.issubset(result.feature_names)
    assert "temperature_c_lag_1" in result.feature_names
    pjm_after_gap = result.frame.loc[
        (result.frame["region"] == "PJM")
        & (result.frame["timestamp_utc"] > pd.Timestamp("2024-01-01T20:00:00Z"))
    ]
    assert pjm_after_gap["timestamp_utc"].min() == pd.Timestamp("2024-01-02T00:00:00Z")
    with pytest.raises(FeatureEngineeringError, match="requires"):
        build_weather_features(weather.assign(weather_data_type="historical"))
    oracle = build_weather_features(
        weather.assign(weather_data_type="historical"),
        mode="historical_oracle",
        lags=(1,),
        rolling_windows=(3,),
    )
    assert oracle.report["weather_mode"] == "historical_oracle"


def test_forecast_weather_simulation_is_past_only() -> None:
    historical = _regional_weather(60, data_type="historical")
    original = simulate_forecast_weather(historical, lag_hours=24)
    changed = historical.copy()
    changed.loc[
        changed["timestamp_utc"] >= pd.Timestamp("2024-01-03T00:00:00Z"), "temperature_c"
    ] += 999
    rebuilt = simulate_forecast_weather(changed, lag_hours=24)
    cutoff = pd.Timestamp("2024-01-03T23:00:00Z")
    pd.testing.assert_series_equal(
        original.loc[original["timestamp_utc"] <= cutoff, "temperature_c"].reset_index(drop=True),
        rebuilt.loc[rebuilt["timestamp_utc"] <= cutoff, "temperature_c"].reset_index(drop=True),
    )


def test_solar_and_wind_derived_features() -> None:
    weather = _regional_weather(24)
    solar = add_solar_weather_features(weather)
    wind = add_wind_weather_features(weather)
    assert solar["daylight_indicator"].sum() == 13
    assert solar["hour_sin_solar"].abs().max() == pytest.approx(1.0)
    assert "month_cos_solar" in solar
    assert (wind["wind_speed_squared"] == 225.0).all()
    assert set(wind["season"]) == {0.0}


def _target_frame(periods: int = 100, future: int = 0) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=periods + future, freq="h", tz="UTC")
    values = [100.0 + index % 24 for index in range(periods)] + [float("nan")] * future
    return pd.DataFrame(
        {
            "region": "PJM",
            "timestamp_utc": timestamps,
            "solar_generation_mw": values,
            "temperature_c": [10.0 + index / 100 for index in range(periods + future)],
        }
    )


def test_target_model_fit_predict_clip_and_serialization(tmp_path: Path) -> None:
    model = create_target_model(
        "lightgbm",
        "solar_generation_mw",
        weather_features=("temperature_c",),
        lags=(1, 24),
        rolling_windows=(3, 24),
        n_jobs=1,
        params={"n_estimators": 10},
    ).fit(_target_frame(80))
    forecasts = model.predict(_target_frame(80, future=2), horizon=2)
    assert len(forecasts) == 2
    assert (forecasts["predicted_value"] >= 0).all()
    path = model.save(tmp_path / "bundle.joblib", metadata={"target": "solar_generation_mw"})
    loaded = load_model_bundle(path)
    assert isinstance(loaded.model, TargetForecaster)
    assert loaded.model.target == "solar_generation_mw"

    class NegativeEstimator:
        def predict(self, _frame: pd.DataFrame) -> np.ndarray:
            return np.array([-1.0])

    model._estimator = NegativeEstimator()  # type: ignore[attr-defined]
    clipped = model.predict(_target_frame(80, future=1), horizon=1)
    assert clipped["predicted_value"].iloc[0] == 0
    assert model.clipping_count == 1


def test_target_model_rejects_gap_and_unsupported_model() -> None:
    model = create_target_model(
        "lightgbm",
        "wind_generation_mw",
        weather_features=(),
        lags=(1, 3),
        rolling_windows=(3,),
        n_jobs=1,
        params={"n_estimators": 5},
    )
    frame = _target_frame(20).rename(columns={"solar_generation_mw": "wind_generation_mw"})
    model.fit(frame)
    gapped = frame.drop(index=18)
    future = pd.concat(
        [
            gapped,
            pd.DataFrame(
                {
                    "region": ["PJM"],
                    "timestamp_utc": [frame["timestamp_utc"].max() + pd.Timedelta(hours=1)],
                    "wind_generation_mw": [float("nan")],
                    "temperature_c": [10.0],
                }
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(InsufficientHistoryError, match="gap"):
        model.predict(future, horizon=1)
    with pytest.raises(TargetForecastError):
        create_target_model("xgboost", "demand_mw")


def test_target_evaluation_shared_windows_and_solar_metrics() -> None:
    frame = _target_frame(100)
    frame["solar_radiation_daylight"] = (frame["timestamp_utc"].dt.hour.between(6, 18)).astype(
        float
    )
    result = evaluate_target_model(
        frame,
        "solar_generation_mw",
        lambda: create_target_model(
            "lightgbm",
            "solar_generation_mw",
            weather_features=("temperature_c", "solar_radiation_daylight"),
            lags=(1, 24),
            rolling_windows=(3, 24),
            n_jobs=1,
            params={"n_estimators": 5},
        ),
        horizon=2,
        windows=2,
        step_size=2,
    )
    assert len(result.selection.windows) == 2
    assert len(result.predictions) == 4
    assert "daytime_mae" in result.target_metrics
    metrics = target_specific_metrics(
        pd.DataFrame({"actual_value": [0.0, 10.0], "predicted_value": [1.0, 8.0]}),
        "wind_generation_mw",
    )
    assert metrics["zero_generation_mae"] == 1.0


def test_target_forecast_contract_and_idempotency(tmp_path: Path) -> None:
    model = create_target_model(
        "lightgbm",
        "solar_generation_mw",
        lags=(1, 3),
        rolling_windows=(3,),
        n_jobs=1,
        params={"n_estimators": 5},
    ).fit(_target_frame(20).drop(columns="temperature_c"))
    frame = model.predict(
        pd.concat(
            [
                _target_frame(20).drop(columns="temperature_c"),
                pd.DataFrame(
                    {
                        "region": ["PJM"],
                        "timestamp_utc": [pd.Timestamp("2024-01-01T20:00:00Z")],
                        "solar_generation_mw": [float("nan")],
                    }
                ),
            ],
            ignore_index=True,
        ),
        horizon=1,
    )
    storage = TargetForecastStorage(tmp_path / "forecast.duckdb")
    assert storage.upsert(frame) == 1
    assert storage.upsert(frame) == 1
    assert len(storage.read(target="solar_generation_mw")) == 1
    with pytest.raises(TargetForecastError):
        validate_target_forecasts(frame.assign(target="coal"))


def test_target_tuning_uses_fixed_gap_safe_windows(tmp_path: Path) -> None:
    frame = _target_frame(100)
    result = tune_target_model(
        frame,
        "solar_generation_mw",
        "lightgbm",
        lambda params: create_target_model(
            "lightgbm",
            "solar_generation_mw",
            weather_features=("temperature_c",),
            lags=(1, 24),
            rolling_windows=(3, 24),
            n_jobs=1,
            params={**params, "n_estimators": 5},
        ),
        horizon=2,
        windows=1,
        step_size=2,
        required_history=24,
        trials=1,
        random_seed=7,
        output_dir=tmp_path / "tuning",
    )
    assert len(result.selection.windows) == 1
    assert result.best_params
    assert all(path.exists() for path in result.artifact_paths)
