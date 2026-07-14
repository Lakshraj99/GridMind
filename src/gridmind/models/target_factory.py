"""Reusable LightGBM/CatBoost models for weather-aware Milestone 3 targets."""

from __future__ import annotations

from datetime import UTC
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from gridmind.continuity import detect_contiguous_segments
from gridmind.exceptions import InsufficientHistoryError, TargetForecastError
from gridmind.features.calendar import add_calendar_features
from gridmind.features.contracts import CALENDAR_FEATURES, FeatureSpecification
from gridmind.renewables.targets import WeatherMode, get_target_definition

TARGET_FORECAST_COLUMNS = [
    "region",
    "target",
    "forecast_origin",
    "timestamp_utc",
    "forecast_step",
    "predicted_value",
    "model_name",
    "model_version",
    "run_id",
    "weather_mode",
    "created_at_utc",
]


class TargetForecaster:
    """One recursive tree model with explicit target and future-weather contract."""

    def __init__(
        self,
        model_kind: Literal["lightgbm", "catboost"],
        target: str,
        *,
        weather_features: tuple[str, ...] = (),
        weather_mode: WeatherMode = "realistic_forecast",
        lags: tuple[int, ...] = (1, 24, 168),
        rolling_windows: tuple[int, ...] = (3, 24, 168),
        random_seed: int = 42,
        n_jobs: int = -1,
        params: dict[str, Any] | None = None,
    ) -> None:
        definition = get_target_definition(target)
        target_features = tuple(f"target_lag_{value}" for value in lags) + tuple(
            f"target_rolling_{stat}_{window}"
            for window in rolling_windows
            for stat in ("mean", "std", "min", "max")
        )
        names = ("region", *CALENDAR_FEATURES, *target_features, *weather_features)
        self.specification = FeatureSpecification(
            feature_names=tuple(names),
            feature_types={name: ("category" if name == "region" else "float64") for name in names},
            lags=lags,
            rolling_windows=rolling_windows,
            calendar_features=CALENDAR_FEATURES,
            target_name=target,
            creation_version="3.0",
        )
        self.target = target
        self.weather_features = weather_features
        self.weather_mode = weather_mode
        self.nonnegative = definition.nonnegative
        self.model_kind = model_kind
        self.name = f"{model_kind}_{target}"
        self.model_version = "unregistered"
        self.run_id = ""
        self.clipping_count = 0
        defaults: dict[str, Any]
        if model_kind == "lightgbm":
            defaults = {
                "n_estimators": 120,
                "learning_rate": 0.05,
                "objective": "regression_l1",
                "random_state": random_seed,
                "n_jobs": n_jobs,
                "verbosity": -1,
            }
            defaults.update(params or {})
            self._estimator: Any = LGBMRegressor(**defaults)
        else:
            defaults = {
                "iterations": 120,
                "learning_rate": 0.05,
                "depth": 7,
                "loss_function": "MAE",
                "random_seed": random_seed,
                "thread_count": n_jobs,
                "verbose": False,
                "allow_writing_files": False,
            }
            defaults.update(params or {})
            self._estimator = CatBoostRegressor(**defaults)
        self._params = defaults

    @property
    def estimator(self) -> Any:
        return self._estimator

    def fit(
        self, frame: pd.DataFrame, validation_frame: pd.DataFrame | None = None
    ) -> TargetForecaster:
        del validation_frame
        built = self._build_training(frame)
        x = self.prepare_feature_matrix(built[list(self.specification.feature_names)])
        kwargs = {"cat_features": ["region"]} if self.model_kind == "catboost" else {}
        self._estimator.fit(x, built[self.target], **kwargs)
        return self

    def predict(self, history: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
        if horizon <= 0:
            raise ValueError("Prediction horizon must be positive.")
        source = history.copy()
        source["timestamp_utc"] = pd.to_datetime(source["timestamp_utc"], utc=True)
        observed = source.loc[source[self.target].notna()].copy()
        origins = observed.groupby("region", observed=True)["timestamp_utc"].max()
        if origins.empty or origins.nunique() != 1:
            raise InsufficientHistoryError("Targets must share one forecast origin across regions.")
        origin = pd.Timestamp(origins.iloc[0])
        rows: list[dict[str, Any]] = []
        working = observed[["region", "timestamp_utc", self.target]].copy()
        self.clipping_count = 0
        for step in range(1, horizon + 1):
            timestamp = origin + pd.Timedelta(hours=step)
            additions: list[dict[str, Any]] = []
            for region in sorted(str(value) for value in observed["region"].unique()):
                weather_row = source.loc[
                    (source["region"] == region) & (source["timestamp_utc"] == timestamp)
                ]
                if self.weather_features and weather_row.empty:
                    raise InsufficientHistoryError(
                        f"Future {self.weather_mode} weather is missing for {region} "
                        f"at {timestamp}."
                    )
                feature = self._future_row(working, weather_row, region, timestamp)
                raw = float(
                    np.asarray(
                        self._estimator.predict(self.prepare_feature_matrix(feature))
                    ).reshape(-1)[0]
                )
                value = raw
                if self.nonnegative and raw < 0:
                    value = 0.0
                    self.clipping_count += 1
                rows.append(
                    {
                        "region": region,
                        "target": self.target,
                        "forecast_origin": origin,
                        "timestamp_utc": timestamp,
                        "forecast_step": step,
                        "predicted_value": value,
                        "model_name": self.name,
                        "model_version": self.model_version,
                        "run_id": self.run_id,
                        "weather_mode": self.weather_mode,
                        "created_at_utc": pd.Timestamp.now(tz=UTC),
                    }
                )
                additions.append({"region": region, "timestamp_utc": timestamp, self.target: value})
            working = pd.concat([working, pd.DataFrame(additions)], ignore_index=True)
        return pd.DataFrame(rows)[TARGET_FORECAST_COLUMNS]

    def _future_row(
        self,
        working: pd.DataFrame,
        weather: pd.DataFrame,
        region: str,
        timestamp: pd.Timestamp,
    ) -> pd.DataFrame:
        values = working.loc[working["region"] == region].set_index("timestamp_utc")[self.target]
        context = pd.date_range(
            end=timestamp - pd.Timedelta(hours=1),
            periods=self.specification.required_history,
            freq="h",
        )
        if values.reindex(context).isna().any():
            raise InsufficientHistoryError("Recursive target forecasting cannot cross a gap.")
        row = pd.DataFrame({"timestamp_utc": [timestamp], "region": [region]})
        add_calendar_features(row)
        for lag in self.specification.lags:
            row[f"target_lag_{lag}"] = float(values.loc[timestamp - pd.Timedelta(hours=lag)])
        for window in self.specification.rolling_windows:
            recent = values.reindex(
                pd.date_range(end=timestamp - pd.Timedelta(hours=1), periods=window, freq="h")
            )
            for stat, value in (
                ("mean", recent.mean()),
                ("std", recent.std()),
                ("min", recent.min()),
                ("max", recent.max()),
            ):
                row[f"target_rolling_{stat}_{window}"] = float(value)
        for name in self.weather_features:
            row[name] = float(weather.iloc[0][name])
        return row[list(self.specification.feature_names)]

    def _build_training(self, frame: pd.DataFrame) -> pd.DataFrame:
        required = {"region", "timestamp_utc", self.target, *self.weather_features}
        missing = required.difference(frame.columns)
        if missing:
            raise TargetForecastError(f"Target training data is missing {sorted(missing)}.")
        source = frame.loc[frame[self.target].notna()].copy()
        source["timestamp_utc"] = pd.to_datetime(source["timestamp_utc"], utc=True)
        outputs: list[pd.DataFrame] = []
        for _segment, group in detect_contiguous_segments(source).frame.groupby(
            "region_segment_id", observed=True
        ):
            featured = group.copy().reset_index(drop=True)
            add_calendar_features(featured)
            target = featured[self.target]
            for lag in self.specification.lags:
                featured[f"target_lag_{lag}"] = target.shift(lag)
            shifted = target.shift(1)
            for window in self.specification.rolling_windows:
                rolling = shifted.rolling(window, min_periods=window)
                featured[f"target_rolling_mean_{window}"] = rolling.mean()
                featured[f"target_rolling_std_{window}"] = rolling.std()
                featured[f"target_rolling_min_{window}"] = rolling.min()
                featured[f"target_rolling_max_{window}"] = rolling.max()
            outputs.append(featured)
        result = pd.concat(outputs, ignore_index=True)
        return result.dropna(subset=[*self.specification.feature_names, self.target]).reset_index(
            drop=True
        )

    def prepare_feature_matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame[list(self.specification.feature_names)].copy()
        if self.model_kind == "lightgbm":
            result["region"] = result["region"].astype("category")
        else:
            result["region"] = result["region"].astype(str)
        return result

    def training_features(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        built = self._build_training(frame)
        return (
            self.prepare_feature_matrix(built[list(self.specification.feature_names)]),
            built[self.target],
        )

    def get_params(self) -> dict[str, Any]:
        return dict(self._params)

    def feature_names(self) -> list[str]:
        return list(self.specification.feature_names)

    def save(self, path: Path, metadata: dict[str, Any] | None = None) -> Path:
        from gridmind.models.serialization import save_model_bundle

        return save_model_bundle(self, path, metadata=metadata)


def create_target_model(
    model_name: str,
    target: str,
    **kwargs: Any,
) -> TargetForecaster:
    if model_name not in {"lightgbm", "catboost"}:
        raise TargetForecastError("Target models must be lightgbm or catboost.")
    return TargetForecaster(model_name, target, **kwargs)  # type: ignore[arg-type]
