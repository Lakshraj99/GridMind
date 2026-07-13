"""Global CatBoost demand forecaster."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from catboost import CatBoostError, CatBoostRegressor

from gridmind.exceptions import ModelTrainingError
from gridmind.features.builder import FeatureBuilder
from gridmind.features.contracts import FeatureSpecification
from gridmind.models._shared import recursive_predict


class CatBoostGlobalForecaster:
    """One CPU CatBoost regressor with region as a categorical static feature."""

    name = "catboost_global"

    def __init__(
        self,
        *,
        specification: FeatureSpecification | None = None,
        builder: FeatureBuilder | None = None,
        random_seed: int = 42,
        n_jobs: int = -1,
        params: dict[str, Any] | None = None,
    ) -> None:
        self.specification = specification or (
            builder.specification if builder is not None else FeatureSpecification.create()
        )
        self.builder = builder or FeatureBuilder(self.specification)
        defaults: dict[str, Any] = {
            "iterations": 150,
            "depth": 7,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3.0,
            "random_strength": 1.0,
            "bootstrap_type": "Bernoulli",
            "subsample": 0.9,
            "loss_function": "MAE",
            "random_seed": random_seed,
            "thread_count": n_jobs,
            "verbose": False,
            "allow_writing_files": False,
            "task_type": "CPU",
        }
        defaults.update(params or {})
        self._params = defaults
        self._estimator = CatBoostRegressor(**defaults)
        self._regions: list[str] = []
        self.model_version = "unregistered"
        self.run_id = ""

    @property
    def estimator(self) -> CatBoostRegressor:
        """Return the underlying fitted estimator."""
        return self._estimator

    def fit(
        self, frame: pd.DataFrame, validation_frame: pd.DataFrame | None = None
    ) -> CatBoostGlobalForecaster:
        """Fit globally with optional chronological early-stopping data."""
        built = self.builder.build_training(frame)
        self._regions = sorted(str(value) for value in built.frame["region"].unique())
        x_train = self.prepare_feature_matrix(built.frame[list(self.specification.feature_names)])
        y_train = built.frame[self.specification.target_name]
        fit_kwargs: dict[str, Any] = {"cat_features": ["region"]}
        if validation_frame is not None and not validation_frame.empty:
            combined = pd.concat([frame, validation_frame], ignore_index=True)
            validation_built = self.builder.build_training(combined).frame
            validation_start = validation_frame["timestamp_utc"].min()
            validation_rows = validation_built.loc[
                validation_built["timestamp_utc"] >= validation_start
            ]
            if not validation_rows.empty:
                fit_kwargs.update(
                    {
                        "eval_set": (
                            self.prepare_feature_matrix(
                                validation_rows[list(self.specification.feature_names)]
                            ),
                            validation_rows[self.specification.target_name],
                        ),
                        "early_stopping_rounds": 20,
                        "use_best_model": True,
                    }
                )
        try:
            self._estimator.fit(x_train, y_train, **fit_kwargs)
        except (ValueError, CatBoostError) as exc:
            raise ModelTrainingError(f"CatBoost training failed: {exc}") from exc
        return self

    def predict(self, history: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
        """Generate recursive next-hour predictions for every region."""
        if not self._estimator.is_fitted():
            raise ValueError("CatBoost model must be fitted before prediction.")
        return recursive_predict(
            history=history,
            horizon=horizon,
            builder=self.builder,
            model_name=self.name,
            model_version=self.model_version,
            run_id=self.run_id,
            predict_one=lambda row: self._estimator.predict(
                self.prepare_feature_matrix(row[list(self.specification.feature_names)])
            ),
        )

    def prepare_feature_matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply stable string categorical values and fitted feature order."""
        result = frame[list(self.specification.feature_names)].copy()
        result["region"] = result["region"].astype(str)
        return result

    def training_features(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Return estimator-ready training inputs for explainability."""
        built = self.builder.build_training(frame).frame
        return (
            self.prepare_feature_matrix(built[list(self.specification.feature_names)]),
            built[self.specification.target_name],
        )

    def get_params(self) -> dict[str, Any]:
        """Return configured estimator parameters and best iteration when fitted."""
        result = dict(self._params)
        if self._estimator.is_fitted():
            result["best_iteration"] = self._estimator.get_best_iteration()
        return result

    def feature_names(self) -> list[str]:
        """Return ordered model feature names."""
        return list(self.specification.feature_names)

    def save(self, path: Path, metadata: dict[str, Any] | None = None) -> Path:
        """Serialize a complete model bundle."""
        from gridmind.models.serialization import save_model_bundle

        return save_model_bundle(self, path, metadata=metadata)
