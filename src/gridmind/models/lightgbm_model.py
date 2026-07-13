"""Global LightGBM demand forecaster."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightgbm as lgb
import pandas as pd

from gridmind.exceptions import ModelTrainingError
from gridmind.features.builder import FeatureBuilder
from gridmind.features.contracts import FeatureSpecification
from gridmind.models._shared import recursive_predict


class LightGBMGlobalForecaster:
    """One LightGBM regressor trained across every configured region."""

    name = "lightgbm_global"

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
            "objective": "regression_l1",
            "n_estimators": 150,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "random_state": random_seed,
            "n_jobs": n_jobs,
            "verbosity": -1,
        }
        defaults.update(params or {})
        self._params = defaults
        self._estimator = lgb.LGBMRegressor(**defaults)
        self._regions: list[str] = []
        self.model_version = "unregistered"
        self.run_id = ""

    @property
    def estimator(self) -> lgb.LGBMRegressor:
        """Return the underlying fitted estimator."""
        return self._estimator

    def fit(
        self, frame: pd.DataFrame, validation_frame: pd.DataFrame | None = None
    ) -> LightGBMGlobalForecaster:
        """Fit globally; optional chronological validation enables early stopping."""
        built = self.builder.build_training(frame)
        self._regions = sorted(str(value) for value in built.frame["region"].unique())
        features = built.frame[list(self.specification.feature_names)]
        x_train = self.prepare_feature_matrix(features)
        y_train = built.frame[self.specification.target_name]
        fit_kwargs: dict[str, Any] = {"categorical_feature": ["region"]}
        if validation_frame is not None and not validation_frame.empty:
            combined = pd.concat([frame, validation_frame], ignore_index=True)
            validation_built = self.builder.build_training(combined).frame
            validation_start = validation_frame["timestamp_utc"].min()
            validation_rows = validation_built.loc[
                validation_built["timestamp_utc"] >= validation_start
            ]
            if not validation_rows.empty:
                fit_kwargs["eval_set"] = [
                    (
                        self.prepare_feature_matrix(
                            validation_rows[list(self.specification.feature_names)]
                        ),
                        validation_rows[self.specification.target_name],
                    )
                ]
                fit_kwargs["callbacks"] = [lgb.early_stopping(20, verbose=False)]
        try:
            self._estimator.fit(x_train, y_train, **fit_kwargs)
        except (ValueError, lgb.basic.LightGBMError) as exc:
            raise ModelTrainingError(f"LightGBM training failed: {exc}") from exc
        return self

    def predict(self, history: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
        """Generate recursive next-hour predictions for every region."""
        if not hasattr(self._estimator, "fitted_"):
            raise ValueError("LightGBM model must be fitted before prediction.")
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
        """Apply stable categorical region levels and fitted feature order."""
        result = frame[list(self.specification.feature_names)].copy()
        regions = self._regions or sorted(str(value) for value in result["region"].unique())
        result["region"] = pd.Categorical(result["region"], categories=regions)
        return result

    def training_features(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Return estimator-ready training inputs for explainability."""
        built = self.builder.build_training(frame).frame
        return (
            self.prepare_feature_matrix(built[list(self.specification.feature_names)]),
            built[self.specification.target_name],
        )

    def get_params(self) -> dict[str, Any]:
        """Return configured estimator parameters."""
        return dict(self._params)

    def feature_names(self) -> list[str]:
        """Return ordered model feature names."""
        return list(self.specification.feature_names)

    def save(self, path: Path, metadata: dict[str, Any] | None = None) -> Path:
        """Serialize a complete model bundle."""
        from gridmind.models.serialization import save_model_bundle

        return save_model_bundle(self, path, metadata=metadata)
