"""Typed interface shared by trainable global forecasting models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from gridmind.features.contracts import FeatureSpecification


@runtime_checkable
class TrainableForecastModel(Protocol):
    """Minimal interface needed by training, evaluation, SHAP, and prediction pipelines."""

    name: str
    model_version: str
    run_id: str
    specification: FeatureSpecification

    def fit(
        self, frame: pd.DataFrame, validation_frame: pd.DataFrame | None = None
    ) -> TrainableForecastModel:
        """Fit a global model across every region in the supplied history."""

    def predict(self, history: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
        """Recursively forecast every region without accessing future actual demand."""

    def get_params(self) -> dict[str, Any]:
        """Return reproducibility-relevant estimator parameters."""

    def feature_names(self) -> list[str]:
        """Return model inputs in fitted order."""

    def prepare_feature_matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Convert contract features into estimator-ready dtypes."""

    def training_features(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Build an explainability-ready feature matrix and target."""

    @property
    def estimator(self) -> Any:
        """Return the fitted tree estimator for explainability."""

    def save(self, path: Path, metadata: dict[str, Any] | None = None) -> Path:
        """Serialize a complete batch-prediction bundle."""
