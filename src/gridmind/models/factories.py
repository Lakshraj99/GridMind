"""Model construction from validated configuration and tuned parameters."""

from __future__ import annotations

from typing import Any

from gridmind.features.builder import FeatureBuilder
from gridmind.features.contracts import FeatureSpecification
from gridmind.models.catboost_model import CatBoostGlobalForecaster
from gridmind.models.lightgbm_model import LightGBMGlobalForecaster
from gridmind.models.protocols import TrainableForecastModel

SUPPORTED_MODELS = ("lightgbm", "catboost")


def create_model(
    name: str,
    *,
    specification: FeatureSpecification | None = None,
    builder: FeatureBuilder | None = None,
    random_seed: int = 42,
    n_jobs: int = -1,
    params: dict[str, Any] | None = None,
) -> TrainableForecastModel:
    """Create a supported global forecaster with deterministic settings."""
    normalized = name.lower().strip()
    if normalized == "lightgbm":
        return LightGBMGlobalForecaster(
            specification=specification,
            builder=builder,
            random_seed=random_seed,
            n_jobs=n_jobs,
            params=params,
        )
    if normalized == "catboost":
        return CatBoostGlobalForecaster(
            specification=specification,
            builder=builder,
            random_seed=random_seed,
            n_jobs=n_jobs,
            params=params,
        )
    raise ValueError(f"Unsupported model '{name}'. Choose from {', '.join(SUPPORTED_MODELS)}.")
