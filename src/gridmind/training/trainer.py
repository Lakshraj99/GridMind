"""Final model fitting helpers."""

from __future__ import annotations

import pandas as pd

from gridmind.models.factories import create_model
from gridmind.models.protocols import TrainableForecastModel


def fit_final_model(
    frame: pd.DataFrame,
    model_name: str,
    *,
    params: dict[str, object] | None = None,
    random_seed: int = 42,
    n_jobs: int = -1,
) -> TrainableForecastModel:
    """Fit the selected global model on all available validated history."""
    model = create_model(
        model_name,
        params=params,
        random_seed=random_seed,
        n_jobs=n_jobs,
    )
    model.fit(frame)
    return model
