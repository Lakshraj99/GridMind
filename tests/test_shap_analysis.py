"""Small deterministic SHAP artifact creation tests."""

from pathlib import Path

import pandas as pd

from gridmind.explainability.shap_analysis import generate_shap_artifacts
from gridmind.features.contracts import FeatureSpecification
from gridmind.models.lightgbm_model import LightGBMGlobalForecaster


def test_shap_artifacts_and_sample_limit(tmp_path: Path, hourly_frame: pd.DataFrame) -> None:
    specification = FeatureSpecification.create(lags=(1, 24), rolling_windows=(3, 24))
    model = LightGBMGlobalForecaster(
        specification=specification,
        n_jobs=1,
        params={"n_estimators": 20},
    ).fit(hourly_frame.iloc[:100])
    artifacts = generate_shap_artifacts(
        model,
        hourly_frame.iloc[:100],
        tmp_path,
        sample_size=12,
        top_features=1,
        local_rows=2,
    )
    assert artifacts.sample_rows == 12
    assert artifacts.importance_csv.exists()
    assert artifacts.summary_plot.exists()
    assert len(artifacts.dependence_plots) == 1
    assert artifacts.local_explanations.exists()
    importance = pd.read_csv(artifacts.importance_csv)
    assert set(importance.columns) == {"feature", "mean_absolute_shap"}
