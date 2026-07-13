"""Global tree model, protocol, determinism, and serialization tests."""

from pathlib import Path

import pandas as pd
import pytest

from gridmind.features.contracts import FeatureSpecification
from gridmind.models.catboost_model import CatBoostGlobalForecaster
from gridmind.models.factories import create_model
from gridmind.models.lightgbm_model import LightGBMGlobalForecaster
from gridmind.models.protocols import TrainableForecastModel
from gridmind.models.serialization import load_model_bundle


@pytest.fixture()
def model_frame(ml_hourly_frame: pd.DataFrame) -> pd.DataFrame:
    return ml_hourly_frame.groupby("region", observed=True).head(100).copy()


@pytest.fixture()
def model_specification() -> FeatureSpecification:
    return FeatureSpecification.create(lags=(1, 2, 24), rolling_windows=(3, 24))


@pytest.mark.parametrize("model_name", ["lightgbm", "catboost"])
def test_global_models_fit_predict_and_follow_protocol(
    model_name: str,
    model_frame: pd.DataFrame,
    model_specification: FeatureSpecification,
) -> None:
    params = {"n_estimators": 20} if model_name == "lightgbm" else {"iterations": 20}
    model = create_model(
        model_name,
        specification=model_specification,
        random_seed=7,
        n_jobs=1,
        params=params,
    )
    assert isinstance(model, TrainableForecastModel)
    model.fit(model_frame)
    predictions = model.predict(model_frame, horizon=4)
    assert len(predictions) == 8
    assert predictions.groupby("region").size().tolist() == [4, 4]
    assert predictions["forecast_origin"].max() < predictions["timestamp_utc"].min()
    assert model.feature_names() == list(model_specification.feature_names)


def test_lightgbm_is_deterministic(
    model_frame: pd.DataFrame, model_specification: FeatureSpecification
) -> None:
    outputs = []
    for _ in range(2):
        model = LightGBMGlobalForecaster(
            specification=model_specification,
            random_seed=11,
            n_jobs=1,
            params={"n_estimators": 20},
        ).fit(model_frame)
        outputs.append(model.predict(model_frame, horizon=3)["predicted_demand_mw"])
    pd.testing.assert_series_equal(outputs[0], outputs[1])


def test_model_bundle_round_trip(
    tmp_path: Path,
    model_frame: pd.DataFrame,
    model_specification: FeatureSpecification,
) -> None:
    model = CatBoostGlobalForecaster(
        specification=model_specification,
        n_jobs=1,
        params={"iterations": 15},
    ).fit(model_frame)
    before = model.predict(model_frame, horizon=2)
    path = model.save(
        tmp_path / "bundle" / "model_bundle.joblib",
        metadata={"regions": ["MISO", "PJM"]},
    )
    loaded = load_model_bundle(path)
    after = loaded.model.predict(model_frame, horizon=2)
    pd.testing.assert_series_equal(before["predicted_demand_mw"], after["predicted_demand_mw"])
    assert (path.parent / "feature_schema.json").exists()
    assert loaded.metadata["regions"] == ["MISO", "PJM"]


def test_invalid_model_and_unfitted_prediction(
    model_frame: pd.DataFrame, model_specification: FeatureSpecification
) -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        create_model("xgboost")
    with pytest.raises(ValueError, match="fitted"):
        LightGBMGlobalForecaster(specification=model_specification).predict(model_frame)
