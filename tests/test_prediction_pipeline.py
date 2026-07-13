"""Batch model loading, schema, Parquet, and DuckDB idempotency tests."""

from pathlib import Path

import pandas as pd
import pytest

from gridmind.config import Settings
from gridmind.data.storage import DuckDBStorage
from gridmind.exceptions import PredictionValidationError
from gridmind.features.contracts import FeatureSpecification
from gridmind.models.lightgbm_model import LightGBMGlobalForecaster
from gridmind.models.serialization import load_model_bundle
from gridmind.pipelines.predict import run_prediction_pipeline, validate_batch_predictions


def test_batch_prediction_persists_idempotently(tmp_path: Path, hourly_frame: pd.DataFrame) -> None:
    specification = FeatureSpecification.create(lags=(1, 24), rolling_windows=(3, 24))
    model = LightGBMGlobalForecaster(
        specification=specification,
        n_jobs=1,
        params={"n_estimators": 20},
    ).fit(hourly_frame)
    bundle_path = model.save(
        tmp_path / "bundle" / "model_bundle.joblib", metadata={"regions": ["PJM"]}
    )
    bundle = load_model_bundle(bundle_path)
    database = tmp_path / "grid.duckdb"
    DuckDBStorage(database).upsert(hourly_frame)
    settings = Settings(
        DUCKDB_PATH=database,
        DATA_DIR=tmp_path / "data",
        MLFLOW_ENABLED=False,
        _env_file=None,
    )
    first = run_prediction_pipeline(
        settings,
        region="PJM",
        horizon=4,
        bundle=bundle,
        output_dir=tmp_path / "predictions",
    )
    second = run_prediction_pipeline(
        settings,
        region="PJM",
        horizon=4,
        bundle=bundle,
        output_dir=tmp_path / "predictions",
    )
    assert first.parquet_path.exists()
    assert first.metadata_path.exists()
    assert first.duckdb_rows == second.duckdb_rows == 4
    stored = DuckDBStorage(database).read_forecasts("PJM")
    assert len(stored) == 4
    assert stored["forecast_step"].tolist() == [1, 2, 3, 4]


@pytest.mark.parametrize("value", [-0.1, float("nan")])
def test_batch_prediction_rejects_invalid_values(value: float) -> None:
    now = pd.Timestamp("2024-01-01", tz="UTC")
    frame = pd.DataFrame(
        {
            "region": ["PJM"],
            "forecast_origin": [now],
            "timestamp_utc": [now + pd.Timedelta(hours=1)],
            "forecast_step": [1],
            "predicted_demand_mw": [value],
            "model_name": ["test"],
            "model_version": ["1"],
            "run_id": ["run"],
            "created_at_utc": [now],
        }
    )
    with pytest.raises(PredictionValidationError):
        validate_batch_predictions(frame)


def test_batch_rejects_unsupported_region(tmp_path: Path, hourly_frame: pd.DataFrame) -> None:
    specification = FeatureSpecification.create(lags=(1,), rolling_windows=(3,))
    model = LightGBMGlobalForecaster(
        specification=specification, n_jobs=1, params={"n_estimators": 5}
    ).fit(hourly_frame.iloc[:20])
    bundle_path = model.save(tmp_path / "model.joblib", metadata={"regions": ["PJM"]})
    settings = Settings(DUCKDB_PATH=tmp_path / "empty.duckdb", _env_file=None)
    with pytest.raises(PredictionValidationError, match="not present"):
        run_prediction_pipeline(
            settings,
            region="MISO",
            bundle=load_model_bundle(bundle_path),
        )
