"""Temporary-registry candidate/champion promotion gate tests."""

from pathlib import Path

import pandas as pd
from mlflow import MlflowClient

from gridmind.features.contracts import FeatureSpecification
from gridmind.models.lightgbm_model import LightGBMGlobalForecaster
from gridmind.models.promotion import (
    apply_promotion_gate,
    effective_registry_uri,
    ensure_registered_model,
)


def test_candidate_champion_and_failed_challenger_preserves_champion(
    tmp_path: Path, hourly_frame: pd.DataFrame
) -> None:
    uri = effective_registry_uri("./legacy-mlruns", tmp_path / "registry.db")
    client = MlflowClient(tracking_uri=uri)
    model_name = "gridmind-test-demand"
    ensure_registered_model(client, model_name)
    specification = FeatureSpecification.create(lags=(1, 24), rolling_windows=(3, 24))
    model = LightGBMGlobalForecaster(
        specification=specification,
        n_jobs=1,
        params={"n_estimators": 10},
    ).fit(hourly_frame.iloc[:100])
    bundle = model.save(tmp_path / "bundle" / "model_bundle.joblib")

    version_one = client.create_model_version(
        model_name, source="file:///tmp/gridmind-model-1", run_id=None
    )
    promoted = apply_promotion_gate(
        client,
        registered_model_name=model_name,
        version=str(version_one.version),
        metrics={"wape": 0.5, "forecast_bias": 1.0},
        reference_metric=1.0,
        primary_metric="wape",
        threshold=0.1,
        bundle_path=bundle,
    )
    assert promoted.candidate_assigned is True
    assert promoted.champion_promoted is True

    version_two = client.create_model_version(
        model_name, source="file:///tmp/gridmind-model-2", run_id=None
    )
    rejected = apply_promotion_gate(
        client,
        registered_model_name=model_name,
        version=str(version_two.version),
        metrics={"wape": 1.2, "forecast_bias": 0.0},
        reference_metric=1.0,
        primary_metric="wape",
        threshold=0.0,
        bundle_path=bundle,
    )
    assert rejected.champion_promoted is False
    assert str(client.get_model_version_by_alias(model_name, "candidate").version) == "2"
    assert str(client.get_model_version_by_alias(model_name, "champion").version) == "1"


def test_nonfinite_metrics_do_not_promote(tmp_path: Path, hourly_frame: pd.DataFrame) -> None:
    uri = effective_registry_uri("local", tmp_path / "registry.db")
    client = MlflowClient(tracking_uri=uri)
    ensure_registered_model(client, "gridmind-nonfinite")
    version = client.create_model_version(
        "gridmind-nonfinite", source="file:///tmp/gridmind-model", run_id=None
    )
    specification = FeatureSpecification.create(lags=(1,), rolling_windows=(3,))
    model = LightGBMGlobalForecaster(
        specification=specification, n_jobs=1, params={"n_estimators": 5}
    ).fit(hourly_frame.iloc[:20])
    bundle = model.save(tmp_path / "model.joblib")
    decision = apply_promotion_gate(
        client,
        registered_model_name="gridmind-nonfinite",
        version=str(version.version),
        metrics={"wape": float("nan"), "forecast_bias": 0.0},
        reference_metric=1.0,
        primary_metric="wape",
        threshold=0.0,
        bundle_path=bundle,
    )
    assert decision.candidate_assigned
    assert not decision.champion_promoted
