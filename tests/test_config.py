"""Milestone 2 configuration validation and compatibility tests."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from gridmind.config import Settings


def test_positive_ml_settings_and_metric_are_validated() -> None:
    with pytest.raises(ValidationError):
        Settings(FORECAST_HORIZON=0, _env_file=None)
    with pytest.raises(ValidationError):
        Settings(OPTUNA_TRIALS=-1, _env_file=None)
    with pytest.raises(ValidationError):
        Settings(PRIMARY_SELECTION_METRIC="accuracy", _env_file=None)


def test_mlflow_legacy_and_new_aliases_are_supported() -> None:
    assert Settings(MLFLOW_ENABLED=False, _env_file=None).mlflow_enabled is False
    assert Settings(ENABLE_MLFLOW=False, _env_file=None).mlflow_enabled is False


def test_missing_demand_policy_is_validated() -> None:
    assert Settings(_env_file=None).missing_demand_policy == "error"
    assert Settings(MISSING_DEMAND_POLICY="drop", _env_file=None).missing_demand_policy == "drop"
    with pytest.raises(ValidationError):
        Settings(MISSING_DEMAND_POLICY="fill", _env_file=None)


def test_eia_key_is_excluded_from_settings_representation_and_dump() -> None:
    secret = "never-persist-this-key"
    settings = Settings(EIA_API_KEY=secret, _env_file=None)
    assert secret not in repr(settings)
    assert "eia_api_key" not in settings.model_dump()


def test_data_quality_directory_is_configurable() -> None:
    assert Settings(_env_file=None).data_quality_dir == Path("artifacts/data_quality")
    assert Settings(DATA_QUALITY_DIR="data/reports", _env_file=None).data_quality_dir == Path(
        "data/reports"
    )


def test_mlflow_defaults_to_sqlite_with_separate_artifacts() -> None:
    assert Settings.model_fields["mlflow_tracking_uri"].default == "sqlite:///mlflow.db"
    assert Settings.model_fields["mlflow_artifact_root"].default == Path("mlartifacts")


def test_milestone_three_lists_and_weather_settings_are_validated() -> None:
    settings = Settings(
        WEATHER_LAGS="24,1,3,3",
        WEATHER_ROLLING_WINDOWS="6,3",
        RENEWABLE_TARGETS="solar_generation_mw,wind_generation_mw",
        _env_file=None,
    )
    assert settings.weather_lags == (1, 3, 24)
    assert settings.weather_rolling_windows == (3, 6)
    assert settings.weather_timezone == "UTC"
    assert settings.weather_provider == "open_meteo"
    assert settings.total_renewable_model_name == "gridmind-total-renewable-generation-forecast"
    with pytest.raises(ValidationError):
        Settings(WEATHER_LAGS="1,0", _env_file=None)
    with pytest.raises(ValidationError):
        Settings(RENEWABLE_TARGETS="coal_generation_mw", _env_file=None)
    with pytest.raises(ValidationError):
        Settings(WEATHER_TIMEZONE="America/New_York", _env_file=None)
