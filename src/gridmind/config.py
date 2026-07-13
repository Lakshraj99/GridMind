"""Environment-backed configuration for GridMind."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from gridmind.exceptions import ConfigurationError


class Settings(BaseSettings):
    """GridMind settings loaded from environment variables and an optional .env file."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    eia_api_key: str | None = Field(default=None, alias="EIA_API_KEY", exclude=True, repr=False)
    eia_base_url: str = Field(default="https://api.eia.gov/v2", alias="EIA_BASE_URL")
    grid_region: str = Field(default="PJM", alias="GRID_REGION")
    data_start_date: str | None = Field(default=None, alias="DATA_START_DATE")
    data_end_date: str | None = Field(default=None, alias="DATA_END_DATE")
    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")
    data_quality_dir: Path = Field(default=Path("artifacts/data_quality"), alias="DATA_QUALITY_DIR")
    duckdb_path: Path = Field(default=Path("data/gridmind.duckdb"), alias="DUCKDB_PATH")
    mlflow_tracking_uri: str = Field(default="sqlite:///mlflow.db", alias="MLFLOW_TRACKING_URI")
    mlflow_artifact_root: Path = Field(default=Path("mlartifacts"), alias="MLFLOW_ARTIFACT_ROOT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    missing_demand_policy: Literal["error", "drop"] = Field(
        default="error", alias="MISSING_DEMAND_POLICY"
    )
    mlflow_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("ENABLE_MLFLOW", "MLFLOW_ENABLED"),
    )
    forecast_horizon: int = Field(default=24, gt=0, alias="FORECAST_HORIZON")
    validation_windows: int = Field(default=5, gt=0, alias="VALIDATION_WINDOWS")
    validation_step_size: int = Field(default=24, gt=0, alias="VALIDATION_STEP_SIZE")
    tuning_windows: int = Field(default=4, gt=0, alias="TUNING_WINDOWS")
    optuna_trials: int = Field(default=20, gt=0, alias="OPTUNA_TRIALS")
    optuna_timeout_seconds: int | None = Field(default=None, gt=0, alias="OPTUNA_TIMEOUT_SECONDS")
    primary_selection_metric: Literal["mae", "rmse", "wape", "mase", "bias"] = Field(
        default="wape", alias="PRIMARY_SELECTION_METRIC"
    )
    model_random_seed: int = Field(default=42, alias="MODEL_RANDOM_SEED")
    model_n_jobs: int = Field(default=-1, alias="MODEL_N_JOBS")
    mlflow_experiment_name: str = Field(
        default="gridmind-demand-forecasting", alias="MLFLOW_EXPERIMENT_NAME"
    )
    mlflow_model_name: str = Field(default="gridmind-demand-forecast", alias="MLFLOW_MODEL_NAME")
    mlflow_register_model: bool = Field(default=True, alias="MLFLOW_REGISTER_MODEL")
    model_promotion_threshold: float = Field(default=0.0, ge=0.0, alias="MODEL_PROMOTION_THRESHOLD")
    shap_sample_size: int = Field(default=2000, gt=0, alias="SHAP_SAMPLE_SIZE")

    def require_eia_api_key(self) -> str:
        """Return the API key or raise an actionable configuration error."""
        if not self.eia_api_key:
            raise ConfigurationError(
                "EIA_API_KEY is required for ingestion. Obtain a key from "
                "https://www.eia.gov/opendata/register.php and set it in the environment or .env."
            )
        return self.eia_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance for application use."""
    return Settings()
