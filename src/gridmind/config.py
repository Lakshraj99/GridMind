"""Environment-backed configuration for GridMind."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

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
    weather_provider: Literal["open_meteo"] = Field(default="open_meteo", alias="WEATHER_PROVIDER")
    weather_base_url: str = Field(
        default="https://archive-api.open-meteo.com/v1/archive", alias="WEATHER_BASE_URL"
    )
    weather_forecast_base_url: str = Field(
        default="https://api.open-meteo.com/v1/forecast", alias="WEATHER_FORECAST_BASE_URL"
    )
    weather_timezone: Literal["UTC"] = Field(default="UTC", alias="WEATHER_TIMEZONE")
    weather_cache_dir: Path = Field(default=Path("data/weather"), alias="WEATHER_CACHE_DIR")
    weather_request_timeout_seconds: float = Field(
        default=30.0, gt=0, alias="WEATHER_REQUEST_TIMEOUT_SECONDS"
    )
    weather_max_retries: int = Field(default=3, ge=0, alias="WEATHER_MAX_RETRIES")
    grid_location_config: Path = Field(
        default=Path("configs/grid_locations.yaml"), alias="GRID_LOCATION_CONFIG"
    )
    weather_features_enabled: bool = Field(default=True, alias="WEATHER_FEATURES_ENABLED")
    weather_lags: Annotated[tuple[int, ...], NoDecode] = Field(
        default=(1, 3, 6, 12, 24), alias="WEATHER_LAGS"
    )
    weather_rolling_windows: Annotated[tuple[int, ...], NoDecode] = Field(
        default=(3, 6, 12, 24), alias="WEATHER_ROLLING_WINDOWS"
    )
    renewable_targets: Annotated[tuple[str, ...], NoDecode] = Field(
        default=("solar_generation_mw", "wind_generation_mw"), alias="RENEWABLE_TARGETS"
    )
    net_load_enabled: bool = Field(default=True, alias="NET_LOAD_ENABLED")
    demand_weather_model_name: str = Field(
        default="gridmind-weather-demand-forecast", alias="DEMAND_WEATHER_MODEL_NAME"
    )
    solar_model_name: str = Field(
        default="gridmind-solar-generation-forecast", alias="SOLAR_MODEL_NAME"
    )
    wind_model_name: str = Field(
        default="gridmind-wind-generation-forecast", alias="WIND_MODEL_NAME"
    )
    total_renewable_model_name: str = Field(
        default="gridmind-total-renewable-generation-forecast",
        alias="TOTAL_RENEWABLE_MODEL_NAME",
    )
    net_load_model_name: str = Field(
        default="gridmind-net-load-forecast", alias="NET_LOAD_MODEL_NAME"
    )
    anomaly_detection_enabled: bool = Field(default=True, alias="ANOMALY_DETECTION_ENABLED")
    anomaly_lookback_hours: int = Field(default=720, gt=0, alias="ANOMALY_LOOKBACK_HOURS")
    anomaly_min_training_rows: int = Field(default=336, gt=0, alias="ANOMALY_MIN_TRAINING_ROWS")
    anomaly_contamination: float = Field(
        default=0.01, gt=0.0, lt=0.5, alias="ANOMALY_CONTAMINATION"
    )
    anomaly_random_seed: int = Field(default=42, alias="ANOMALY_RANDOM_SEED")
    residual_zscore_warning: float = Field(default=2.5, gt=0.0, alias="RESIDUAL_ZSCORE_WARNING")
    residual_zscore_critical: float = Field(default=4.0, gt=0.0, alias="RESIDUAL_ZSCORE_CRITICAL")
    residual_mad_warning: float = Field(default=3.5, gt=0.0, alias="RESIDUAL_MAD_WARNING")
    residual_mad_critical: float = Field(default=6.0, gt=0.0, alias="RESIDUAL_MAD_CRITICAL")
    demand_spike_pct_threshold: float = Field(
        default=0.20, ge=0.0, alias="DEMAND_SPIKE_PCT_THRESHOLD"
    )
    renewable_drop_pct_threshold: float = Field(
        default=0.30, ge=0.0, alias="RENEWABLE_DROP_PCT_THRESHOLD"
    )
    flatline_hours: int = Field(default=4, gt=0, alias="FLATLINE_HOURS")
    missing_hour_warning_count: int = Field(default=1, gt=0, alias="MISSING_HOUR_WARNING_COUNT")
    missing_hour_critical_count: int = Field(default=3, gt=0, alias="MISSING_HOUR_CRITICAL_COUNT")
    alert_dedup_window_hours: int = Field(default=6, gt=0, alias="ALERT_DEDUP_WINDOW_HOURS")
    alert_auto_resolve_hours: int = Field(default=24, gt=0, alias="ALERT_AUTO_RESOLVE_HOURS")
    anomaly_experiment_name: str = Field(
        default="gridmind-anomaly-detection", alias="ANOMALY_EXPERIMENT_NAME"
    )

    @field_validator("weather_lags", "weather_rolling_windows", mode="before")
    @classmethod
    def _parse_positive_integer_list(cls, value: object) -> tuple[int, ...]:
        values: tuple[object, ...]
        if isinstance(value, str):
            values = tuple(value.split(","))
        elif isinstance(value, (list, tuple, set)):
            values = tuple(value)
        else:
            raise ValueError("Expected a comma-separated list of positive integers.")
        try:
            parsed = tuple(sorted({int(str(item)) for item in values}))
        except (TypeError, ValueError) as exc:
            raise ValueError("Expected a comma-separated list of positive integers.") from exc
        if not parsed or any(item <= 0 for item in parsed):
            raise ValueError("Configured lag and rolling lists must contain positive integers.")
        return parsed

    @field_validator("renewable_targets", mode="before")
    @classmethod
    def _parse_renewable_targets(cls, value: object) -> tuple[str, ...]:
        values: tuple[object, ...]
        if isinstance(value, str):
            values = tuple(value.split(","))
        elif isinstance(value, (list, tuple, set)):
            values = tuple(value)
        else:
            raise ValueError("RENEWABLE_TARGETS must be a comma-separated list.")
        parsed = tuple(str(item).strip() for item in values if str(item).strip())
        allowed = {"solar_generation_mw", "wind_generation_mw"}
        if not parsed or not set(parsed).issubset(allowed):
            raise ValueError(f"RENEWABLE_TARGETS must contain only {sorted(allowed)}.")
        return parsed

    @model_validator(mode="after")
    def _validate_anomaly_threshold_order(self) -> Settings:
        if self.residual_zscore_critical <= self.residual_zscore_warning:
            raise ValueError("RESIDUAL_ZSCORE_CRITICAL must exceed its warning threshold.")
        if self.residual_mad_critical <= self.residual_mad_warning:
            raise ValueError("RESIDUAL_MAD_CRITICAL must exceed its warning threshold.")
        if self.missing_hour_critical_count <= self.missing_hour_warning_count:
            raise ValueError("MISSING_HOUR_CRITICAL_COUNT must exceed its warning count.")
        return self

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
