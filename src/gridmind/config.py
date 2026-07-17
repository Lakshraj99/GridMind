"""Environment-backed configuration for GridMind."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlparse

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
    log_format: Literal["json", "text"] = Field(default="json", alias="LOG_FORMAT")
    api_enabled: bool = Field(default=True, alias="API_ENABLED")
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, ge=1, le=65535, alias="API_PORT")
    api_workers: int = Field(default=1, gt=0, alias="API_WORKERS")
    api_title: str = Field(default="GridMind API", min_length=1, alias="API_TITLE")
    api_version: str = Field(default="0.6.0", min_length=1, alias="API_VERSION")
    api_root_path: str = Field(default="", alias="API_ROOT_PATH")
    api_cors_origins: Annotated[tuple[str, ...], NoDecode] = Field(
        default=("http://localhost:8501",), alias="API_CORS_ORIGINS"
    )
    api_key_enabled: bool = Field(default=False, alias="API_KEY_ENABLED")
    gridmind_api_key: str | None = Field(
        default=None, alias="GRIDMIND_API_KEY", exclude=True, repr=False
    )
    api_default_page_size: int = Field(default=50, gt=0, alias="API_DEFAULT_PAGE_SIZE")
    api_max_page_size: int = Field(default=500, gt=0, alias="API_MAX_PAGE_SIZE")
    api_cache_ttl_seconds: float = Field(default=30.0, ge=0, alias="API_CACHE_TTL_SECONDS")
    metrics_enabled: bool = Field(default=True, alias="METRICS_ENABLED")
    dashboard_enabled: bool = Field(default=True, alias="DASHBOARD_ENABLED")
    dashboard_host: str = Field(default="0.0.0.0", alias="DASHBOARD_HOST")
    dashboard_port: int = Field(default=8501, ge=1, le=65535, alias="DASHBOARD_PORT")
    dashboard_api_base_url: str = Field(
        default="http://localhost:8000", alias="DASHBOARD_API_BASE_URL"
    )
    dashboard_request_timeout_seconds: float = Field(
        default=15.0, gt=0, alias="DASHBOARD_REQUEST_TIMEOUT_SECONDS"
    )
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
    flatline_tolerance: float = Field(default=0.0, ge=0.0, alias="FLATLINE_TOLERANCE")
    solar_daylight_radiation_threshold_wm2: float = Field(
        default=25.0, ge=0.0, alias="SOLAR_DAYLIGHT_RADIATION_THRESHOLD_WM2"
    )
    solar_min_expected_generation_mw: float = Field(
        default=100.0, ge=0.0, alias="SOLAR_MIN_EXPECTED_GENERATION_MW"
    )
    solar_min_absolute_drop_mw: float = Field(
        default=100.0, ge=0.0, alias="SOLAR_MIN_ABSOLUTE_DROP_MW"
    )
    solar_min_drop_duration_hours: int = Field(
        default=2, gt=0, alias="SOLAR_MIN_DROP_DURATION_HOURS"
    )
    isolation_demand_score_quantile: float = Field(
        default=0.995, gt=0.5, lt=1.0, alias="ISOLATION_DEMAND_SCORE_QUANTILE"
    )
    isolation_solar_score_quantile: float = Field(
        default=0.995, gt=0.5, lt=1.0, alias="ISOLATION_SOLAR_SCORE_QUANTILE"
    )
    isolation_wind_score_quantile: float = Field(
        default=0.995, gt=0.5, lt=1.0, alias="ISOLATION_WIND_SCORE_QUANTILE"
    )
    isolation_net_load_score_quantile: float = Field(
        default=0.999, gt=0.5, lt=1.0, alias="ISOLATION_NET_LOAD_SCORE_QUANTILE"
    )
    isolation_extreme_score_quantile: float = Field(
        default=0.9995, gt=0.5, lt=1.0, alias="ISOLATION_EXTREME_SCORE_QUANTILE"
    )
    anomaly_max_rate: float = Field(default=0.10, gt=0.0, le=1.0, alias="ANOMALY_MAX_RATE")
    missing_hour_warning_count: int = Field(default=1, gt=0, alias="MISSING_HOUR_WARNING_COUNT")
    missing_hour_critical_count: int = Field(default=3, gt=0, alias="MISSING_HOUR_CRITICAL_COUNT")
    alert_dedup_window_hours: int = Field(default=6, gt=0, alias="ALERT_DEDUP_WINDOW_HOURS")
    alert_auto_resolve_hours: int = Field(default=24, gt=0, alias="ALERT_AUTO_RESOLVE_HOURS")
    anomaly_experiment_name: str = Field(
        default="gridmind-anomaly-detection", alias="ANOMALY_EXPERIMENT_NAME"
    )
    battery_optimization_enabled: bool = Field(default=True, alias="BATTERY_OPTIMIZATION_ENABLED")
    battery_capacity_mwh: float = Field(default=500.0, gt=0.0, alias="BATTERY_CAPACITY_MWH")
    battery_max_charge_mw: float = Field(default=100.0, gt=0.0, alias="BATTERY_MAX_CHARGE_MW")
    battery_max_discharge_mw: float = Field(default=100.0, gt=0.0, alias="BATTERY_MAX_DISCHARGE_MW")
    battery_min_soc_mwh: float = Field(default=50.0, ge=0.0, alias="BATTERY_MIN_SOC_MWH")
    battery_max_soc_mwh: float = Field(default=500.0, gt=0.0, alias="BATTERY_MAX_SOC_MWH")
    battery_initial_soc_mwh: float = Field(default=250.0, ge=0.0, alias="BATTERY_INITIAL_SOC_MWH")
    battery_terminal_soc_mwh: float = Field(default=250.0, ge=0.0, alias="BATTERY_TERMINAL_SOC_MWH")
    battery_charge_efficiency: float = Field(
        default=0.95, gt=0.0, le=1.0, alias="BATTERY_CHARGE_EFFICIENCY"
    )
    battery_discharge_efficiency: float = Field(
        default=0.95, gt=0.0, le=1.0, alias="BATTERY_DISCHARGE_EFFICIENCY"
    )
    battery_self_discharge_per_hour: float = Field(
        default=0.0001, ge=0.0, lt=1.0, alias="BATTERY_SELF_DISCHARGE_PER_HOUR"
    )
    battery_max_equivalent_cycles_per_day: float = Field(
        default=1.5, gt=0.0, alias="BATTERY_MAX_EQUIVALENT_CYCLES_PER_DAY"
    )
    battery_degradation_cost_per_mwh: float = Field(
        default=5.0, ge=0.0, alias="BATTERY_DEGRADATION_COST_PER_MWH"
    )
    battery_reserve_soc_pct: float = Field(
        default=0.10, ge=0.0, lt=1.0, alias="BATTERY_RESERVE_SOC_PCT"
    )
    dispatch_horizon_hours: int = Field(default=24, gt=0, alias="DISPATCH_HORIZON_HOURS")
    dispatch_step_hours: float = Field(default=1.0, gt=0.0, alias="DISPATCH_STEP_HOURS")
    dispatch_solver_timeout_seconds: float = Field(
        default=60.0, gt=0.0, alias="DISPATCH_SOLVER_TIMEOUT_SECONDS"
    )
    peak_shaving_weight: float = Field(default=1.0, ge=0.0, alias="PEAK_SHAVING_WEIGHT")
    energy_cost_weight: float = Field(default=1.0, ge=0.0, alias="ENERGY_COST_WEIGHT")
    renewable_utilization_weight: float = Field(
        default=0.5, ge=0.0, alias="RENEWABLE_UTILIZATION_WEIGHT"
    )
    degradation_weight: float = Field(default=1.0, ge=0.0, alias="DEGRADATION_WEIGHT")
    terminal_soc_penalty_weight: float = Field(
        default=10.0, ge=0.0, alias="TERMINAL_SOC_PENALTY_WEIGHT"
    )
    fallback_energy_price_per_mwh: float | None = Field(
        default=None, ge=0.0, alias="FALLBACK_ENERGY_PRICE_PER_MWH"
    )
    robust_demand_uplift_pct: float = Field(default=0.03, ge=0.0, alias="ROBUST_DEMAND_UPLIFT_PCT")
    robust_renewable_reduction_pct: float = Field(
        default=0.10, ge=0.0, le=1.0, alias="ROBUST_RENEWABLE_REDUCTION_PCT"
    )
    robust_extra_reserve_pct: float = Field(
        default=0.05, ge=0.0, lt=1.0, alias="ROBUST_EXTRA_RESERVE_PCT"
    )
    battery_experiment_name: str = Field(
        default="gridmind-battery-optimization", alias="BATTERY_EXPERIMENT_NAME"
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

    @field_validator("api_cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, value: object) -> tuple[str, ...]:
        if isinstance(value, str):
            origins = tuple(item.strip() for item in value.split(",") if item.strip())
        elif isinstance(value, (list, tuple, set)):
            origins = tuple(str(item).strip() for item in value if str(item).strip())
        else:
            raise ValueError("API_CORS_ORIGINS must be a comma-separated list.")
        if not origins:
            raise ValueError("API_CORS_ORIGINS must contain at least one origin.")
        for origin in origins:
            parsed = urlparse(origin)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or parsed.path
                not in {
                    "",
                    "/",
                }
            ):
                raise ValueError(f"Invalid CORS origin: {origin}")
        return origins

    @model_validator(mode="after")
    def _validate_anomaly_threshold_order(self) -> Settings:
        if self.api_max_page_size < self.api_default_page_size:
            raise ValueError("API_MAX_PAGE_SIZE must be at least API_DEFAULT_PAGE_SIZE.")
        if self.api_key_enabled and not self.gridmind_api_key:
            raise ValueError("GRIDMIND_API_KEY is required when API_KEY_ENABLED=true.")
        if self.residual_zscore_critical <= self.residual_zscore_warning:
            raise ValueError("RESIDUAL_ZSCORE_CRITICAL must exceed its warning threshold.")
        if self.residual_mad_critical <= self.residual_mad_warning:
            raise ValueError("RESIDUAL_MAD_CRITICAL must exceed its warning threshold.")
        if self.missing_hour_critical_count <= self.missing_hour_warning_count:
            raise ValueError("MISSING_HOUR_CRITICAL_COUNT must exceed its warning count.")
        operational_quantiles = (
            self.isolation_demand_score_quantile,
            self.isolation_solar_score_quantile,
            self.isolation_wind_score_quantile,
            self.isolation_net_load_score_quantile,
        )
        if self.isolation_extreme_score_quantile <= max(operational_quantiles):
            raise ValueError(
                "ISOLATION_EXTREME_SCORE_QUANTILE must exceed every target score quantile."
            )
        if self.battery_min_soc_mwh >= self.battery_max_soc_mwh:
            raise ValueError("BATTERY_MIN_SOC_MWH must be below BATTERY_MAX_SOC_MWH.")
        if self.battery_max_soc_mwh > self.battery_capacity_mwh:
            raise ValueError("BATTERY_MAX_SOC_MWH must not exceed BATTERY_CAPACITY_MWH.")
        for name, value in (
            ("BATTERY_INITIAL_SOC_MWH", self.battery_initial_soc_mwh),
            ("BATTERY_TERMINAL_SOC_MWH", self.battery_terminal_soc_mwh),
        ):
            if not self.battery_min_soc_mwh <= value <= self.battery_max_soc_mwh:
                raise ValueError(f"{name} must be within configured SOC bounds.")
        if self.battery_reserve_soc_pct * self.battery_capacity_mwh > self.battery_max_soc_mwh:
            raise ValueError("BATTERY_RESERVE_SOC_PCT exceeds the maximum SOC.")
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
