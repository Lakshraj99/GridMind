"""Domain-specific exceptions raised by GridMind."""


class GridMindError(Exception):
    """Base exception for expected GridMind failures."""


class ConfigurationError(GridMindError):
    """Raised when required configuration is missing or invalid."""


class EIAClientError(GridMindError):
    """Base exception for EIA API communication failures."""


class EIAAuthenticationError(EIAClientError):
    """Raised when the EIA API rejects credentials."""


class EIARateLimitError(EIAClientError):
    """Raised when EIA rate limits remain exhausted after retries."""


class EIANetworkError(EIAClientError):
    """Raised when a network or temporary server error persists."""


class EIAMalformedResponseError(EIAClientError):
    """Raised when an EIA response does not follow the expected shape."""


class DataValidationError(GridMindError):
    """Raised when grid data violates the canonical contract."""


class MissingDemandError(DataValidationError):
    """Raised after missing actual-demand observations have been reported."""


class StorageError(GridMindError):
    """Raised when canonical data cannot be safely read or persisted."""


class ConflictingDuplicateError(DataValidationError):
    """Raised for differing measurements at the same region and timestamp."""


class InsufficientHistoryError(GridMindError):
    """Raised when a baseline model cannot access its required history."""


class FeatureEngineeringError(GridMindError):
    """Raised when leakage-safe forecasting features cannot be constructed."""


class ModelTrainingError(GridMindError):
    """Raised when a trainable forecasting model cannot be fit or evaluated."""


class ModelSerializationError(GridMindError):
    """Raised when a model bundle cannot be saved, loaded, or validated."""


class ModelPromotionError(GridMindError):
    """Raised when a registry candidate operation cannot be completed safely."""


class PredictionValidationError(GridMindError):
    """Raised when batch forecast output violates its contract."""


class ExplainabilityError(GridMindError):
    """Raised when explainability artifacts cannot be generated."""


class WeatherError(GridMindError):
    """Base exception for weather configuration, transport, and validation failures."""


class WeatherClientError(WeatherError):
    """Raised when a weather provider request cannot be completed safely."""


class WeatherLocationError(WeatherError):
    """Raised when a region-to-location mapping is missing or invalid."""


class RenewableDataError(GridMindError):
    """Raised when renewable-generation data cannot be normalized or validated."""


class TargetForecastError(GridMindError):
    """Raised for unsupported or invalid Milestone 3 target workflows."""


class AnomalyDetectionError(GridMindError):
    """Raised when anomaly detection or evaluation cannot complete safely."""


class AlertLifecycleError(GridMindError):
    """Raised when an alert transition violates the lifecycle contract."""


class BatteryOptimizationError(GridMindError):
    """Raised when battery dispatch inputs, solving, or validation fail safely."""
