"""Serializable contracts for model feature generation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_LAGS = (1, 2, 3, 6, 12, 24, 48, 72, 168, 336)
DEFAULT_ROLLING_WINDOWS = (3, 6, 12, 24, 72, 168)
CALENDAR_FEATURES = (
    "hour",
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "month",
    "quarter",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
)
ROLLING_STATISTICS = ("mean", "std", "min", "max")


@dataclass(frozen=True)
class FeatureSpecification:
    """Complete, serializable description of the model input contract."""

    feature_names: tuple[str, ...]
    feature_types: dict[str, str]
    lags: tuple[int, ...]
    rolling_windows: tuple[int, ...]
    calendar_features: tuple[str, ...]
    target_name: str = "demand_mw"
    entity_column: str = "region"
    frequency: str = "h"
    creation_version: str = "2.0"

    @classmethod
    def create(
        cls,
        *,
        lags: tuple[int, ...] = DEFAULT_LAGS,
        rolling_windows: tuple[int, ...] = DEFAULT_ROLLING_WINDOWS,
    ) -> FeatureSpecification:
        """Create the standard Milestone 2 feature specification."""
        if not lags or any(value <= 0 for value in lags):
            raise ValueError("Demand lags must contain positive integers.")
        if not rolling_windows or any(value <= 1 for value in rolling_windows):
            raise ValueError("Rolling windows must contain integers greater than one.")
        names = ["region", *CALENDAR_FEATURES]
        names.extend(f"demand_lag_{lag}" for lag in lags)
        names.extend(
            f"demand_rolling_{stat}_{window}"
            for window in rolling_windows
            for stat in ROLLING_STATISTICS
        )
        feature_types = {name: ("category" if name == "region" else "float64") for name in names}
        return cls(
            feature_names=tuple(names),
            feature_types=feature_types,
            lags=tuple(sorted(set(lags))),
            rolling_windows=tuple(sorted(set(rolling_windows))),
            calendar_features=CALENDAR_FEATURES,
        )

    @property
    def required_history(self) -> int:
        """Return the maximum number of prior hourly observations required."""
        return max((*self.lags, *self.rolling_windows))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)

    def save(self, path: Path) -> Path:
        """Write the feature specification beside a model bundle."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


@dataclass(frozen=True)
class FeatureBuildReport:
    """Data-loss and gap accounting for one feature-build operation."""

    source_rows: int
    expanded_rows: int
    output_rows: int
    removed_rows: int
    timestamp_gap_count: int
    gap_affected_rows: int
    insufficient_contiguous_history_rows: int
    missing_target_rows: int
    required_history: int

    def to_dict(self) -> dict[str, int]:
        """Return a serializable report."""
        return asdict(self)
