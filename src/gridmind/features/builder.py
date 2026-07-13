"""Feature builder that preserves entity boundaries and temporal causality."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pandas as pd

from gridmind.continuity import detect_contiguous_segments
from gridmind.exceptions import FeatureEngineeringError, InsufficientHistoryError
from gridmind.features.calendar import add_calendar_features
from gridmind.features.contracts import (
    FeatureBuildReport,
    FeatureSpecification,
)
from gridmind.features.demand import add_demand_features


@dataclass(frozen=True)
class FeatureBuildResult:
    """Model-ready features, their specification, and removal accounting."""

    frame: pd.DataFrame
    specification: FeatureSpecification
    report: FeatureBuildReport


class FeatureBuilder:
    """Build past-only demand features for one or many regions."""

    def __init__(
        self,
        specification: FeatureSpecification | None = None,
        *,
        cache_enabled: bool = False,
    ) -> None:
        self.specification = specification or FeatureSpecification.create()
        self.cache_enabled = cache_enabled
        self._training_cache: dict[str, FeatureBuildResult] = {}

    def build_training(self, frame: pd.DataFrame) -> FeatureBuildResult:
        """Build a training matrix without carrying lags across timestamp gaps."""
        source = self._prepare_source(frame)
        cache_key = self._cache_key(source)
        if self.cache_enabled and cache_key in self._training_cache:
            return self._training_cache[cache_key]
        continuity = detect_contiguous_segments(source)
        outputs: list[pd.DataFrame] = []
        segment_ordinals = {
            segment_id: index
            for _region, region_segments in continuity.segments.groupby(
                "region", sort=True, observed=True
            )
            for index, segment_id in enumerate(region_segments["region_segment_id"], start=1)
        }
        for segment_id, group in continuity.frame.groupby(
            "region_segment_id", sort=True, observed=True
        ):
            featured_segment = group.copy().reset_index(drop=True)
            add_calendar_features(featured_segment)
            add_demand_features(featured_segment, self.specification)
            featured_segment["_gap_affected"] = segment_ordinals[str(segment_id)] > 1
            outputs.append(featured_segment)
        featured = pd.concat(outputs, ignore_index=True)
        required = [*self.specification.feature_names, self.specification.target_name]
        invalid_rows = featured[required].isna().any(axis=1)
        gap_affected_rows = int((featured["_gap_affected"] & invalid_rows).sum())
        feature_missing = featured[list(self.specification.feature_names)].isna().any(axis=1)
        insufficient_history_rows = int(
            (feature_missing & featured[self.specification.target_name].notna()).sum()
        )
        clean = featured.dropna(subset=required).copy()
        clean = clean.drop(columns=["_gap_affected", "region_segment_id"])
        clean["region"] = clean["region"].astype("string")
        clean = clean.sort_values(["timestamp_utc", "region"], ignore_index=True)
        report = FeatureBuildReport(
            source_rows=len(source),
            expanded_rows=len(source),
            output_rows=len(clean),
            removed_rows=len(source) - len(clean),
            timestamp_gap_count=int(continuity.segments["missing_expected_hours_before"].sum()),
            gap_affected_rows=gap_affected_rows,
            insufficient_contiguous_history_rows=insufficient_history_rows,
            missing_target_rows=int(source[self.specification.target_name].isna().sum()),
            required_history=self.specification.required_history,
        )
        if clean.empty:
            raise InsufficientHistoryError(
                "No training rows remain after leakage-safe feature generation; "
                f"at least {self.specification.required_history + 1} contiguous hourly "
                "observations per region are required."
            )
        result = FeatureBuildResult(clean, self.specification, report)
        if self.cache_enabled:
            self._training_cache[cache_key] = result
        return result

    def build_future_row(
        self, history: pd.DataFrame, *, region: str, timestamp: pd.Timestamp
    ) -> pd.DataFrame:
        """Build one recursive forecast row using only values before its timestamp."""
        prepared = self._prepare_source(history)
        entity = prepared.loc[prepared["region"] == region, ["timestamp_utc", "demand_mw"]]
        if entity.empty:
            raise InsufficientHistoryError(f"No demand history is available for region {region}.")
        if timestamp.tzinfo is None:
            raise FeatureEngineeringError("Future forecast timestamps must be timezone-aware UTC.")
        timestamp = timestamp.tz_convert("UTC")
        if entity["timestamp_utc"].max() >= timestamp:
            entity = entity.loc[entity["timestamp_utc"] < timestamp]
        values = entity.set_index("timestamp_utc")["demand_mw"]
        context = pd.date_range(
            end=timestamp - pd.Timedelta(hours=1),
            periods=self.specification.required_history,
            freq="h",
        )
        missing_context = values.reindex(context).isna()
        if missing_context.any():
            raise InsufficientHistoryError(
                f"Region {region} lacks {int(missing_context.sum())} required contiguous "
                f"history hours before {timestamp.isoformat()}; recursive forecasting "
                "cannot cross a timestamp gap."
            )
        row = pd.DataFrame({"timestamp_utc": [timestamp], "region": [region]})
        add_calendar_features(row)
        for lag in self.specification.lags:
            source_timestamp = timestamp - pd.Timedelta(hours=lag)
            row[f"demand_lag_{lag}"] = self._required_value(values, source_timestamp, region)
        for window in self.specification.rolling_windows:
            timestamps = pd.date_range(
                timestamp - pd.Timedelta(hours=window), periods=window, freq="h"
            )
            historical = values.reindex(timestamps)
            if historical.isna().any():
                raise InsufficientHistoryError(
                    f"Region {region} lacks contiguous history for rolling window {window} "
                    f"at {timestamp.isoformat()}."
                )
            row[f"demand_rolling_mean_{window}"] = float(historical.mean())
            row[f"demand_rolling_std_{window}"] = float(historical.std())
            row[f"demand_rolling_min_{window}"] = float(historical.min())
            row[f"demand_rolling_max_{window}"] = float(historical.max())
        return row[["timestamp_utc", *self.specification.feature_names]]

    @staticmethod
    def _prepare_source(frame: pd.DataFrame) -> pd.DataFrame:
        required = {"region", "timestamp_utc", "demand_mw"}
        missing = required.difference(frame.columns)
        if missing:
            raise FeatureEngineeringError(f"Feature source is missing columns: {sorted(missing)}")
        source = frame[["region", "timestamp_utc", "demand_mw"]].copy()
        source["timestamp_utc"] = pd.to_datetime(source["timestamp_utc"], utc=True, errors="raise")
        source["region"] = source["region"].astype("string")
        source["demand_mw"] = pd.to_numeric(source["demand_mw"], errors="raise")
        if source.duplicated(["region", "timestamp_utc"]).any():
            raise FeatureEngineeringError(
                "Feature source contains duplicate region/timestamp keys."
            )
        return source.sort_values(["region", "timestamp_utc"], ignore_index=True)

    @staticmethod
    def _required_value(values: pd.Series, timestamp: pd.Timestamp, region: str) -> float:
        try:
            value = values.loc[timestamp]
        except KeyError as exc:
            raise InsufficientHistoryError(
                f"Region {region} is missing required lag timestamp {timestamp.isoformat()}."
            ) from exc
        if pd.isna(value):
            raise InsufficientHistoryError(
                f"Region {region} has missing demand at required lag {timestamp.isoformat()}."
            )
        return float(value)

    def _cache_key(self, source: pd.DataFrame) -> str:
        hashed = pd.util.hash_pandas_object(
            source[["region", "timestamp_utc", "demand_mw"]], index=False
        )
        digest = hashlib.sha256(hashed.to_numpy().tobytes()).hexdigest()
        return f"{digest}:{self.specification.lags}:{self.specification.rolling_windows}"
