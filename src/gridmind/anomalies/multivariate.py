"""Chronological, per-region IsolationForest anomaly detector."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from gridmind.anomalies.contracts import empty_anomaly_frame, make_anomaly, validate_anomaly_frame
from gridmind.exceptions import AnomalyDetectionError


@dataclass(frozen=True)
class IsolationForestConfig:
    target: str = "demand_mw"
    contamination: float = 0.01
    random_seed: int = 42
    min_training_rows: int = 336
    n_estimators: int = 100
    score_quantile: float = 0.995
    extreme_score_quantile: float = 0.9995
    maximum_anomaly_rate: float = 0.10


@dataclass(frozen=True)
class MultivariateResult:
    anomalies: pd.DataFrame
    scored_rows: pd.DataFrame
    excluded_rows: int
    training_rows: int
    gap_count: int
    calibration: dict[str, Any]
    calibration_warning: str | None


class MultivariateDetector:
    """Fit one deterministic detector per region using chronological history only."""

    name = "isolation_forest"
    version = "1"

    def __init__(
        self, feature_names: tuple[str, ...], config: IsolationForestConfig | None = None
    ) -> None:
        if not feature_names:
            raise ValueError("At least one multivariate feature is required.")
        self.feature_names = feature_names
        self.config = config or IsolationForestConfig()
        self.models: dict[str, IsolationForest] = {}
        self.scalers: dict[str, StandardScaler] = {}
        self.reference: dict[str, dict[str, tuple[float, float]]] = {}
        self.training_end: dict[str, pd.Timestamp] = {}
        self.training_row_counts: dict[str, int] = {}
        self.training_scores: dict[str, np.ndarray[Any, np.dtype[np.float64]]] = {}
        self.score_thresholds: dict[str, float] = {}
        self.extreme_thresholds: dict[str, float] = {}

    def fit(self, history: pd.DataFrame) -> MultivariateDetector:
        source = self._prepare(history)
        for region, group in source.groupby("region", sort=True, observed=True):
            clean = group.dropna(subset=list(self.feature_names)).sort_values("timestamp_utc")
            if len(clean) < self.config.min_training_rows:
                raise AnomalyDetectionError(
                    f"IsolationForest requires {self.config.min_training_rows} complete training "
                    f"rows for {region}; received {len(clean)}."
                )
            x = clean[list(self.feature_names)].to_numpy(dtype=float)
            scaler = StandardScaler().fit(x)
            model = IsolationForest(
                contamination=self.config.contamination,
                random_state=self.config.random_seed,
                n_estimators=self.config.n_estimators,
                n_jobs=1,
            ).fit(scaler.transform(x))
            key = str(region)
            self.models[key] = model
            self.scalers[key] = scaler
            self.reference[key] = {
                feature: (float(clean[feature].mean()), float(clean[feature].std(ddof=0)))
                for feature in self.feature_names
            }
            self.training_end[key] = pd.Timestamp(clean["timestamp_utc"].max())
            self.training_row_counts[key] = len(clean)
            training_scores = -model.score_samples(scaler.transform(x))
            self.training_scores[key] = np.asarray(training_scores, dtype=np.float64)
            self.score_thresholds[key] = float(
                np.quantile(training_scores, self.config.score_quantile)
            )
            self.extreme_thresholds[key] = float(
                np.quantile(training_scores, self.config.extreme_score_quantile)
            )
        return self

    def score(self, frame: pd.DataFrame) -> MultivariateResult:
        if not self.models:
            raise AnomalyDetectionError("IsolationForest must be fit before scoring.")
        source = self._prepare(frame)
        gap_count = int(
            source.groupby("region", observed=True)["timestamp_utc"]
            .diff()
            .gt(pd.Timedelta(hours=1))
            .sum()
        )
        scored_parts: list[pd.DataFrame] = []
        events: list[dict[str, object]] = []
        all_scoring_scores: list[float] = []
        all_training_scores: list[float] = []
        excluded = 0
        for region, group in source.groupby("region", sort=True, observed=True):
            key = str(region)
            if key not in self.models:
                raise AnomalyDetectionError(f"No fitted IsolationForest exists for region {key}.")
            if (group["timestamp_utc"] <= self.training_end[key]).any():
                raise AnomalyDetectionError(
                    f"IsolationForest scoring rows for {key} must follow its training data."
                )
            complete = group.dropna(subset=list(self.feature_names)).copy()
            excluded += len(group) - len(complete)
            if complete.empty:
                continue
            transformed = self.scalers[key].transform(
                complete[list(self.feature_names)].to_numpy(dtype=float)
            )
            decisions = self.models[key].decision_function(transformed)
            outlier_scores = -self.models[key].score_samples(transformed)
            selected_threshold = self.score_thresholds[key]
            extreme_threshold = self.extreme_thresholds[key]
            predictions = outlier_scores >= selected_threshold
            extremes = outlier_scores >= extreme_threshold
            complete["isolation_decision"] = decisions
            complete["isolation_score"] = outlier_scores
            complete["is_outlier"] = predictions
            complete["is_extreme_outlier"] = extremes
            scored_parts.append(complete)
            all_scoring_scores.extend(float(value) for value in outlier_scores)
            all_training_scores.extend(float(value) for value in self.training_scores[key])
            for position in np.flatnonzero(predictions):
                row = complete.iloc[position]
                is_extreme = bool(extremes[position])
                exceedance = max(
                    0.0,
                    float(outlier_scores[position] - selected_threshold)
                    / max(abs(selected_threshold), 1e-12),
                )
                score = (
                    min(69.0, 50.0 + exceedance * 100.0)
                    if is_extreme
                    else min(29.0, 15.0 + exceedance * 100.0)
                )
                deviations = self._deviations(key, row)
                target = self.config.target
                events.append(
                    make_anomaly(
                        region=key,
                        target=target,
                        timestamp=row["timestamp_utc"],
                        detector_name=self.name,
                        anomaly_type="multivariate_outlier",
                        anomaly_score=score,
                        severity="warning" if is_extreme else "info",
                        observed_value=float(row[target])
                        if target in row and pd.notna(row[target])
                        else None,
                        feature_summary=deviations,
                        explanation=(
                            "IsolationForest flagged a joint feature pattern; deviations are "
                            "associative, not causal."
                        ),
                        metadata={
                            "decision_function": float(decisions[position]),
                            "outlier_score": float(outlier_scores[position]),
                            "selected_score_threshold": selected_threshold,
                            "extreme_score_threshold": extreme_threshold,
                            "score_quantile": self.config.score_quantile,
                            "extreme_score_quantile": self.config.extreme_score_quantile,
                            "standalone_extreme": is_extreme,
                        },
                    )
                )
        anomalies = (
            validate_anomaly_frame(pd.DataFrame(events)) if events else empty_anomaly_frame()
        )
        scored = pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame()
        evaluated_rows = len(scored)
        flagged_rows = len(anomalies)
        effective_rate = flagged_rows / evaluated_rows if evaluated_rows else 0.0
        warning = (
            f"IsolationForest {self.config.target} flagged {effective_rate:.2%}, exceeding "
            f"the configured maximum {self.config.maximum_anomaly_rate:.2%}; review calibration."
            if effective_rate > self.config.maximum_anomaly_rate
            else None
        )
        calibration: dict[str, Any] = {
            "target": self.config.target,
            "configured_contamination": self.config.contamination,
            "score_quantile": self.config.score_quantile,
            "extreme_score_quantile": self.config.extreme_score_quantile,
            "selected_score_thresholds": self.score_thresholds,
            "extreme_score_thresholds": self.extreme_thresholds,
            "fitted_score_distribution": self._distribution(all_training_scores),
            "scoring_score_distribution": self._distribution(all_scoring_scores),
            "evaluated_row_count": evaluated_rows,
            "flagged_row_count": flagged_rows,
            "effective_anomaly_percentage": effective_rate * 100.0,
            "maximum_anomaly_rate": self.config.maximum_anomaly_rate,
            "calibration_warning": warning,
        }
        return MultivariateResult(
            anomalies,
            scored,
            excluded,
            sum(self.training_row_counts.values()),
            gap_count,
            calibration,
            warning,
        )

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        schema = {
            "feature_names": list(self.feature_names),
            "detector_version": self.version,
            "regions": sorted(self.models),
            "target": self.config.target,
            "score_quantile": self.config.score_quantile,
            "extreme_score_quantile": self.config.extreme_score_quantile,
            "selected_score_thresholds": self.score_thresholds,
            "extreme_score_thresholds": self.extreme_thresholds,
            "training_end_utc": {
                key: value.isoformat() for key, value in self.training_end.items()
            },
        }
        path.with_suffix(".schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> MultivariateDetector:
        detector = joblib.load(path)
        if not isinstance(detector, cls):
            raise AnomalyDetectionError("Serialized anomaly detector has an unexpected type.")
        return detector

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        required = {"region", "timestamp_utc", *self.feature_names}
        missing = required.difference(frame.columns)
        if missing:
            raise AnomalyDetectionError(
                f"Multivariate input is missing features: {sorted(missing)}"
            )
        source = frame.copy()
        source["timestamp_utc"] = pd.to_datetime(source["timestamp_utc"], utc=True, errors="raise")
        source = source.sort_values(["region", "timestamp_utc"]).reset_index(drop=True)
        for feature in self.feature_names:
            source[feature] = pd.to_numeric(source[feature], errors="coerce")
        return source

    def _deviations(self, region: str, row: pd.Series) -> dict[str, Any]:
        deviations: list[tuple[str, float]] = []
        for feature in self.feature_names:
            mean, std = self.reference[region][feature]
            value = float(row[feature])
            zscore = (value - mean) / std if std > 0 else 0.0
            deviations.append((feature, float(zscore)))
        deviations.sort(key=lambda item: abs(item[1]), reverse=True)
        return {name: {"standard_deviations": value} for name, value in deviations[:5]}

    @staticmethod
    def _distribution(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {key: None for key in ("min", "p25", "p50", "p75", "p95", "p99", "max")}
        data = np.asarray(values, dtype=float)
        return {
            "min": float(np.min(data)),
            "p25": float(np.quantile(data, 0.25)),
            "p50": float(np.quantile(data, 0.50)),
            "p75": float(np.quantile(data, 0.75)),
            "p95": float(np.quantile(data, 0.95)),
            "p99": float(np.quantile(data, 0.99)),
            "max": float(np.max(data)),
        }
