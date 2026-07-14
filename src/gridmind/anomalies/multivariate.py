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
    contamination: float = 0.01
    random_seed: int = 42
    min_training_rows: int = 336
    n_estimators: int = 100


@dataclass(frozen=True)
class MultivariateResult:
    anomalies: pd.DataFrame
    scored_rows: pd.DataFrame
    excluded_rows: int
    training_rows: int
    gap_count: int


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
            predictions = self.models[key].predict(transformed)
            complete["isolation_decision"] = decisions
            complete["is_outlier"] = predictions == -1
            scored_parts.append(complete)
            for position in np.flatnonzero(predictions == -1):
                row = complete.iloc[position]
                raw_score = float(max(0.0, -decisions[position]))
                score = min(69.0, 30.0 + raw_score * 200.0)
                deviations = self._deviations(key, row)
                target = self._target_name(row)
                events.append(
                    make_anomaly(
                        region=key,
                        target=target,
                        timestamp=row["timestamp_utc"],
                        detector_name=self.name,
                        anomaly_type="multivariate_outlier",
                        anomaly_score=score,
                        severity="warning" if score >= 30 else "info",
                        observed_value=float(row[target])
                        if target in row and pd.notna(row[target])
                        else None,
                        feature_summary=deviations,
                        explanation=(
                            "IsolationForest flagged a joint feature pattern; deviations are "
                            "associative, not causal."
                        ),
                        metadata={"decision_function": float(decisions[position])},
                    )
                )
        anomalies = (
            validate_anomaly_frame(pd.DataFrame(events)) if events else empty_anomaly_frame()
        )
        scored = pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame()
        return MultivariateResult(
            anomalies,
            scored,
            excluded,
            sum(self.training_row_counts.values()),
            gap_count,
        )

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        schema = {
            "feature_names": list(self.feature_names),
            "detector_version": self.version,
            "regions": sorted(self.models),
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
    def _target_name(row: pd.Series) -> str:
        for target in ("demand_mw", "net_load_mw", "solar_generation_mw", "wind_generation_mw"):
            if target in row.index:
                return target
        return "demand_mw"
