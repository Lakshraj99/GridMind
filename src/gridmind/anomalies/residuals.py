"""Leakage-safe forecast-residual anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gridmind.anomalies.contracts import empty_anomaly_frame, make_anomaly, validate_anomaly_frame
from gridmind.anomalies.severity import severity_from_score


@dataclass(frozen=True)
class ResidualConfig:
    min_history: int = 24
    window: int = 168
    zscore_warning: float = 2.5
    zscore_critical: float = 4.0
    mad_warning: float = 3.5
    mad_critical: float = 6.0
    solar_daylight_start: int = 6
    solar_daylight_end: int = 18


@dataclass(frozen=True)
class ResidualDetectionResult:
    anomalies: pd.DataFrame
    scored_rows: pd.DataFrame
    insufficient_history_rows: int


def align_actuals_and_forecasts(
    actuals: pd.DataFrame,
    forecasts: pd.DataFrame,
    *,
    target: str,
    model_version: str | None = None,
) -> pd.DataFrame:
    """Select the latest forecast origin strictly before each actual timestamp."""
    actual = actuals[["region", "timestamp_utc", target]].copy()
    actual["timestamp_utc"] = pd.to_datetime(actual["timestamp_utc"], utc=True, errors="raise")
    predicted = forecasts.copy()
    if predicted.empty:
        return pd.DataFrame()
    predicted = predicted.loc[predicted["target"] == target].copy()
    if model_version is not None:
        predicted = predicted.loc[predicted["model_version"].astype(str) == model_version]
    for column in ("timestamp_utc", "forecast_origin"):
        predicted[column] = pd.to_datetime(predicted[column], utc=True, errors="raise")
    predicted = predicted.loc[predicted["forecast_origin"] < predicted["timestamp_utc"]]
    merged = actual.merge(predicted, on=["region", "timestamp_utc"], how="inner")
    merged = merged.loc[merged["forecast_origin"] < merged["timestamp_utc"]]
    merged = merged.dropna(subset=[target, "predicted_value"])
    if merged.empty:
        return merged
    keys = ["region", "timestamp_utc"]
    latest = merged.groupby(keys, observed=True)["forecast_origin"].transform("max")
    return (
        merged.loc[merged["forecast_origin"] == latest]
        .sort_values(["region", "timestamp_utc", "model_name", "model_version"])
        .drop_duplicates(keys, keep="last")
        .reset_index(drop=True)
    )


class ResidualDetector:
    name = "residual"
    version = "1"

    def __init__(self, config: ResidualConfig | None = None) -> None:
        self.config = config or ResidualConfig()

    def detect(
        self,
        actuals: pd.DataFrame,
        forecasts: pd.DataFrame,
        *,
        target: str,
        model_version: str | None = None,
    ) -> ResidualDetectionResult:
        aligned = align_actuals_and_forecasts(
            actuals, forecasts, target=target, model_version=model_version
        )
        if aligned.empty:
            return ResidualDetectionResult(empty_anomaly_frame(), aligned, 0)
        aligned["residual"] = aligned[target] - aligned["predicted_value"]
        aligned["absolute_residual"] = aligned["residual"].abs()
        denominator = aligned["predicted_value"].abs().replace(0, np.nan)
        aligned["percentage_residual"] = aligned["residual"] / denominator
        scored: list[pd.DataFrame] = []
        insufficient = 0
        events: list[dict[str, object]] = []
        for region, group in aligned.groupby("region", sort=True, observed=True):
            result, region_events, region_insufficient = self._score_region(
                group.sort_values("timestamp_utc").reset_index(drop=True), str(region), target
            )
            scored.append(result)
            events.extend(region_events)
            insufficient += region_insufficient
        anomalies = (
            validate_anomaly_frame(pd.DataFrame(events)) if events else empty_anomaly_frame()
        )
        return ResidualDetectionResult(
            anomalies, pd.concat(scored, ignore_index=True), insufficient
        )

    def _score_region(
        self, group: pd.DataFrame, region: str, target: str
    ) -> tuple[pd.DataFrame, list[dict[str, object]], int]:
        residuals = group["residual"].astype(float)
        means: list[float] = []
        stds: list[float] = []
        medians: list[float] = []
        mads: list[float] = []
        zscores: list[float] = []
        mad_scores: list[float] = []
        events: list[dict[str, object]] = []
        insufficient = 0
        for index, residual in enumerate(residuals):
            prior = residuals.iloc[max(0, index - self.config.window) : index]
            mean = float(prior.mean()) if len(prior) else np.nan
            std = float(prior.std(ddof=1)) if len(prior) > 1 else np.nan
            median = float(prior.median()) if len(prior) else np.nan
            mad = float((prior - median).abs().median()) if len(prior) else np.nan
            zscore = self._standard_score(float(residual), mean, std)
            mad_score = self._robust_score(float(residual), median, mad)
            means.append(mean)
            stds.append(std)
            medians.append(median)
            mads.append(mad)
            zscores.append(zscore)
            mad_scores.append(mad_score)
            if len(prior) < self.config.min_history:
                insufficient += 1
                continue
            row = group.iloc[index]
            if target == "solar_generation_mw" and not self._is_daylight(row):
                continue
            level = max(
                abs(zscore) / self.config.zscore_warning,
                abs(mad_score) / self.config.mad_warning,
            )
            if level < 1:
                continue
            critical = (
                abs(zscore) >= self.config.zscore_critical
                or abs(mad_score) >= self.config.mad_critical
            )
            score = (
                min(100.0, 70.0 + 10.0 * (level - 1))
                if critical
                else min(69.0, 30 + 20 * (level - 1))
            )
            anomaly_type = self._anomaly_type(target, float(residual))
            events.append(
                make_anomaly(
                    region=region,
                    target=target,
                    timestamp=row["timestamp_utc"],
                    detector_name=self.name,
                    anomaly_type=anomaly_type,
                    anomaly_score=score,
                    severity="critical" if critical else severity_from_score(score),
                    observed_value=float(row[target]),
                    expected_value=float(row["predicted_value"]),
                    residual=float(residual),
                    threshold=self.config.zscore_critical
                    if critical
                    else self.config.zscore_warning,
                    forecast_origin=row["forecast_origin"],
                    model_name=str(row["model_name"]),
                    model_version=str(row["model_version"]),
                    run_id=str(row["run_id"]),
                    feature_summary={
                        "prior_residual_mean": mean,
                        "prior_residual_std": std,
                        "prior_residual_median": median,
                        "prior_residual_mad": mad,
                        "zscore": zscore,
                        "mad_score": mad_score,
                    },
                    explanation=(
                        f"Residual {residual:.3f} has prior-only z-score {zscore:.2f} "
                        f"and MAD score {mad_score:.2f}."
                    ),
                    metadata={"history_rows": len(prior), "daylight": self._is_daylight(row)},
                )
            )
        group["rolling_residual_mean"] = means
        group["rolling_residual_std"] = stds
        group["prior_residual_median"] = medians
        group["prior_residual_mad"] = mads
        group["residual_zscore"] = zscores
        group["residual_mad_score"] = mad_scores
        return group, events, insufficient

    def _is_daylight(self, row: pd.Series) -> bool:
        if "shortwave_radiation_wm2" in row.index and pd.notna(row["shortwave_radiation_wm2"]):
            return float(row["shortwave_radiation_wm2"]) > 0
        hour = pd.Timestamp(row["timestamp_utc"]).hour
        return self.config.solar_daylight_start <= hour <= self.config.solar_daylight_end

    @staticmethod
    def _standard_score(value: float, center: float, scale: float) -> float:
        if not np.isfinite(scale) or scale <= 0:
            return 0.0 if value == center else float(np.sign(value - center) * np.inf)
        return (value - center) / scale

    @staticmethod
    def _robust_score(value: float, median: float, mad: float) -> float:
        if not np.isfinite(mad) or mad <= 0:
            return 0.0 if value == median else float(np.sign(value - median) * np.inf)
        return 0.6745 * (value - median) / mad

    @staticmethod
    def _anomaly_type(target: str, residual: float) -> str:
        if target == "demand_mw":
            return "unexpected_demand_spike" if residual > 0 else "unexpected_demand_drop"
        if target == "solar_generation_mw":
            return "solar_underproduction"
        if target == "wind_generation_mw":
            return "wind_generation_drop"
        if target == "net_load_mw":
            return "abnormal_net_load"
        return "forecast_residual"
