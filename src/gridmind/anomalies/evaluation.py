"""Controlled synthetic anomaly injection and labelled backtest metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gridmind.time_utils import to_utc_timestamp

INJECTION_TYPES = (
    "single_hour_demand_spike",
    "multi_hour_demand_drop",
    "solar_generation_collapse",
    "wind_generation_spike",
    "flatline_sensor",
    "missing_hour_sequence",
    "weather_value_corruption",
    "gradual_drift",
    "contextual_unusual_hour",
)


@dataclass(frozen=True)
class InjectionResult:
    frame: pd.DataFrame
    labels: pd.DataFrame


@dataclass(frozen=True)
class EvaluationResult:
    overall_metrics: dict[str, float]
    per_type_metrics: pd.DataFrame
    match_results: pd.DataFrame
    detection_delays: pd.DataFrame


def inject_synthetic_anomalies(
    frame: pd.DataFrame,
    *,
    target: str,
    seed: int = 42,
    anomaly_types: tuple[str, ...] = INJECTION_TYPES,
) -> InjectionResult:
    """Inject deterministic anomalies into a deep copy; original data is never mutated."""
    unsupported = set(anomaly_types).difference(INJECTION_TYPES)
    if unsupported:
        raise ValueError(f"Unsupported injection types: {sorted(unsupported)}")
    injected = frame.copy(deep=True)
    injected["timestamp_utc"] = pd.to_datetime(injected["timestamp_utc"], utc=True, errors="raise")
    injected[target] = pd.to_numeric(injected[target], errors="raise").astype(float)
    injected = injected.sort_values(["region", "timestamp_utc"]).reset_index(drop=True)
    if len(injected) < max(24, len(anomaly_types) * 6):
        raise ValueError("Synthetic injection requires enough history to isolate injected events.")
    generator = np.random.default_rng(seed)
    candidates = np.arange(6, len(injected) - 6)
    chosen = generator.choice(candidates, size=len(anomaly_types), replace=False)
    labels: list[dict[str, object]] = []
    drop_indices: list[int] = []
    for injection_type, index in zip(anomaly_types, sorted(chosen), strict=True):
        start = to_utc_timestamp(injected.at[index, "timestamp_utc"])
        region = str(injected.at[index, "region"])
        end = start
        magnitude = 0.0
        affected_target = target
        if injection_type == "single_hour_demand_spike":
            injected.at[index, target] = _as_float(injected.at[index, target]) * 1.75
            magnitude = 0.75
        elif injection_type == "multi_hour_demand_drop":
            injected.loc[index : index + 2, target] = pd.to_numeric(
                injected.loc[index : index + 2, target]
            ).mul(0.40)
            end = to_utc_timestamp(injected.at[index + 2, "timestamp_utc"])
            magnitude = 0.60
        elif injection_type == "solar_generation_collapse":
            affected_target = "solar_generation_mw"
            column = affected_target if affected_target in injected else target
            injected.loc[index : index + 2, column] = 0.0
            end = to_utc_timestamp(injected.at[index + 2, "timestamp_utc"])
            magnitude = 1.0
        elif injection_type == "wind_generation_spike":
            affected_target = "wind_generation_mw"
            column = affected_target if affected_target in injected else target
            injected.at[index, column] = _as_float(injected.at[index, column]) * 2.0
            magnitude = 1.0
        elif injection_type == "flatline_sensor":
            injected.loc[index : index + 3, target] = injected.at[index - 1, target]
            end = to_utc_timestamp(injected.at[index + 3, "timestamp_utc"])
            magnitude = 1.0
        elif injection_type == "missing_hour_sequence":
            drop_indices.extend([index, index + 1])
            end = to_utc_timestamp(injected.at[index + 1, "timestamp_utc"])
            magnitude = 1.0
        elif injection_type == "weather_value_corruption":
            column = "relative_humidity_pct" if "relative_humidity_pct" in injected else target
            affected_target = column
            injected.at[index, column] = 250.0
            magnitude = 1.0
        elif injection_type == "gradual_drift":
            factors = np.linspace(1.1, 1.6, 5)
            injected.loc[index : index + 4, target] = pd.to_numeric(
                injected.loc[index : index + 4, target]
            ).mul(factors)
            end = to_utc_timestamp(injected.at[index + 4, "timestamp_utc"])
            magnitude = 0.60
        elif injection_type == "contextual_unusual_hour":
            injected.at[index, target] = _as_float(injected.at[index, target]) * 1.9
            magnitude = 0.90
        labels.append(
            {
                "injection_id": f"injected-{seed}-{len(labels):03d}",
                "region": region,
                "target": affected_target,
                "anomaly_type": injection_type,
                "start_utc": start,
                "end_utc": end,
                "magnitude": magnitude,
                "expected_severity": "critical" if magnitude >= 0.7 else "warning",
                "synthetic": True,
            }
        )
    if drop_indices:
        injected = injected.drop(index=sorted(set(drop_indices))).reset_index(drop=True)
    return InjectionResult(injected, pd.DataFrame(labels))


def evaluate_detections(
    detected: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    evaluation_start: object,
    evaluation_end: object,
) -> EvaluationResult:
    """Match detections to injected intervals and compute transparent offline metrics."""
    detections = detected.copy()
    if not detections.empty:
        detections["timestamp_utc"] = pd.to_datetime(detections["timestamp_utc"], utc=True)
    matches: list[dict[str, object]] = []
    matched_ids: set[str] = set()
    matched_anomalies: set[str] = set()
    delays: list[dict[str, object]] = []
    for label in labels.itertuples():
        candidates = (
            detections.loc[
                (detections["region"] == label.region)
                & (detections["timestamp_utc"] >= label.start_utc)
                & (detections["timestamp_utc"] <= label.end_utc)
            ]
            if not detections.empty
            else detections
        )
        detected_flag = not candidates.empty
        anomaly_id = str(candidates.iloc[0]["anomaly_id"]) if detected_flag else ""
        delay = (
            float((candidates.iloc[0]["timestamp_utc"] - label.start_utc) / pd.Timedelta(hours=1))
            if detected_flag
            else np.nan
        )
        matches.append(
            {
                "injection_id": label.injection_id,
                "anomaly_type": label.anomaly_type,
                "detected": detected_flag,
                "matched_anomaly_id": anomaly_id,
                "detection_delay_hours": delay,
            }
        )
        if detected_flag:
            matched_ids.add(str(label.injection_id))
            matched_anomalies.add(anomaly_id)
            delays.append(
                {
                    "injection_id": label.injection_id,
                    "anomaly_type": label.anomaly_type,
                    "detection_delay_hours": delay,
                }
            )
    true_positive = len(matched_ids)
    false_positive = max(len(detections) - len(matched_anomalies), 0)
    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    )
    recall = true_positive / len(labels) if len(labels) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    days = max(
        float(
            (to_utc_timestamp(evaluation_end) - to_utc_timestamp(evaluation_start))
            / pd.Timedelta(days=1)
        ),
        1 / 24,
    )
    match_frame = pd.DataFrame(matches)
    per_type = (
        match_frame.groupby("anomaly_type", observed=True)["detected"]
        .agg(injected="size", detected="sum")
        .reset_index()
    )
    per_type["true_positive"] = per_type["detected"]
    per_type["false_negative"] = per_type["injected"] - per_type["detected"]
    per_type["precision"] = (per_type["true_positive"] > 0).astype(float)
    per_type["recall"] = per_type["detected"] / per_type["injected"]
    per_type["f1"] = (
        2
        * per_type["precision"]
        * per_type["recall"]
        / (per_type["precision"] + per_type["recall"]).replace(0, np.nan)
    )
    per_type["f1"] = per_type["f1"].fillna(0.0)
    overall = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positives_per_day": float(false_positive / days),
        "mean_detection_delay_hours": float(
            np.mean([_as_float(item["detection_delay_hours"]) for item in delays])
        )
        if delays
        else 0.0,
        "severity_accuracy": _severity_accuracy(detections, labels),
    }
    return EvaluationResult(overall, per_type, match_frame, pd.DataFrame(delays))


def _severity_accuracy(detections: pd.DataFrame, labels: pd.DataFrame) -> float:
    if detections.empty or labels.empty:
        return 0.0
    correct = 0
    compared = 0
    for label in labels.itertuples():
        candidates = detections.loc[
            (detections["region"] == label.region)
            & (detections["timestamp_utc"] >= label.start_utc)
            & (detections["timestamp_utc"] <= label.end_utc)
        ]
        if candidates.empty:
            continue
        compared += 1
        correct += int(str(candidates.iloc[0]["severity"]) == label.expected_severity)
    return correct / compared if compared else 0.0


def _as_float(value: object) -> float:
    return float(np.asarray(value, dtype=np.float64).item())
