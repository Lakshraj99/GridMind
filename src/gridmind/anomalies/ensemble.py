"""Transparent rule/model voting and escalation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from gridmind.anomalies.contracts import empty_anomaly_frame, make_anomaly, validate_anomaly_frame
from gridmind.anomalies.severity import severity_from_score, severity_score

SEVERITY_RANK = {"info": 0, "warning": 1, "critical": 2}


def combine_detector_events(events: pd.DataFrame) -> pd.DataFrame:
    """Create one ensemble event per region/target/instant while retaining contributions."""
    if events.empty:
        return empty_anomaly_frame()
    source = validate_anomaly_frame(events)
    combined: list[dict[str, object]] = []
    keys = ["region", "target", "timestamp_utc"]
    for (region, target, timestamp), group in source.groupby(keys, sort=True, observed=True):
        contributions = [
            {
                "detector": str(row.detector_name),
                "anomaly_type": str(row.anomaly_type),
                "score": float(np.asarray(row.anomaly_score).item()),
                "severity": str(row.severity),
                "anomaly_id": str(row.anomaly_id),
            }
            for row in group.itertuples()
        ]
        strongest = group.sort_values(
            ["severity", "anomaly_score"],
            key=lambda values: values.map(SEVERITY_RANK) if values.name == "severity" else values,
        ).iloc[-1]
        detectors = set(group["detector_name"].astype(str))
        critical_override = bool(
            (
                (group["detector_name"].isin(["rules", "residual"]))
                & (group["severity"] == "critical")
            ).any()
        )
        warning_votes = int((group["severity"].isin(["warning", "critical"])).sum())
        score = severity_score(
            magnitude=float(group["anomaly_score"].max()) / 100.0,
            detector_count=len(detectors),
        )
        severity = "critical" if critical_override else severity_from_score(score)
        if warning_votes >= 2 and severity == "info":
            severity = "warning"
            score = max(score, 30.0)
        if detectors == {"isolation_forest"}:
            severity = "warning" if float(group["anomaly_score"].max()) >= 50 else "info"
            score = min(score, 69.0 if severity == "warning" else 29.0)
        combined.append(
            make_anomaly(
                region=str(region),
                target=str(target),
                timestamp=timestamp,
                detector_name="ensemble",
                anomaly_type=str(strongest["anomaly_type"]),
                anomaly_score=score,
                severity=severity,
                observed_value=_optional_float(strongest["observed_value"]),
                expected_value=_optional_float(strongest["expected_value"]),
                residual=_optional_float(strongest["residual"]),
                explanation=(
                    f"Ensemble combined {len(group)} event(s) from "
                    f"{', '.join(sorted(detectors))}; detector disagreement is retained "
                    "in metadata."
                ),
                forecast_origin=strongest["forecast_origin"]
                if pd.notna(strongest["forecast_origin"])
                else None,
                model_name=str(strongest["model_name"]),
                model_version=str(strongest["model_version"]),
                run_id=str(strongest["run_id"]),
                metadata={
                    "contributions": contributions,
                    "critical_override": critical_override,
                    "warning_votes": warning_votes,
                    "detector_agreement": len(detectors),
                    "source_metadata": [json.loads(value) for value in group["metadata_json"]],
                },
            )
        )
    return validate_anomaly_frame(pd.DataFrame(combined))


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    numeric = float(np.asarray(value).item())
    return numeric if np.isfinite(numeric) else None
