"""Explicit deterministic anomaly severity scoring."""

from __future__ import annotations

from typing import Literal

import numpy as np

Severity = Literal["info", "warning", "critical"]


def severity_score(
    *,
    magnitude: float,
    duration_hours: int = 1,
    detector_count: int = 1,
    target_importance: float = 1.0,
    completeness: float = 1.0,
    recurrence: int = 1,
) -> float:
    """Return a documented 0-100 weighted operational severity score."""
    magnitude_component = min(max(magnitude, 0.0), 1.0) * 45.0
    duration_component = min(max(duration_hours, 1) / 6.0, 1.0) * 15.0
    agreement_component = min(max(detector_count, 1) / 3.0, 1.0) * 20.0
    importance_component = min(max(target_importance, 0.0), 1.5) / 1.5 * 10.0
    recurrence_component = min(max(recurrence, 1) / 5.0, 1.0) * 5.0
    completeness_penalty = (1.0 - min(max(completeness, 0.0), 1.0)) * 5.0
    return float(
        np.clip(
            magnitude_component
            + duration_component
            + agreement_component
            + importance_component
            + recurrence_component
            - completeness_penalty,
            0.0,
            100.0,
        )
    )


def severity_from_score(score: float) -> Severity:
    if score >= 70:
        return "critical"
    if score >= 30:
        return "warning"
    return "info"
