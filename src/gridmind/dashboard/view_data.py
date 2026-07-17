"""Pure dashboard data normalization and summarization helpers."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd


def records_frame(
    payload: dict[str, Any] | None,
    *,
    timestamp_columns: tuple[str, ...] = (),
    sort_by: str | None = None,
    ascending: bool = True,
) -> pd.DataFrame:
    """Build a stable frame from optional API records and canonicalize timestamps."""
    items = (payload or {}).get("items", [])
    frame = pd.DataFrame(items if isinstance(items, list) else [])
    for column in timestamp_columns:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce", format="mixed")
    if sort_by and sort_by in frame.columns:
        frame = frame.sort_values(sort_by, ascending=ascending, na_position="last")
    return frame.reset_index(drop=True)


def parse_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value:
        return {}
    try:
        result = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return result if isinstance(result, dict) else {}


def count_values(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    counts = frame[column].fillna("unknown").astype(str).value_counts()
    return {str(key): int(value) for key, value in counts.items()}


def effective_anomaly_rate(frame: pd.DataFrame) -> float | None:
    """Return detections per target-hour in the returned investigation window."""
    if frame.empty or "timestamp_utc" not in frame or "target" not in frame:
        return None
    timestamps = pd.to_datetime(
        frame["timestamp_utc"], utc=True, errors="coerce", format="mixed"
    ).dropna()
    if timestamps.empty:
        return None
    hours = max(1, int((timestamps.max() - timestamps.min()).total_seconds() // 3_600) + 1)
    target_count = max(1, int(frame["target"].nunique()))
    return min(1.0, len(frame) / (hours * target_count))


def available_options(frame: pd.DataFrame, column: str) -> list[str]:
    if frame.empty or column not in frame:
        return []
    return sorted({str(value) for value in frame[column].dropna() if str(value)})
