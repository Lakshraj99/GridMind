"""Canonical UTC timestamp parsing and display helpers."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd


def to_utc_timestamp(value: Any) -> pd.Timestamp:
    """Normalize a timestamp-like value to a timezone-aware UTC timestamp."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def format_utc_timestamp(value: Any) -> str:
    """Render a timestamp in canonical ISO-8601 UTC form using ``Z``."""
    return to_utc_timestamp(value).isoformat().replace("+00:00", "Z")


def inclusive_hourly_range(
    start: str | date | datetime, end: str | date | datetime
) -> pd.DatetimeIndex:
    """Return inclusive UTC hours, treating a date-only end as its full UTC day."""
    start_timestamp = to_utc_timestamp(start).floor("h")
    end_timestamp = to_utc_timestamp(end).floor("h")
    date_object = isinstance(end, date) and not isinstance(end, datetime)
    date_string = isinstance(end, str) and len(end.strip()) == 10
    if date_object or date_string:
        end_timestamp += pd.Timedelta(hours=23)
    return pd.date_range(start_timestamp, end_timestamp, freq="h")
