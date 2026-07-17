"""Consistent, defensive formatting for dashboard-visible values."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from gridmind.time_utils import to_utc_timestamp

MISSING = "—"

TARGET_LABELS = {
    "demand_mw": "Demand",
    "net_load_mw": "Net load",
    "solar_generation_mw": "Solar generation",
    "wind_generation_mw": "Wind generation",
    "renewable_generation_mw": "Renewable generation",
}


def is_missing(value: Any) -> bool:
    """Return whether a scalar should be presented as unavailable."""
    if value is None or value == "":
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, bool) else False


def number(value: Any, *, decimals: int = 1, compact: bool = False) -> str:
    """Format a finite scalar with optional compact suffixes."""
    if is_missing(value):
        return MISSING
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return MISSING
    if not math.isfinite(numeric):
        return MISSING
    if compact:
        for threshold, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
            if abs(numeric) >= threshold:
                return f"{numeric / threshold:,.{decimals}f}{suffix}"
    return f"{numeric:,.{decimals}f}"


def integer(value: Any) -> str:
    return number(value, decimals=0)


def megawatts(value: Any) -> str:
    formatted = number(value)
    return formatted if formatted == MISSING else f"{formatted} MW"


def megawatt_hours(value: Any) -> str:
    formatted = number(value)
    return formatted if formatted == MISSING else f"{formatted} MWh"


def percentage(value: Any, *, fraction: bool = False, decimals: int = 1) -> str:
    if is_missing(value):
        return MISSING
    try:
        numeric = float(value) * (100 if fraction else 1)
    except (TypeError, ValueError):
        return MISSING
    formatted = number(numeric, decimals=decimals)
    return formatted if formatted == MISSING else f"{formatted}%"


def wape(value: Any) -> str:
    return percentage(value, decimals=2)


def currency(value: Any, *, symbol: str = "$") -> str:
    formatted = number(value, decimals=2)
    return formatted if formatted == MISSING else f"{symbol}{formatted}"


def duration_seconds(value: Any) -> str:
    if is_missing(value):
        return MISSING
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return MISSING
    if not math.isfinite(seconds):
        return MISSING
    if seconds < 1:
        return f"{seconds * 1_000:,.0f} ms"
    if seconds < 60:
        return f"{seconds:,.2f} s"
    hours, remainder = divmod(int(seconds), 3_600)
    minutes, remaining_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m {remaining_seconds}s"


def utc_label(value: Any) -> str:
    if is_missing(value):
        return MISSING
    try:
        return to_utc_timestamp(value).isoformat().replace("+00:00", "Z")
    except (TypeError, ValueError):
        return MISSING


def target_label(value: Any, *, include_unit: bool = False) -> str:
    raw = str(value or "").strip()
    label = TARGET_LABELS.get(raw, raw.replace("_", " ").replace(" mw", "").title() or MISSING)
    return f"{label} MW" if include_unit and label != MISSING else label
