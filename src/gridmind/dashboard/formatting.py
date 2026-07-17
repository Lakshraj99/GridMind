"""Dashboard-safe formatting helpers."""

from __future__ import annotations

from typing import Any

from gridmind.time_utils import to_utc_timestamp


def utc_label(value: Any) -> str:
    if value in (None, ""):
        return "—"
    return to_utc_timestamp(value).isoformat().replace("+00:00", "Z")


def megawatts(value: Any) -> str:
    return "—" if value is None else f"{float(value):,.1f} MW"
