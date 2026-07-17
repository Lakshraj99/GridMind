"""Reusable, accessible Streamlit presentation components."""

from __future__ import annotations

import html
from collections.abc import Mapping
from datetime import UTC
from typing import Any

import pandas as pd

from gridmind.dashboard.formatting import MISSING, utc_label

TONE_CLASSES = {"neutral", "info", "success", "warning", "critical"}


def _safe(value: Any) -> str:
    return html.escape(str(value if value not in (None, "") else MISSING))


def badge_html(label: Any, tone: str = "neutral") -> str:
    """Return safe badge markup for compact status communication."""
    selected = tone if tone in TONE_CLASSES else "neutral"
    return f'<span class="gm-badge gm-{selected}">{_safe(label)}</span>'


def status_badge(status: Any) -> str:
    normalized = str(status or "unknown").lower()
    tone = {
        "ready": "success",
        "healthy": "success",
        "alive": "success",
        "optimal": "success",
        "passed": "success",
        "open": "warning",
        "acknowledged": "info",
        "suppressed": "neutral",
        "resolved": "success",
        "unavailable": "critical",
        "not ready": "critical",
        "error": "critical",
        "infeasible": "critical",
    }.get(normalized, "neutral")
    return badge_html(normalized, tone)


def severity_badge(severity: Any) -> str:
    normalized = str(severity or "unknown").lower()
    tone = {"info": "info", "warning": "warning", "critical": "critical"}.get(normalized, "neutral")
    return badge_html(normalized, tone)


def freshness_badge(timestamp: Any, *, now: Any | None = None) -> str:
    """Classify data age without inventing a timestamp when one is unavailable."""
    try:
        observed = pd.to_datetime(timestamp, utc=True)
    except (TypeError, ValueError):
        return badge_html("unknown freshness", "neutral")
    if pd.isna(observed):
        return badge_html("unknown freshness", "neutral")
    reference = pd.to_datetime(now, utc=True) if now is not None else pd.Timestamp.now(tz=UTC)
    hours = max(0.0, float((reference - observed).total_seconds()) / 3_600)
    if hours <= 2:
        return badge_html("fresh", "success")
    if hours <= 24:
        return badge_html("aging", "warning")
    return badge_html("stale", "critical")


def metric_card(
    container: Any,
    label: str,
    value: Any,
    *,
    detail: str = "",
    badge: str = "",
) -> None:
    badge_markup = f"<div>{badge}</div>" if badge else ""
    container.markdown(
        f'<div class="gm-card"><div class="gm-card-label">{_safe(label)}</div>'
        f'<div class="gm-card-value">{_safe(value)}</div>'
        f'<div class="gm-card-detail">{_safe(detail)}</div>{badge_markup}</div>',
        unsafe_allow_html=True,
    )


def page_header(st: Any, title: str, description: str, *, refreshed_at: Any = None) -> None:
    st.markdown(
        '<div class="gm-page-head"><div><div class="gm-page-kicker">GridMind workspace</div>'
        f'<div class="gm-page-title">{_safe(title)}</div>'
        f'<div class="gm-page-description">{_safe(description)}</div></div>'
        f'<div class="gm-refresh">Refreshed {utc_label(refreshed_at)}</div></div>',
        unsafe_allow_html=True,
    )


def section_header(st: Any, title: str, caption: str = "") -> None:
    st.markdown(
        f'<div class="gm-section"><div class="gm-section-title">{_safe(title)}</div>'
        f'<div class="gm-section-caption">{_safe(caption)}</div></div>',
        unsafe_allow_html=True,
    )


def empty_state(st: Any, title: str, message: str) -> None:
    st.markdown(
        f'<div class="gm-state"><div class="gm-state-title">{_safe(title)}</div>'
        f'<div class="gm-state-copy">{_safe(message)}</div></div>',
        unsafe_allow_html=True,
    )


def error_state(st: Any, message: str, *, authentication: bool = False) -> None:
    action = (
        "Check the dashboard API key configuration and retry."
        if authentication
        else "Confirm the API is running and ready, then use Refresh data."
    )
    st.markdown(
        '<div class="gm-state"><div class="gm-state-title">Data could not be loaded</div>'
        f'<div class="gm-state-copy">{_safe(message)} {_safe(action)}</div></div>',
        unsafe_allow_html=True,
    )


def disclaimer_panel(st: Any, message: str) -> None:
    st.markdown(
        f'<div class="gm-strip gm-disclaimer">{_safe(message)}</div>', unsafe_allow_html=True
    )


def lineage_panel(st: Any, lineage: Mapping[str, Any] | None) -> None:
    safe_lineage = dict(lineage or {})
    if not safe_lineage:
        empty_state(st, "Lineage unavailable", "No model lineage was returned for this record.")
        return
    items = "".join(
        '<div class="gm-lineage-item">'
        f'<div class="gm-lineage-label">{_safe(key.replace("_", " "))}</div>'
        f'<div class="gm-lineage-value">{_safe(value)}</div></div>'
        for key, value in safe_lineage.items()
        if key not in {"artifact_path", "source_path"}
    )
    st.markdown(f'<div class="gm-lineage">{items}</div>', unsafe_allow_html=True)


def filter_panel(st: Any) -> Any:
    """Return a bordered container for compact page controls."""
    return st.container(border=True)


def formatted_dataframe(st: Any, frame: pd.DataFrame, *, height: int = 360) -> None:
    display = frame.copy()
    for column in display.columns:
        if column.endswith("_utc") or column in {"forecast_origin", "timestamp_utc"}:
            display[column] = display[column].map(utc_label)
    st.dataframe(display, use_container_width=True, hide_index=True, height=height)


def chart_container(st: Any, figure: Any, *, key: str | None = None) -> None:
    st.plotly_chart(figure, use_container_width=True, key=key, config={"displaylogo": False})
