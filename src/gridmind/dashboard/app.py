"""Professional Streamlit shell with safe API-only page composition."""

from __future__ import annotations

from typing import Any

import streamlit as st

from gridmind.config import Settings
from gridmind.dashboard.api_client import DashboardAPIError, DashboardAuthenticationError
from gridmind.dashboard.components import (
    badge_html,
    disclaimer_panel,
    error_state,
    status_badge,
)
from gridmind.dashboard.formatting import utc_label
from gridmind.dashboard.pages import alerts, anomalies, battery, forecasts, models, overview
from gridmind.dashboard.state import DashboardContext, create_client, safe_get, utc_now
from gridmind.dashboard.styles import apply_theme

PAGES = ["Overview", "Forecasts", "Anomalies", "Alerts", "Battery dispatch", "Models"]
DISCLAIMER = (
    "GridMind is decision-support software. Forecasts, detections, alerts, and simulated dispatch "
    "require qualified human review and do not control physical equipment."
)


def _status_from_readiness(payload: dict[str, Any] | None) -> bool:
    return bool(payload and str(payload.get("status", "")).lower() == "ready")


def _sidebar(
    client: Any,
    live_payload: dict[str, Any] | None,
    ready_payload: dict[str, Any] | None,
    forecast_summary: dict[str, Any] | None,
) -> tuple[str, str, Any]:
    st.sidebar.markdown('<div class="gm-shell-title">GridMind</div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div class="gm-shell-subtitle">Energy ML decision support</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")
    page = str(st.sidebar.radio("Workspace", PAGES, label_visibility="collapsed"))
    regions = list((forecast_summary or {}).get("available_regions") or ["PJM"])
    region = str(st.sidebar.selectbox("Region", regions, index=0))
    if st.sidebar.button("Refresh data", use_container_width=True):
        client.clear_cache()
        st.session_state["gridmind_refreshed_at"] = utc_now()
        st.rerun()
    refreshed_at = st.session_state.setdefault("gridmind_refreshed_at", utc_now())
    live = bool(live_payload and live_payload.get("status") == "alive")
    ready = _status_from_readiness(ready_payload)
    st.sidebar.markdown("#### API status")
    st.sidebar.markdown(
        f"{status_badge('alive' if live else 'unavailable')} "
        f"{status_badge('ready' if ready else 'not ready')}",
        unsafe_allow_html=True,
    )
    st.sidebar.caption(f"Last refresh: {utc_label(refreshed_at)}")
    st.sidebar.markdown(f'<div class="gm-strip">{DISCLAIMER}</div>', unsafe_allow_html=True)
    return page, region, refreshed_at


def main() -> None:
    settings = Settings()
    st.set_page_config(page_title="GridMind", layout="wide")
    apply_theme(st)
    client = create_client(settings)
    try:
        live_payload, live_error = safe_get(client, "/health/live")
        ready_payload, ready_error = safe_get(client, "/health/ready")
        forecast_summary, _ = safe_get(client, "/api/v1/forecasts/summary")
        page, region, refreshed_at = _sidebar(client, live_payload, ready_payload, forecast_summary)
        context = DashboardContext(
            region=region,
            refreshed_at=refreshed_at,
            live=bool(live_payload and live_payload.get("status") == "alive"),
            ready=_status_from_readiness(ready_payload),
            readiness=ready_payload or {},
            forecast_summary=forecast_summary or {},
        )
        if live_error or ready_error:
            st.markdown(badge_html("API connection degraded", "warning"), unsafe_allow_html=True)
        renderers = {
            "Overview": overview.render,
            "Forecasts": forecasts.render,
            "Anomalies": anomalies.render,
            "Alerts": alerts.render,
            "Battery dispatch": battery.render,
            "Models": models.render,
        }
        with st.spinner("Loading verified API data…"):
            renderers[page](st, client, context)
        disclaimer_panel(st, DISCLAIMER)
        st.markdown(
            '<div class="gm-footer">GridMind · API-backed analytics · canonical UTC</div>',
            unsafe_allow_html=True,
        )
    except DashboardAuthenticationError as exc:
        error_state(st, str(exc), authentication=True)
    except DashboardAPIError as exc:
        error_state(st, str(exc))
    finally:
        client.close()


if __name__ == "__main__":
    main()
