"""Streamlit dashboard composition and safe API error handling."""

from __future__ import annotations

import streamlit as st

from gridmind.config import Settings
from gridmind.dashboard.api_client import DashboardAPIError
from gridmind.dashboard.pages import alerts, anomalies, battery, forecasts, models, overview
from gridmind.dashboard.state import create_client


def main() -> None:
    settings = Settings()
    st.set_page_config(page_title="GridMind", page_icon="⚡", layout="wide")
    st.title("GridMind decision support")
    page = str(
        st.sidebar.radio(
            "Workspace",
            ["Overview", "Forecasts", "Anomalies", "Alerts", "Battery dispatch", "Models"],
        )
    )
    renderers = {
        "Overview": overview.render,
        "Forecasts": forecasts.render,
        "Anomalies": anomalies.render,
        "Alerts": alerts.render,
        "Battery dispatch": battery.render,
        "Models": models.render,
    }
    client = create_client(settings)
    try:
        with st.spinner("Loading GridMind data…"):
            renderers[page](st, client)
    except DashboardAPIError as exc:
        st.error(str(exc))
    finally:
        client.close()


if __name__ == "__main__":
    main()
