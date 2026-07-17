"""Operational decision-support overview."""

from typing import Any


def render(st: Any, client: Any) -> None:
    st.header("Grid operations overview")
    columns = st.columns(3)
    forecast = client.get("/api/v1/forecasts/summary")
    alerts = client.get("/api/v1/alerts", status="open", limit=1)
    models = client.get("/api/v1/models/summary")
    columns[0].metric("Forecast rows", forecast.get("total_rows", 0))
    columns[1].metric("Open alerts", alerts["pagination"]["total"])
    columns[2].metric("Registered models", models.get("registered_models", 0))
    st.caption("Data freshness and status reflect the latest persisted API records.")
