"""Simulated battery dispatch decision-support page."""

from typing import Any

import pandas as pd
import plotly.express as px


def render(st: Any, client: Any) -> None:
    st.header("Battery dispatch simulations")
    st.warning("Decision-support simulation only; no physical battery is controlled.")
    runs = client.get("/api/v1/dispatches", limit=100).get("items", [])
    if not runs:
        st.info("No dispatch simulations are available.")
        return
    run_id = st.selectbox("Dispatch run", [row["dispatch_run_id"] for row in runs])
    points = pd.DataFrame(
        client.get(f"/api/v1/dispatches/{run_id}/points", limit=500).get("items", [])
    )
    summary = client.get(f"/api/v1/dispatches/{run_id}/summary")
    st.metric("Peak reduction", f"{summary['peak_reduction_mw']:,.1f} MW")
    if not points.empty:
        st.plotly_chart(
            px.line(
                points,
                x="timestamp_utc",
                y=["net_load_before_battery_mw", "net_load_after_battery_mw"],
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            px.line(points, x="timestamp_utc", y="soc_end_mwh"), use_container_width=True
        )
    st.json(summary)
