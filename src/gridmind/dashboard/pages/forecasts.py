"""Forecast exploration page."""

from typing import Any

import pandas as pd
import plotly.express as px


def render(st: Any, client: Any) -> None:
    st.header("Forecasts")
    region = st.text_input("Region", "PJM")
    target = st.selectbox(
        "Target", ["demand_mw", "net_load_mw", "solar_generation_mw", "wind_generation_mw"]
    )
    response = client.get(
        "/api/v1/forecasts/latest", region=region, target=target, horizon=24, model_alias="champion"
    )
    frame = pd.DataFrame(response.get("items", []))
    if frame.empty:
        st.info("No forecasts match these filters.")
        return
    st.plotly_chart(
        px.line(frame, x="timestamp_utc", y="predicted_value", markers=True),
        use_container_width=True,
    )
    st.dataframe(frame, use_container_width=True)
    st.json(response.get("lineage", {}))
