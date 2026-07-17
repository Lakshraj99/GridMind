"""Human-review anomaly exploration page."""

from typing import Any

import pandas as pd
import plotly.express as px


def render(st: Any, client: Any) -> None:
    st.header("Anomaly detections")
    st.warning("Detections require human review and are not confirmed operational incidents.")
    response = client.get("/api/v1/anomalies", limit=200)
    frame = pd.DataFrame(response.get("items", []))
    if frame.empty:
        st.info("No anomaly detections are available.")
        return
    st.plotly_chart(
        px.scatter(
            frame,
            x="timestamp_utc",
            y="anomaly_score",
            color="severity",
            hover_data=["target", "anomaly_type", "detector_name"],
        ),
        use_container_width=True,
    )
    st.dataframe(frame, use_container_width=True)
    summary = client.get("/api/v1/anomalies/summary")
    for warning in summary.get("calibration_warnings", []):
        st.warning(warning)
