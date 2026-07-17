"""Read-only model registry page."""

from typing import Any

import pandas as pd


def render(st: Any, client: Any) -> None:
    st.header("Model registry")
    frame = pd.DataFrame(client.get("/api/v1/models").get("items", []))
    if frame.empty:
        st.info("No registered models are available.")
        return
    st.dataframe(frame, use_container_width=True)
    st.caption("Training and model promotion are intentionally unavailable in this dashboard.")
