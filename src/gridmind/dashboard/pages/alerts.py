"""Controlled alert lifecycle page."""

from typing import Any

import pandas as pd


def render(st: Any, client: Any) -> None:
    st.header("Alerts")
    status = st.selectbox("Status", ["open", "acknowledged", "resolved", "suppressed"])
    response = client.get("/api/v1/alerts", status=status, limit=200)
    frame = pd.DataFrame(response.get("items", []))
    if frame.empty:
        st.info("No alerts match this status.")
        return
    st.dataframe(frame, use_container_width=True)
    alert_id = st.selectbox("Selected alert", frame["alert_id"].tolist())
    next_status = st.selectbox(
        "Lifecycle action", ["acknowledged", "resolved", "suppressed", "open"]
    )
    confirmed = st.checkbox("I confirm this lifecycle update")
    if st.button("Apply update", disabled=not confirmed):
        client.patch(f"/api/v1/alerts/{alert_id}", {"status": next_status})
        st.success("Alert status updated.")
