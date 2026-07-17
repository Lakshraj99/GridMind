"""Controlled alert triage and lifecycle workflow."""

from __future__ import annotations

from typing import Any

from gridmind.dashboard.components import (
    empty_state,
    filter_panel,
    formatted_dataframe,
    metric_card,
    page_header,
    section_header,
    severity_badge,
    status_badge,
)
from gridmind.dashboard.formatting import integer, target_label, utc_label
from gridmind.dashboard.state import DashboardContext, default_context, safe_get
from gridmind.dashboard.view_data import available_options, count_values, records_frame

ALLOWED_TRANSITIONS = {
    "open": ["acknowledged", "resolved", "suppressed"],
    "acknowledged": ["open", "resolved", "suppressed"],
    "resolved": ["open"],
    "suppressed": ["open", "resolved"],
}


def _optional(value: str) -> str | None:
    return None if value == "All" else value


def render(st: Any, client: Any, context: DashboardContext | None = None) -> None:
    context = context or default_context()
    page_header(
        st,
        "Alert triage",
        "Review deduplicated alert state and apply explicit, authenticated lifecycle transitions.",
        refreshed_at=context.refreshed_at,
    )
    flash = st.session_state.pop("gridmind_alert_flash", None)
    if flash:
        st.success(str(flash))

    base_payload, _ = safe_get(client, "/api/v1/alerts", region=context.region, limit=500)
    base = records_frame(
        base_payload,
        timestamp_columns=("first_seen_utc", "last_seen_utc", "updated_at_utc"),
        sort_by="last_seen_utc",
        ascending=False,
    )
    controls = filter_panel(st)
    row = controls.columns([1, 1, 1, 1.4, 1])
    status = str(
        row[0].selectbox(
            "Status", ["All", "open", "acknowledged", "resolved", "suppressed"], index=1
        )
    )
    severity = str(row[1].selectbox("Severity", ["All", "info", "warning", "critical"]))
    target = str(row[2].selectbox("Target", ["All", *available_options(base, "target")]))
    search = str(row[3].text_input("Search", placeholder="Summary, type, or alert ID"))
    sort_label = str(row[4].selectbox("Sort", ["Last seen", "First seen", "Occurrences"]))
    payload, _ = safe_get(
        client,
        "/api/v1/alerts",
        region=context.region,
        status=_optional(status),
        severity=_optional(severity),
        target=_optional(target),
        limit=500,
    )
    frame = records_frame(
        payload,
        timestamp_columns=("first_seen_utc", "last_seen_utc", "updated_at_utc"),
    )
    if search and not frame.empty:
        searchable = frame.reindex(columns=["alert_id", "summary", "title", "anomaly_type"]).fillna(
            ""
        )
        mask = (
            searchable.astype(str)
            .apply(lambda column: column.str.contains(search, case=False, regex=False))
            .any(axis=1)
        )
        frame = frame[mask].copy()
    sort_column = {
        "Last seen": "last_seen_utc",
        "First seen": "first_seen_utc",
        "Occurrences": "occurrence_count",
    }[sort_label]
    if sort_column in frame:
        frame = frame.sort_values(sort_column, ascending=False).reset_index(drop=True)

    statuses = count_values(base, "status")
    severities = count_values(base, "severity")
    cards = st.columns(6)
    metric_card(cards[0], "Open", integer(statuses.get("open", 0)), detail=context.region)
    metric_card(cards[1], "Acknowledged", integer(statuses.get("acknowledged", 0)))
    metric_card(cards[2], "Resolved", integer(statuses.get("resolved", 0)))
    metric_card(cards[3], "Suppressed", integer(statuses.get("suppressed", 0)))
    metric_card(cards[4], "Warning", integer(severities.get("warning", 0)))
    metric_card(cards[5], "Critical", integer(severities.get("critical", 0)))

    if frame.empty:
        empty_state(
            st,
            "No alerts match the current triage view",
            "Change the status, severity, target, or search filters and refresh the data.",
        )
        return

    section_header(
        st,
        "Alert queue",
        f"Showing {len(frame):,} returned alerts after local search and sorting.",
    )
    table_columns = [
        column
        for column in (
            "severity",
            "status",
            "target",
            "anomaly_type",
            "first_seen_utc",
            "last_seen_utc",
            "occurrence_count",
            "summary",
        )
        if column in frame
    ]
    formatted_dataframe(st, frame[table_columns], height=330)

    labels = {
        str(record["alert_id"]): (
            f"{str(record.get('severity', '')).upper()} · {target_label(record.get('target'))} · "
            f"{str(record.get('anomaly_type', '')).replace('_', ' ')} · "
            f"{utc_label(record.get('last_seen_utc'))}"
        )
        for _, record in frame.iterrows()
    }
    selected_id = str(
        st.selectbox("Selected alert", list(labels), format_func=lambda key: labels[key])
    )
    detail, detail_error = safe_get(client, f"/api/v1/alerts/{selected_id}")
    if detail_error or detail is None:
        empty_state(st, "Alert detail unavailable", detail_error or "Refresh the alert queue.")
        return

    section_header(
        st,
        detail.get("title") or "Alert detail",
        "Investigation context precedes lifecycle action.",
    )
    st.markdown(
        f"{severity_badge(detail.get('severity'))} &nbsp; {status_badge(detail.get('status'))}",
        unsafe_allow_html=True,
    )
    st.write(detail.get("summary") or "No alert summary was provided.")
    detail_columns = st.columns(4)
    metric_card(detail_columns[0], "Target", target_label(detail.get("target")))
    metric_card(detail_columns[1], "Occurrences", integer(detail.get("occurrence_count")))
    metric_card(detail_columns[2], "First seen", utc_label(detail.get("first_seen_utc")))
    metric_card(detail_columns[3], "Last seen", utc_label(detail.get("last_seen_utc")))

    history = records_frame(
        {"items": detail.get("history", [])},
        timestamp_columns=("changed_at_utc",),
        sort_by="changed_at_utc",
        ascending=False,
    )
    with st.expander("Lifecycle history", expanded=False):
        if history.empty:
            st.caption("No lifecycle history was returned.")
        else:
            formatted_dataframe(st, history, height=240)

    with st.expander("Update lifecycle status", expanded=False):
        current = str(detail.get("status") or "open")
        choices = ALLOWED_TRANSITIONS.get(current, [])
        if not choices:
            st.caption("No valid transitions are available from this state.")
            return
        next_status = str(st.selectbox("New status", choices, key=f"next_{selected_id}"))
        confirmed = bool(
            st.checkbox(
                f"I confirm changing this alert from {current} to {next_status}",
                key=f"confirm_{selected_id}_{next_status}",
            )
        )
        if st.button(
            "Apply lifecycle update",
            disabled=not confirmed or next_status == current,
            type="secondary",
            key=f"update_{selected_id}",
        ):
            with st.spinner("Applying authenticated lifecycle update…"):
                updated = client.patch(f"/api/v1/alerts/{selected_id}", {"status": next_status})
            st.session_state["gridmind_alert_flash"] = (
                f"Alert updated to {updated.get('status', next_status)}."
            )
            st.rerun()
