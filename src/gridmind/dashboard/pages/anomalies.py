"""Human-review anomaly investigation workflow."""

from __future__ import annotations

from typing import Any

from gridmind.dashboard.charts import anomaly_timeline, category_bar, daily_rate_figure
from gridmind.dashboard.components import (
    chart_container,
    disclaimer_panel,
    empty_state,
    filter_panel,
    lineage_panel,
    metric_card,
    page_header,
    section_header,
    severity_badge,
)
from gridmind.dashboard.formatting import (
    integer,
    megawatts,
    number,
    percentage,
    target_label,
    utc_label,
)
from gridmind.dashboard.state import DashboardContext, default_context, safe_get
from gridmind.dashboard.view_data import (
    available_options,
    count_values,
    effective_anomaly_rate,
    parse_mapping,
    records_frame,
)

DISCLAIMER = "Detected anomalies require human review and are not confirmed grid incidents."


def _total(payload: dict[str, Any] | None) -> int | None:
    pagination = (payload or {}).get("pagination")
    return (
        int(pagination["total"]) if isinstance(pagination, dict) and "total" in pagination else None
    )


def _optional_filter(value: str) -> str | None:
    return None if value == "All" else value


def render(st: Any, client: Any, context: DashboardContext | None = None) -> None:
    context = context or default_context()
    page_header(
        st,
        "Anomaly investigation",
        "Prioritize and explain persisted detections without treating them as confirmed incidents.",
        refreshed_at=context.refreshed_at,
    )
    disclaimer_panel(st, DISCLAIMER)
    base_payload, base_error = safe_get(
        client, "/api/v1/anomalies", region=context.region, limit=500
    )
    base = records_frame(
        base_payload,
        timestamp_columns=("timestamp_utc", "forecast_origin", "detected_at_utc"),
        sort_by="timestamp_utc",
        ascending=False,
    )
    controls = filter_panel(st)
    columns = controls.columns(4)
    target = str(columns[0].selectbox("Target", ["All", *available_options(base, "target")]))
    severity = str(columns[1].selectbox("Severity", ["All", "info", "warning", "critical"]))
    detector = str(
        columns[2].selectbox("Detector", ["All", *available_options(base, "detector_name")])
    )
    anomaly_type = str(
        columns[3].selectbox("Anomaly type", ["All", *available_options(base, "anomaly_type")])
    )
    filters = {
        "region": context.region,
        "target": _optional_filter(target),
        "severity": _optional_filter(severity),
        "detector": _optional_filter(detector),
        "anomaly_type": _optional_filter(anomaly_type),
        "limit": 500,
    }
    payload, filtered_error = safe_get(client, "/api/v1/anomalies", **filters)
    frame = records_frame(
        payload,
        timestamp_columns=("timestamp_utc", "forecast_origin", "detected_at_utc"),
        sort_by="timestamp_utc",
        ascending=False,
    )
    summary, _ = safe_get(client, "/api/v1/anomalies/summary")
    warning_payload, _ = safe_get(
        client,
        "/api/v1/anomalies",
        region=context.region,
        target=_optional_filter(target),
        detector=_optional_filter(detector),
        anomaly_type=_optional_filter(anomaly_type),
        severity="warning",
        limit=1,
    )
    critical_payload, _ = safe_get(
        client,
        "/api/v1/anomalies",
        region=context.region,
        target=_optional_filter(target),
        detector=_optional_filter(detector),
        anomaly_type=_optional_filter(anomaly_type),
        severity="critical",
        limit=1,
    )
    warnings = list((summary or {}).get("calibration_warnings") or [])

    cards = st.columns(5)
    metric_card(cards[0], "Total anomalies", integer(_total(payload)), detail="Current filters")
    metric_card(cards[1], "Warning", integer(_total(warning_payload)), detail=context.region)
    metric_card(cards[2], "Critical", integer(_total(critical_payload)), detail=context.region)
    metric_card(
        cards[3],
        "Effective anomaly rate",
        percentage(effective_anomaly_rate(frame), fraction=True),
        detail="Returned target-hour window",
    )
    metric_card(
        cards[4], "Calibration warnings", integer(len(warnings)), detail="Review before escalation"
    )

    if frame.empty:
        empty_state(
            st,
            "No detections match these filters",
            filtered_error or base_error or "Broaden the investigation filters or refresh the API.",
        )
        return

    section_header(
        st,
        "Detection timeline",
        "Severity communicates review priority; informational outliers are not incidents.",
    )
    chart_container(st, anomaly_timeline(frame), key="anomaly_timeline")

    distributions = st.tabs(["Severity", "Target", "Detector", "Anomaly type", "Daily activity"])
    with distributions[0]:
        chart_container(st, category_bar(count_values(frame, "severity")), key="anomaly_severity")
    with distributions[1]:
        chart_container(st, category_bar(count_values(frame, "target")), key="anomaly_target")
    with distributions[2]:
        chart_container(
            st, category_bar(count_values(frame, "detector_name")), key="anomaly_detector"
        )
    with distributions[3]:
        chart_container(st, category_bar(count_values(frame, "anomaly_type")), key="anomaly_type")
    with distributions[4]:
        chart_container(st, daily_rate_figure(frame), key="anomaly_daily")

    section_header(st, "Selected detection", "Inspect evidence and lineage before taking action.")
    labels = {
        str(row["anomaly_id"]): (
            f"{utc_label(row.get('timestamp_utc'))} · {target_label(row.get('target'))} · "
            f"{str(row.get('anomaly_type', '')).replace('_', ' ')}"
        )
        for _, row in frame.iterrows()
    }
    selected_id = str(
        st.selectbox("Detection", list(labels), format_func=lambda value: labels[value])
    )
    detail, _ = safe_get(client, f"/api/v1/anomalies/{selected_id}")
    selected = detail or frame[frame["anomaly_id"] == selected_id].iloc[0].to_dict()
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown(
            f"{severity_badge(selected.get('severity'))} &nbsp; "
            f"`{utc_label(selected.get('timestamp_utc'))}`",
            unsafe_allow_html=True,
        )
        selected_type = str(selected.get("anomaly_type", "unknown")).replace("_", " ")
        st.markdown(f"**{target_label(selected.get('target'))} · {selected_type}**")
        st.write(selected.get("explanation") or "No explanation was provided by the detector.")
        if (
            str(selected.get("detector_name", "")).lower().startswith("isolation")
            and str(selected.get("severity", "")).lower() == "info"
        ):
            st.caption(
                "This informational IsolationForest result is a statistical outlier for review."
            )
        details = st.columns(3)
        metric_card(details[0], "Observed", megawatts(selected.get("observed_value")))
        metric_card(details[1], "Expected", megawatts(selected.get("expected_value")))
        metric_card(details[2], "Score", number(selected.get("anomaly_score"), decimals=1))
    with right:
        detector_name = selected.get("detector_name") or "—"
        detector_version = selected.get("detector_version") or "—"
        st.caption(f"Detector: {detector_name} v{detector_version}")
        st.caption(f"Threshold: {number(selected.get('threshold'), decimals=2)}")
        metadata = parse_mapping(selected.get("metadata_json"))
        feature_summary = parse_mapping(selected.get("feature_summary"))
        with st.expander("Detector contribution metadata", expanded=True):
            if metadata or feature_summary:
                st.json({"metadata": metadata, "features": feature_summary})
            else:
                st.caption("No contribution metadata was returned.")
        lineage_panel(
            st,
            {
                "model_name": selected.get("model_name"),
                "model_version": selected.get("model_version"),
                "run_id": selected.get("run_id"),
                "forecast_origin": utc_label(selected.get("forecast_origin")),
            },
        )
    for warning in warnings:
        st.warning(str(warning))
