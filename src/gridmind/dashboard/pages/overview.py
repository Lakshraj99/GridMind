"""Screenshot-ready operational decision-support overview."""

from __future__ import annotations

from typing import Any

from gridmind.dashboard.charts import (
    anomaly_timeline,
    category_bar,
    combined_forecast_figure,
    dispatch_load_figure,
)
from gridmind.dashboard.components import (
    chart_container,
    empty_state,
    freshness_badge,
    lineage_panel,
    metric_card,
    page_header,
    section_header,
    status_badge,
)
from gridmind.dashboard.formatting import integer, megawatts, utc_label
from gridmind.dashboard.state import DashboardContext, default_context, safe_get
from gridmind.dashboard.view_data import count_values, records_frame


def _pagination_total(payload: dict[str, Any] | None) -> int | None:
    pagination = (payload or {}).get("pagination", {})
    value = pagination.get("total") if isinstance(pagination, dict) else None
    return int(value) if value is not None else None


def _next_forecast(frame: Any) -> tuple[Any, str]:
    if frame.empty or "predicted_value" not in frame:
        return None, "No complete horizon"
    timestamp = frame.iloc[0].get("timestamp_utc")
    return frame.iloc[0].get("predicted_value"), utc_label(timestamp)


def render(st: Any, client: Any, context: DashboardContext | None = None) -> None:
    context = context or default_context()
    page_header(
        st,
        "Grid operations overview",
        "A real-data snapshot of forecasts, human-review signals, model lineage, and "
        "simulated dispatch.",
        refreshed_at=context.refreshed_at,
    )

    demand_payload, _ = safe_get(
        client,
        "/api/v1/forecasts/latest",
        region=context.region,
        target="demand_mw",
        horizon=24,
        model_alias="champion",
    )
    net_payload, _ = safe_get(
        client,
        "/api/v1/forecasts/latest",
        region=context.region,
        target="net_load_mw",
        horizon=24,
        model_alias="champion",
    )
    open_alerts, _ = safe_get(
        client, "/api/v1/alerts", region=context.region, status="open", limit=200
    )
    warning_alerts, _ = safe_get(
        client,
        "/api/v1/alerts",
        region=context.region,
        status="open",
        severity="warning",
        limit=1,
    )
    critical_alerts, _ = safe_get(
        client,
        "/api/v1/alerts",
        region=context.region,
        status="open",
        severity="critical",
        limit=1,
    )
    anomalies_payload, _ = safe_get(client, "/api/v1/anomalies", region=context.region, limit=200)
    dispatches_payload, _ = safe_get(client, "/api/v1/dispatches", region=context.region, limit=20)
    model_summary, _ = safe_get(client, "/api/v1/models/summary")

    demand = records_frame(
        demand_payload, timestamp_columns=("timestamp_utc",), sort_by="timestamp_utc"
    )
    net_load = records_frame(
        net_payload, timestamp_columns=("timestamp_utc",), sort_by="timestamp_utc"
    )
    alerts = records_frame(
        open_alerts, timestamp_columns=("first_seen_utc", "last_seen_utc"), sort_by="last_seen_utc"
    )
    anomalies = records_frame(
        anomalies_payload,
        timestamp_columns=("timestamp_utc",),
        sort_by="timestamp_utc",
    )
    dispatches = records_frame(
        dispatches_payload,
        timestamp_columns=("forecast_origin",),
        sort_by="forecast_origin",
        ascending=False,
    )
    demand_value, demand_time = _next_forecast(demand)
    net_value, net_time = _next_forecast(net_load)

    latest_dispatch = dispatches.iloc[0].to_dict() if not dispatches.empty else {}
    dispatch_id = latest_dispatch.get("dispatch_run_id")
    dispatch_summary, _ = (
        safe_get(client, f"/api/v1/dispatches/{dispatch_id}/summary")
        if dispatch_id
        else (None, None)
    )
    dispatch_points_payload, _ = (
        safe_get(client, f"/api/v1/dispatches/{dispatch_id}/points", limit=500)
        if dispatch_id
        else (None, None)
    )
    dispatch_points = records_frame(
        dispatch_points_payload,
        timestamp_columns=("timestamp_utc",),
        sort_by="timestamp_utc",
    )

    warning_count = _pagination_total(warning_alerts)
    critical_count = _pagination_total(critical_alerts)
    alert_risk = (
        None if warning_count is None or critical_count is None else warning_count + critical_count
    )
    cards = st.columns(6)
    metric_card(cards[0], "Next demand forecast", megawatts(demand_value), detail=demand_time)
    metric_card(cards[1], "Next net-load forecast", megawatts(net_value), detail=net_time)
    metric_card(
        cards[2], "Open alerts", integer(_pagination_total(open_alerts)), detail=context.region
    )
    metric_card(
        cards[3],
        "Warning / critical",
        integer(alert_risk),
        detail=(
            f"{integer(warning_count)} warning · {integer(critical_count)} critical"
            if alert_risk is not None
            else "Alert severities unavailable"
        ),
    )
    metric_card(
        cards[4],
        "Latest peak reduction",
        megawatts((dispatch_summary or {}).get("peak_reduction_mw")),
        detail=utc_label(latest_dispatch.get("forecast_origin")),
    )
    metric_card(
        cards[5],
        "Champion versions",
        integer((model_summary or {}).get("champion_versions")),
        detail=f"{integer((model_summary or {}).get('registered_models'))} registered models",
    )

    section_header(
        st,
        "Forecast outlook",
        "Persisted predictions only; forecast values are not presented as observations.",
    )
    chart_container(
        st,
        combined_forecast_figure({"demand_mw": demand, "net_load_mw": net_load}),
        key="overview_forecasts",
    )

    left, right = st.columns(2)
    with left:
        section_header(st, "Alert severity", "Open alerts returned for the selected region.")
        chart_container(
            st,
            category_bar(count_values(alerts, "severity"), title="Open alerts"),
            key="overview_alerts",
        )
    with right:
        section_header(st, "Recent anomaly activity", "Detections require investigation.")
        chart_container(st, anomaly_timeline(anomalies), key="overview_anomalies")

    section_header(
        st,
        "Latest simulated battery response",
        "Net load before and after the latest persisted decision-support simulation.",
    )
    if dispatch_points.empty:
        empty_state(
            st,
            "No dispatch profile available",
            "Run and persist a battery optimization to populate this panel.",
        )
    else:
        chart_container(st, dispatch_load_figure(dispatch_points), key="overview_dispatch")

    freshness, health = st.columns(2)
    with freshness:
        section_header(st, "Data freshness", "Canonical UTC timestamps from API records.")
        latest_forecast_origin = context.forecast_summary.get("latest_forecast_origin")
        st.markdown(
            f"{freshness_badge(latest_forecast_origin)} &nbsp; Latest forecast origin: "
            f"`{utc_label(latest_forecast_origin)}`",
            unsafe_allow_html=True,
        )
        st.caption(
            "Anomaly activity: "
            + utc_label(
                anomalies["timestamp_utc"].max()
                if not anomalies.empty and "timestamp_utc" in anomalies
                else None
            )
        )
        st.caption(
            "Alert activity: "
            + utc_label(
                alerts["last_seen_utc"].max()
                if not alerts.empty and "last_seen_utc" in alerts
                else None
            )
        )
    with health:
        section_header(
            st, "Model and service status", "Serving metadata only; no training runs here."
        )
        registry_status = (
            "ready" if (model_summary or {}).get("champion_versions") else "unavailable"
        )
        st.markdown(
            f"API {status_badge('ready' if context.ready else 'not ready')} &nbsp; "
            f"Champion registry {status_badge(registry_status)}",
            unsafe_allow_html=True,
        )
        lineage_panel(st, (demand_payload or {}).get("lineage"))
