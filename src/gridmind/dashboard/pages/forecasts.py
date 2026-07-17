"""Forecast exploration with explicit lineage and target-aware presentation."""

from __future__ import annotations

from typing import Any

import pandas as pd

from gridmind.dashboard.charts import forecast_figure
from gridmind.dashboard.components import (
    chart_container,
    empty_state,
    error_state,
    filter_panel,
    formatted_dataframe,
    lineage_panel,
    metric_card,
    page_header,
    section_header,
)
from gridmind.dashboard.formatting import megawatts, target_label, utc_label
from gridmind.dashboard.state import DashboardContext, default_context, safe_get
from gridmind.dashboard.view_data import available_options, records_frame

TARGETS = ["demand_mw", "net_load_mw", "solar_generation_mw", "wind_generation_mw"]


def _origin_label(value: Any) -> str:
    return "Latest complete horizon" if value == "latest" else utc_label(value)


def _selected_forecast(
    history: pd.DataFrame,
    latest_payload: dict[str, Any] | None,
    origin: str,
    horizon: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if origin == "latest":
        return (
            records_frame(
                latest_payload,
                timestamp_columns=("timestamp_utc", "forecast_origin"),
                sort_by="timestamp_utc",
            ),
            dict((latest_payload or {}).get("lineage") or {}),
        )
    selected = history[history["forecast_origin"].map(utc_label) == utc_label(origin)].copy()
    selected = selected.sort_values("timestamp_utc").head(horizon).reset_index(drop=True)
    lineage: dict[str, Any] = {}
    if not selected.empty:
        value = selected.iloc[0].get("lineage")
        if isinstance(value, dict):
            lineage = dict(value)
        else:
            lineage = {
                key: selected.iloc[0].get(key)
                for key in ("model_name", "model_version", "run_id")
                if key in selected
            }
    return selected, lineage


def render(st: Any, client: Any, context: DashboardContext | None = None) -> None:
    context = context or default_context()
    page_header(
        st,
        "Forecast analysis",
        "Inspect persisted model predictions, completeness, and serving lineage in canonical UTC.",
        refreshed_at=context.refreshed_at,
    )
    regions = list(context.forecast_summary.get("available_regions") or [context.region])
    if context.region not in regions:
        regions.append(context.region)
    controls = filter_panel(st)
    columns = controls.columns([1, 1.5, 1, 1])
    region = str(columns[0].selectbox("Region", regions, index=regions.index(context.region)))
    target = str(
        columns[1].selectbox(
            "Target",
            TARGETS,
            format_func=lambda value: target_label(value, include_unit=True),
        )
    )
    model_alias = str(columns[2].selectbox("Model alias", ["champion", "candidate"]))
    horizon = int(columns[3].selectbox("Horizon", [12, 24, 48, 72], index=1))

    history_payload, history_error = safe_get(
        client,
        "/api/v1/forecasts",
        region=region,
        target=target,
        model_alias=model_alias,
        limit=500,
    )
    latest_payload, latest_error = safe_get(
        client,
        "/api/v1/forecasts/latest",
        region=region,
        target=target,
        horizon=horizon,
        model_alias=model_alias,
    )
    history = records_frame(
        history_payload,
        timestamp_columns=("timestamp_utc", "forecast_origin"),
        sort_by="timestamp_utc",
    )
    origin_values = history.get("forecast_origin", pd.Series(dtype="object")).map(utc_label)
    origins = available_options(pd.DataFrame({"origin": origin_values}), "origin")
    origin_options = ["latest", *sorted(origins, reverse=True)]
    selected_origin = str(
        controls.selectbox("Forecast origin", origin_options, format_func=_origin_label)
    )
    frame, lineage = _selected_forecast(history, latest_payload, selected_origin, horizon)

    if frame.empty:
        if latest_error and history_error:
            error_state(st, latest_error)
        else:
            empty_state(
                st,
                "No forecast series found",
                "Choose another region, target, alias, origin, or horizon and refresh the data.",
            )
        return

    completeness = len(frame) == horizon and frame["timestamp_utc"].nunique() == horizon
    predicted_values = (
        frame["predicted_value"] if "predicted_value" in frame else pd.Series(dtype=float)
    )
    predicted = pd.to_numeric(predicted_values, errors="coerce")
    cards = st.columns(4)
    metric_card(
        cards[0],
        "First forecast point",
        megawatts(predicted.iloc[0] if not predicted.empty else None),
        detail=utc_label(frame.iloc[0].get("timestamp_utc")),
    )
    metric_card(
        cards[1], "Horizon average", megawatts(predicted.mean()), detail=target_label(target)
    )
    metric_card(cards[2], "Horizon maximum", megawatts(predicted.max()), detail=region)
    metric_card(
        cards[3],
        "Horizon completeness",
        f"{len(frame)} / {horizon}",
        detail="Complete and contiguous" if completeness else "Review missing or duplicate points",
    )

    section_header(
        st,
        f"{target_label(target)} forecast",
        "The forecast trace is a model prediction. Actual observations appear only when "
        "supplied by the API.",
    )
    chart_container(st, forecast_figure(frame, target=target), key="forecast_primary")

    lineage_column, context_column = st.columns([1.5, 1])
    with lineage_column:
        section_header(st, "Model lineage", "Serving metadata returned with this forecast.")
        lineage_panel(st, lineage)
        run_id = lineage.get("run_id")
        if run_id:
            st.caption("Run ID")
            st.code(str(run_id), language=None)
    with context_column:
        section_header(st, "Forecast context")
        st.caption("Forecast origin")
        st.code(utc_label(frame.iloc[0].get("forecast_origin")), language=None)
        weather_mode = frame.iloc[0].get("weather_mode") if "weather_mode" in frame else None
        st.caption(f"Weather mode: {weather_mode or 'not provided'}")
        st.caption(f"Requested alias: {model_alias}")

    with st.expander("Forecast points", expanded=False):
        visible = [
            column
            for column in (
                "timestamp_utc",
                "predicted_value",
                "actual_value",
                "forecast_origin",
                "model_name",
                "model_version",
                "weather_mode",
            )
            if column in frame
        ]
        formatted_dataframe(st, frame[visible], height=420)
