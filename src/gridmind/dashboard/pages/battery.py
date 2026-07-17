"""Portfolio-ready simulated battery dispatch decision-support page."""

from __future__ import annotations

from typing import Any

from gridmind.dashboard.charts import dispatch_load_figure, dispatch_power_figure, soc_figure
from gridmind.dashboard.components import (
    chart_container,
    disclaimer_panel,
    empty_state,
    filter_panel,
    lineage_panel,
    metric_card,
    page_header,
    section_header,
    status_badge,
)
from gridmind.dashboard.formatting import (
    duration_seconds,
    megawatt_hours,
    megawatts,
    number,
    percentage,
    utc_label,
)
from gridmind.dashboard.state import DashboardContext, default_context, safe_get
from gridmind.dashboard.view_data import available_options, records_frame

DISCLAIMER = "This is a decision-support simulation and does not control a physical battery."


def _option(value: str) -> str | None:
    return None if value == "All" else value


def render(st: Any, client: Any, context: DashboardContext | None = None) -> None:
    context = context or default_context()
    page_header(
        st,
        "Battery dispatch simulation",
        "Review physics-validated schedules, forecast lineage, and solver diagnostics.",
        refreshed_at=context.refreshed_at,
    )
    disclaimer_panel(st, DISCLAIMER)
    base_payload, base_error = safe_get(
        client, "/api/v1/dispatches", region=context.region, limit=500
    )
    base = records_frame(
        base_payload,
        timestamp_columns=("forecast_origin", "created_at_utc"),
        sort_by="forecast_origin",
        ascending=False,
    )
    controls = filter_panel(st)
    columns = controls.columns(3)
    battery_id = str(
        columns[0].selectbox("Battery", ["All", *available_options(base, "battery_id")])
    )
    objective = str(
        columns[1].selectbox("Objective", ["All", *available_options(base, "objective_mode")])
    )
    solver = str(
        columns[2].selectbox("Solver status", ["All", *available_options(base, "solver_status")])
    )
    payload, _ = safe_get(
        client,
        "/api/v1/dispatches",
        region=context.region,
        battery_id=_option(battery_id),
        objective_mode=_option(objective),
        solver_status=_option(solver),
        limit=500,
    )
    runs = records_frame(
        payload,
        timestamp_columns=("forecast_origin", "created_at_utc"),
        sort_by="forecast_origin",
        ascending=False,
    )
    if runs.empty:
        empty_state(
            st,
            "No dispatch simulations match these filters",
            base_error or "Persist a simulation or choose another battery, objective, or status.",
        )
        return

    labels = {
        str(row["dispatch_run_id"]): (
            f"{utc_label(row.get('forecast_origin'))} · {row.get('battery_id', 'battery')} · "
            f"{str(row.get('objective_mode', '')).replace('_', ' ')}"
        )
        for _, row in runs.iterrows()
    }
    run_id = str(
        controls.selectbox("Dispatch run", list(labels), format_func=lambda value: labels[value])
    )
    detail, detail_error = safe_get(client, f"/api/v1/dispatches/{run_id}")
    summary, summary_error = safe_get(client, f"/api/v1/dispatches/{run_id}/summary")
    points_payload, points_error = safe_get(
        client, f"/api/v1/dispatches/{run_id}/points", limit=500
    )
    if detail is None or summary is None:
        empty_state(
            st,
            "Dispatch detail unavailable",
            detail_error or summary_error or "Refresh the selected simulation.",
        )
        return
    points = records_frame(
        points_payload,
        timestamp_columns=("timestamp_utc", "forecast_origin", "created_at_utc"),
        sort_by="timestamp_utc",
    )
    terminal_soc = points.iloc[-1].get("soc_end_mwh") if not points.empty else None
    raw_physics = summary.get("physics")
    physics: dict[str, Any] = raw_physics if isinstance(raw_physics, dict) else {}
    peak_reduction = summary.get("peak_reduction_mw")

    first_row = st.columns(4)
    metric_card(
        first_row[0],
        "Solver status",
        str(detail.get("solver_status") or "—").title(),
        badge=status_badge(detail.get("solver_status")),
    )
    metric_card(first_row[1], "Peak before", megawatts(detail.get("peak_before_mw")))
    metric_card(first_row[2], "Peak after", megawatts(detail.get("peak_after_mw")))
    metric_card(first_row[3], "Peak reduction", megawatts(peak_reduction))
    second_row = st.columns(4)
    metric_card(second_row[0], "Charge energy", megawatt_hours(detail.get("total_charge_mwh")))
    metric_card(
        second_row[1], "Discharge energy", megawatt_hours(detail.get("total_discharge_mwh"))
    )
    metric_card(second_row[2], "Terminal SOC", megawatt_hours(terminal_soc))
    metric_card(
        second_row[3],
        "Equivalent cycles",
        number(physics.get("equivalent_cycles"), decimals=3),
        detail=f"Throughput {megawatt_hours(physics.get('total_throughput_mwh'))}",
    )

    section_header(
        st,
        "Net-load impact",
        "Same-origin forecast profile before and after the simulated battery schedule.",
    )
    if points.empty:
        empty_state(
            st,
            "Dispatch points unavailable",
            points_error or "The run summary exists, but no schedule points were returned.",
        )
    else:
        chart_container(st, dispatch_load_figure(points), key="battery_load")
        schedule, soc = st.columns(2)
        with schedule:
            section_header(st, "Charge and discharge", "Charging is shown below zero.")
            chart_container(st, dispatch_power_figure(points), key="battery_power")
        with soc:
            section_header(st, "State of charge", "End-of-step energy in MWh.")
            chart_container(st, soc_figure(points), key="battery_soc")

    tabs = st.tabs(
        ["Battery specification", "Forecast lineage", "Objective", "Solver and constraints"]
    )
    with tabs[0]:
        specification = summary.get("battery_specification")
        if isinstance(specification, dict) and specification:
            st.json(specification)
        else:
            st.caption("Battery specification was not returned.")
    with tabs[1]:
        lineage_panel(
            st, summary.get("lineage") if isinstance(summary.get("lineage"), dict) else {}
        )
    with tabs[2]:
        objective_breakdown = summary.get("objective")
        if isinstance(objective_breakdown, dict) and objective_breakdown:
            st.json(objective_breakdown)
        else:
            st.caption("Objective contributions were not returned.")
        st.caption(f"Objective value: {number(summary.get('objective_value'), decimals=3)}")
    with tabs[3]:
        diagnostics = st.columns(4)
        metric_card(diagnostics[0], "Solver", detail.get("solver_name") or "—")
        metric_card(
            diagnostics[1], "Solve time", duration_seconds(detail.get("solve_time_seconds"))
        )
        metric_card(
            diagnostics[2],
            "Optimality gap",
            percentage(detail.get("optimality_gap"), fraction=True),
        )
        passed = bool(summary.get("constraint_validation_passed"))
        metric_card(
            diagnostics[3],
            "Constraint validation",
            "Passed" if passed else "Review required",
            badge=status_badge("passed" if passed else "not ready"),
        )
        st.caption(
            "The dashboard reports persisted validation results and does not independently "
            "actuate or alter the schedule."
        )
