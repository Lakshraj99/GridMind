"""Read-only, portfolio-quality MLflow Registry metadata view."""

from __future__ import annotations

from typing import Any

from gridmind.dashboard.charts import model_metrics_figure
from gridmind.dashboard.components import (
    badge_html,
    chart_container,
    empty_state,
    formatted_dataframe,
    metric_card,
    page_header,
    section_header,
)
from gridmind.dashboard.formatting import integer, target_label, utc_label
from gridmind.dashboard.state import DashboardContext, default_context, safe_get
from gridmind.dashboard.view_data import available_options, records_frame


def _aliases(value: Any) -> list[str]:
    return [str(alias) for alias in value] if isinstance(value, list) else []


def render(st: Any, client: Any, context: DashboardContext | None = None) -> None:
    context = context or default_context()
    page_header(
        st,
        "Model registry",
        "Understand registered versions, aliases, evaluation context, and serving lineage.",
        refreshed_at=context.refreshed_at,
    )
    payload, payload_error = safe_get(client, "/api/v1/models", limit=500)
    summary, _ = safe_get(client, "/api/v1/models/summary")
    frame = records_frame(
        payload,
        timestamp_columns=("created_at_utc",),
        sort_by="created_at_utc",
        ascending=False,
    )
    cards = st.columns(4)
    metric_card(
        cards[0],
        "Registered models",
        integer((summary or {}).get("registered_models")),
        detail="Distinct registry names",
    )
    metric_card(
        cards[1],
        "Model versions",
        integer((summary or {}).get("model_versions")),
        detail="All registered versions",
    )
    metric_card(
        cards[2],
        "Candidate aliases",
        integer((summary or {}).get("candidate_versions")),
        badge=badge_html("candidate", "info"),
    )
    metric_card(
        cards[3],
        "Champion aliases",
        integer((summary or {}).get("champion_versions")),
        badge=badge_html("champion", "success"),
    )
    if frame.empty:
        empty_state(
            st,
            "No registered models available",
            payload_error
            or "Enable the MLflow backend and register a candidate or champion model.",
        )
        return

    targets = available_options(frame, "target")
    target = str(st.selectbox("Target", ["All", *targets], format_func=target_label))
    selected = frame if target == "All" else frame[frame["target"].astype(str) == target]
    selected = selected.copy().reset_index(drop=True)
    selected["raw_aliases"] = selected.get("aliases")
    selected["aliases"] = selected["raw_aliases"].map(
        lambda value: ", ".join(_aliases(value)) or "unassigned"
    )
    if "created_at_utc" in selected:
        selected["created_at_utc"] = selected["created_at_utc"].map(utc_label)

    section_header(
        st,
        "Registered versions",
        "Artifact locations are intentionally excluded from API and dashboard metadata.",
    )
    visible = [
        column
        for column in (
            "name",
            "version",
            "aliases",
            "target",
            "region",
            "status",
            "created_at_utc",
            "run_id",
        )
        if column in selected
    ]
    formatted_dataframe(st, selected[visible], height=350)

    section_header(
        st,
        "Evaluation comparison",
        "Select one target before interpreting raw WAPE, MAE, or RMSE values.",
    )
    if target == "All" and len(targets) > 1:
        empty_state(
            st,
            "Choose a target to compare metrics",
            "Raw error metrics are not compared across incompatible target scales.",
        )
    else:
        chart_container(st, model_metrics_figure(selected), key="model_metrics")

    section_header(st, "Version detail", "Compact aliases, metrics, and traceable run identity.")
    labels = {
        index: (
            f"{row.get('name', 'model')} v{row.get('version', '—')} · "
            f"{row.get('aliases', 'unassigned')}"
        )
        for index, row in selected.iterrows()
    }
    selected_index = int(
        st.selectbox("Model version", list(labels), format_func=lambda i: labels[i])
    )
    record = selected.loc[selected_index]
    st.markdown(
        " ".join(
            badge_html(alias, "success" if alias == "champion" else "info")
            for alias in _aliases(record.get("raw_aliases"))
        )
        or badge_html("unassigned", "neutral"),
        unsafe_allow_html=True,
    )
    detail_columns = st.columns(3)
    metric_card(detail_columns[0], "Target", target_label(record.get("target")))
    metric_card(detail_columns[1], "Region", record.get("region") or "—")
    metric_card(detail_columns[2], "Created", record.get("created_at_utc") or "—")
    st.caption("Run ID")
    st.code(str(record.get("run_id") or "not provided"), language=None)
    metrics = record.get("training_metrics")
    with st.expander("Training evaluation metrics", expanded=True):
        if isinstance(metrics, dict) and metrics:
            st.json(metrics)
        else:
            st.caption("No compatible training metrics were returned for this version.")
    st.caption("Training and model promotion are intentionally unavailable in this dashboard.")
