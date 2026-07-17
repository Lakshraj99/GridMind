"""Theme-consistent Plotly figure construction for dashboard pages."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from gridmind.dashboard.formatting import target_label

COLORS = {
    "blue": "#55A6FF",
    "cyan": "#46C2C8",
    "green": "#3ECF8E",
    "amber": "#F5B942",
    "red": "#F06A6A",
    "muted": "#93A4BD",
    "info": "#55A6FF",
    "warning": "#F5B942",
    "critical": "#F06A6A",
}


def style_figure(
    figure: go.Figure,
    *,
    height: int = 390,
    y_title: str = "",
    show_legend: bool = True,
) -> go.Figure:
    figure.update_layout(
        template="plotly_dark",
        height=height,
        margin={"l": 42, "r": 22, "t": 32, "b": 42},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#DCE5F2", "family": "Inter, system-ui, sans-serif", "size": 12},
        hovermode="x unified",
        showlegend=show_legend,
        legend={"orientation": "h", "y": 1.08, "x": 0},
        xaxis={
            "title": "UTC",
            "gridcolor": "rgba(147,164,189,.12)",
            "rangeslider": {"visible": False},
        },
        yaxis={
            "title": y_title,
            "gridcolor": "rgba(147,164,189,.12)",
            "zerolinecolor": "rgba(147,164,189,.28)",
        },
    )
    return figure


def empty_figure(message: str, *, height: int = 330) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(
        text=message, x=0.5, y=0.5, showarrow=False, font={"color": COLORS["muted"]}
    )
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return style_figure(figure, height=height, show_legend=False)


def forecast_figure(frame: pd.DataFrame, *, target: str) -> go.Figure:
    if frame.empty or not {"timestamp_utc", "predicted_value"}.issubset(frame.columns):
        return empty_figure("No forecast points match the current selection.")
    ordered = frame.copy()
    ordered["timestamp_utc"] = pd.to_datetime(
        ordered["timestamp_utc"], utc=True, errors="coerce", format="mixed"
    )
    ordered = ordered.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=ordered["timestamp_utc"],
            y=ordered["predicted_value"],
            name="Forecast",
            mode="lines+markers",
            line={"color": COLORS["blue"], "width": 2.4},
            marker={"size": 5},
            hovertemplate="%{x|%Y-%m-%d %H:%MZ}<br>Forecast %{y:,.1f} MW<extra></extra>",
        )
    )
    if "actual_value" in ordered and ordered["actual_value"].notna().any():
        figure.add_trace(
            go.Scatter(
                x=ordered["timestamp_utc"],
                y=ordered["actual_value"],
                name="Actual",
                mode="lines",
                line={"color": COLORS["green"], "width": 2},
                hovertemplate="%{x|%Y-%m-%d %H:%MZ}<br>Actual %{y:,.1f} MW<extra></extra>",
            )
        )
    result = style_figure(figure, y_title=target_label(target, include_unit=True))
    result.update_xaxes(
        rangeselector={
            "buttons": [
                {"count": 12, "label": "12h", "step": "hour", "stepmode": "backward"},
                {"count": 1, "label": "1d", "step": "day", "stepmode": "backward"},
                {"step": "all", "label": "All"},
            ]
        }
    )
    return result


def combined_forecast_figure(series: Mapping[str, pd.DataFrame]) -> go.Figure:
    figure = go.Figure()
    palette = [COLORS["blue"], COLORS["cyan"], COLORS["green"]]
    for index, (target, frame) in enumerate(series.items()):
        if frame.empty or not {"timestamp_utc", "predicted_value"}.issubset(frame.columns):
            continue
        ordered = frame.copy()
        ordered["timestamp_utc"] = pd.to_datetime(
            ordered["timestamp_utc"], utc=True, errors="coerce", format="mixed"
        )
        ordered = ordered.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
        figure.add_trace(
            go.Scatter(
                x=ordered["timestamp_utc"],
                y=ordered["predicted_value"],
                name=target_label(target),
                mode="lines",
                line={"color": palette[index % len(palette)], "width": 2.2},
                hovertemplate="%{x|%Y-%m-%d %H:%MZ}<br>%{y:,.1f} MW<extra></extra>",
            )
        )
    return (
        style_figure(figure, y_title="Forecast MW")
        if figure.data
        else empty_figure("Demand and net-load forecasts are unavailable.")
    )


def anomaly_timeline(frame: pd.DataFrame) -> go.Figure:
    required = {"timestamp_utc", "anomaly_score", "severity"}
    if frame.empty or not required.issubset(frame.columns):
        return empty_figure("No anomaly detections match the current filters.")
    ordered = frame.copy()
    ordered["timestamp_utc"] = pd.to_datetime(
        ordered["timestamp_utc"], utc=True, errors="coerce", format="mixed"
    )
    ordered = ordered.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    figure = go.Figure()
    for severity in ("info", "warning", "critical"):
        selected = ordered[ordered["severity"].astype(str).str.lower() == severity]
        if selected.empty:
            continue
        custom = selected.reindex(columns=["target", "anomaly_type", "detector_name"]).fillna("—")
        figure.add_trace(
            go.Scatter(
                x=selected["timestamp_utc"],
                y=selected["anomaly_score"],
                name=severity.title(),
                mode="markers",
                marker={"color": COLORS[severity], "size": 8, "opacity": 0.82},
                customdata=custom,
                hovertemplate=(
                    "%{x|%Y-%m-%d %H:%MZ}<br>Score %{y:.1f}<br>Target %{customdata[0]}"
                    "<br>Type %{customdata[1]}<br>Detector %{customdata[2]}<extra></extra>"
                ),
            )
        )
    return style_figure(figure, y_title="Anomaly score")


def category_bar(counts: Mapping[str, int], *, title: str = "Count") -> go.Figure:
    if not counts:
        return empty_figure("No category data is available.", height=310)
    ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    figure = go.Figure(
        go.Bar(
            x=[item[0].replace("_", " ").title() for item in ordered],
            y=[item[1] for item in ordered],
            marker_color=[COLORS.get(item[0].lower(), COLORS["blue"]) for item in ordered],
            hovertemplate="%{x}<br>%{y:,} detections<extra></extra>",
        )
    )
    return style_figure(figure, height=310, y_title=title, show_legend=False)


def daily_rate_figure(frame: pd.DataFrame) -> go.Figure:
    if frame.empty or "timestamp_utc" not in frame:
        return empty_figure("Daily anomaly activity is unavailable.", height=310)
    work = frame.copy()
    work["timestamp_utc"] = pd.to_datetime(
        work["timestamp_utc"], utc=True, errors="coerce", format="mixed"
    )
    work = work.dropna(subset=["timestamp_utc"])
    daily = work.groupby(work["timestamp_utc"].dt.floor("D")).size().rename("detections")
    figure = go.Figure(
        go.Scatter(
            x=daily.index,
            y=daily.values,
            mode="lines+markers",
            line={"color": COLORS["cyan"], "width": 2},
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:,} detections<extra></extra>",
        )
    )
    return style_figure(figure, height=310, y_title="Detections / day", show_legend=False)


def dispatch_load_figure(points: pd.DataFrame) -> go.Figure:
    required = {"timestamp_utc", "net_load_before_battery_mw", "net_load_after_battery_mw"}
    if points.empty or not required.issubset(points.columns):
        return empty_figure("Dispatch load points are unavailable.")
    ordered = points.sort_values("timestamp_utc")
    figure = go.Figure()
    for column, name, color in (
        ("net_load_before_battery_mw", "Before battery", COLORS["muted"]),
        ("net_load_after_battery_mw", "After battery", COLORS["blue"]),
    ):
        figure.add_trace(
            go.Scatter(
                x=ordered["timestamp_utc"],
                y=ordered[column],
                name=name,
                mode="lines",
                line={"color": color, "width": 2.2},
                hovertemplate=f"%{{x|%Y-%m-%d %H:%MZ}}<br>{name} %{{y:,.1f}} MW<extra></extra>",
            )
        )
    return style_figure(figure, y_title="Net load MW")


def dispatch_power_figure(points: pd.DataFrame) -> go.Figure:
    required = {"timestamp_utc", "charge_mw", "discharge_mw"}
    if points.empty or not required.issubset(points.columns):
        return empty_figure("Charge and discharge points are unavailable.", height=330)
    ordered = points.sort_values("timestamp_utc")
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=ordered["timestamp_utc"],
            y=-ordered["charge_mw"].astype(float),
            name="Charge",
            marker_color=COLORS["cyan"],
            hovertemplate="%{x|%Y-%m-%d %H:%MZ}<br>Charge %{customdata:,.1f} MW<extra></extra>",
            customdata=ordered["charge_mw"],
        )
    )
    figure.add_trace(
        go.Bar(
            x=ordered["timestamp_utc"],
            y=ordered["discharge_mw"],
            name="Discharge",
            marker_color=COLORS["amber"],
            hovertemplate="%{x|%Y-%m-%d %H:%MZ}<br>Discharge %{y:,.1f} MW<extra></extra>",
        )
    )
    result = style_figure(figure, height=330, y_title="Battery power MW")
    result.update_layout(barmode="relative")
    return result


def soc_figure(points: pd.DataFrame) -> go.Figure:
    if points.empty or not {"timestamp_utc", "soc_end_mwh"}.issubset(points.columns):
        return empty_figure("State-of-charge points are unavailable.", height=330)
    ordered = points.sort_values("timestamp_utc")
    figure = go.Figure(
        go.Scatter(
            x=ordered["timestamp_utc"],
            y=ordered["soc_end_mwh"],
            name="End SOC",
            mode="lines+markers",
            fill="tozeroy",
            fillcolor="rgba(62,207,142,.10)",
            line={"color": COLORS["green"], "width": 2.2},
            hovertemplate="%{x|%Y-%m-%d %H:%MZ}<br>SOC %{y:,.1f} MWh<extra></extra>",
        )
    )
    return style_figure(figure, height=330, y_title="State of charge MWh", show_legend=False)


def model_metrics_figure(frame: pd.DataFrame) -> go.Figure:
    if frame.empty or "training_metrics" not in frame:
        return empty_figure("Comparable evaluation metrics are unavailable.")
    rows: list[dict[str, Any]] = []
    for _, record in frame.iterrows():
        metrics = record.get("training_metrics") or {}
        if not isinstance(metrics, dict):
            continue
        for metric in ("wape", "mae", "rmse"):
            if metric in metrics:
                rows.append(
                    {
                        "version": f"{record.get('name', 'model')} v{record.get('version', '—')}",
                        "metric": metric.upper(),
                        "value": metrics[metric],
                    }
                )
    if not rows:
        return empty_figure("WAPE, MAE, or RMSE metrics were not returned.")
    metric_frame = pd.DataFrame(rows)
    figure = go.Figure()
    for metric_value, selected in metric_frame.groupby("metric"):
        metric_name = str(metric_value)
        figure.add_trace(
            go.Bar(
                x=selected["version"],
                y=selected["value"],
                name=metric_name,
                hovertemplate=f"%{{x}}<br>{metric_name} %{{y:,.3f}}<extra></extra>",
            )
        )
    result = style_figure(figure, y_title="Metric value")
    result.update_layout(barmode="group")
    return result
