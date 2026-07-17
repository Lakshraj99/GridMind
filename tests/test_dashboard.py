"""Offline tests for dashboard presentation, data preparation, and API safety."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import httpx
import pandas as pd
import pytest

from gridmind.dashboard.api_client import (
    DashboardAPIError,
    DashboardAuthenticationError,
    DashboardConnectionError,
    GridMindAPIClient,
)
from gridmind.dashboard.charts import (
    anomaly_timeline,
    dispatch_power_figure,
    forecast_figure,
    model_metrics_figure,
)
from gridmind.dashboard.components import (
    badge_html,
    empty_state,
    freshness_badge,
    severity_badge,
    status_badge,
)
from gridmind.dashboard.formatting import (
    MISSING,
    currency,
    duration_seconds,
    megawatt_hours,
    megawatts,
    number,
    percentage,
    target_label,
    utc_label,
)
from gridmind.dashboard.pages.alerts import ALLOWED_TRANSITIONS
from gridmind.dashboard.pages.forecasts import _selected_forecast
from gridmind.dashboard.view_data import (
    effective_anomaly_rate,
    parse_mapping,
    records_frame,
)


class FakeStreamlit:
    def __init__(self) -> None:
        self.markdown_calls: list[tuple[str, bool]] = []

    def markdown(self, value: str, *, unsafe_allow_html: bool = False) -> None:
        self.markdown_calls.append((value, unsafe_allow_html))


def test_formatting_helpers_are_consistent_and_defensive() -> None:
    assert number(12345.67) == "12,345.7"
    assert number(1_250_000, compact=True) == "1.2M"
    assert megawatts(1250) == "1,250.0 MW"
    assert megawatt_hours(250) == "250.0 MWh"
    assert percentage(0.125, fraction=True) == "12.5%"
    assert currency(9.5) == "$9.50"
    assert duration_seconds(0.25) == "250 ms"
    assert duration_seconds(65) == "1m 5s"
    assert utc_label("2026-07-14 11:30:00+05:30") == "2026-07-14T06:00:00Z"
    assert target_label("net_load_mw", include_unit=True) == "Net load MW"
    assert megawatts(None) == MISSING
    assert number(float("nan")) == MISSING
    assert utc_label("not-a-date") == MISSING


def test_badges_escape_content_and_map_status_tones() -> None:
    assert "gm-success" in status_badge("ready")
    assert "gm-critical" in status_badge("unavailable")
    assert "gm-warning" in severity_badge("warning")
    escaped = badge_html("<script>alert(1)</script>", "info")
    assert "<script>" not in escaped
    assert "&lt;script&gt;" in escaped


def test_freshness_badges_use_utc_age_and_handle_missing_values() -> None:
    now = "2026-07-14T06:00:00Z"
    assert "gm-success" in freshness_badge("2026-07-14T05:00:00Z", now=now)
    assert "gm-warning" in freshness_badge("2026-07-13T18:00:00Z", now=now)
    assert "gm-critical" in freshness_badge("2026-07-10T00:00:00Z", now=now)
    assert "unknown freshness" in freshness_badge(None, now=now)


def test_empty_state_renders_safe_owned_markup() -> None:
    fake = FakeStreamlit()
    empty_state(fake, "No <records>", "Try another filter.")
    markup, unsafe = fake.markdown_calls[0]
    assert unsafe is True
    assert "gm-state" in markup
    assert "&lt;records&gt;" in markup


def test_api_client_maps_authentication_and_structured_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth":
            return httpx.Response(401, request=request)
        return httpx.Response(
            422,
            request=request,
            json={"error": {"message": "The selected record is invalid."}},
        )

    client = GridMindAPIClient("http://test", transport=httpx.MockTransport(handler), get_retries=0)
    with pytest.raises(DashboardAuthenticationError, match="rejected"):
        client.get("/auth")
    with pytest.raises(DashboardAPIError, match="selected record"):
        client.get("/invalid")
    client.close()


def test_api_client_maps_timeouts_and_invalid_payloads() -> None:
    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("hidden transport detail", request=request)

    timeout_client = GridMindAPIClient(
        "http://test", transport=httpx.MockTransport(timeout_handler), get_retries=0
    )
    with pytest.raises(DashboardConnectionError, match="unavailable") as exc_info:
        timeout_client.get("/slow")
    assert "hidden transport detail" not in str(exc_info.value)
    timeout_client.close()

    invalid_client = GridMindAPIClient(
        "http://test",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, text="no-json", request=request)
        ),
        get_retries=0,
    )
    with pytest.raises(DashboardAPIError, match="invalid response"):
        invalid_client.get("/invalid-json")
    invalid_client.close()


def test_api_client_caches_gets_per_rerun_and_invalidates_after_patch() -> None:
    calls = {"get": 0, "patch": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls[request.method.lower()] += 1
        return httpx.Response(200, request=request, json={"status": "ok"})

    client = GridMindAPIClient("http://test", transport=httpx.MockTransport(handler), get_retries=0)
    assert client.get("/resource", region="PJM") == {"status": "ok"}
    assert client.get("/resource", region="PJM") == {"status": "ok"}
    assert calls["get"] == 1
    client.patch("/resource/1", {"status": "resolved"})
    client.get("/resource", region="PJM")
    assert calls == {"get": 2, "patch": 1}
    client.close()


def test_records_frame_normalizes_utc_sorts_and_accepts_missing_fields() -> None:
    payload = {
        "items": [
            {"timestamp_utc": "2026-07-14T07:00:00Z", "predicted_value": 2},
            {"timestamp_utc": "2026-07-14 11:30:00+05:30", "predicted_value": 1},
        ]
    }
    frame = records_frame(payload, timestamp_columns=("timestamp_utc",), sort_by="timestamp_utc")
    assert frame["predicted_value"].tolist() == [1, 2]
    assert str(frame["timestamp_utc"].dt.tz) == "UTC"
    assert records_frame({"items": None}).empty
    assert parse_mapping(None) == {}
    assert parse_mapping("not-json") == {}


def test_effective_anomaly_rate_uses_returned_target_hour_window() -> None:
    frame = pd.DataFrame(
        {
            "timestamp_utc": ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"],
            "target": ["demand_mw", "demand_mw"],
        }
    )
    assert effective_anomaly_rate(frame) == 1.0
    assert effective_anomaly_rate(pd.DataFrame()) is None


def test_forecast_selection_is_chronological_and_preserves_lineage() -> None:
    history = pd.DataFrame(
        [
            {
                "timestamp_utc": pd.Timestamp("2026-01-01T02:00:00Z"),
                "forecast_origin": pd.Timestamp("2026-01-01T00:00:00Z"),
                "predicted_value": 2,
                "lineage": {"run_id": "run-1"},
            },
            {
                "timestamp_utc": pd.Timestamp("2026-01-01T01:00:00Z"),
                "forecast_origin": pd.Timestamp("2026-01-01T00:00:00Z"),
                "predicted_value": 1,
                "lineage": {"run_id": "run-1"},
            },
        ]
    )
    selected, lineage = _selected_forecast(history, None, "2026-01-01T00:00:00Z", 2)
    assert selected["predicted_value"].tolist() == [1, 2]
    assert lineage == {"run_id": "run-1"}


def test_plotly_forecast_and_anomaly_figures_sort_chronologically() -> None:
    forecasts = pd.DataFrame(
        {
            "timestamp_utc": ["2026-01-01T02:00:00Z", "2026-01-01T01:00:00Z"],
            "predicted_value": [2.0, 1.0],
        }
    )
    forecast = forecast_figure(forecasts, target="demand_mw")
    assert list(forecast.data[0].y) == [1.0, 2.0]
    assert forecast.layout.hovermode == "x unified"

    anomalies = pd.DataFrame(
        {
            "timestamp_utc": ["2026-01-01T02:00:00Z", "2026-01-01T01:00:00Z"],
            "anomaly_score": [5.0, 4.0],
            "severity": ["warning", "warning"],
        }
    )
    anomaly = anomaly_timeline(anomalies)
    assert list(anomaly.data[0].y) == [4.0, 5.0]


def test_dispatch_chart_places_charge_below_zero() -> None:
    points = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(
                ["2026-01-01T01:00:00Z", "2026-01-01T02:00:00Z"], utc=True
            ),
            "charge_mw": [10.0, 0.0],
            "discharge_mw": [0.0, 8.0],
        }
    )
    figure = dispatch_power_figure(points)
    assert list(figure.data[0].y) == [-10.0, -0.0]
    assert list(figure.data[1].y) == [0.0, 8.0]


def test_model_metric_chart_handles_missing_optional_metrics() -> None:
    missing = model_metrics_figure(pd.DataFrame([{"name": "demand", "version": "1"}]))
    assert not missing.data
    available = model_metrics_figure(
        pd.DataFrame(
            [
                {
                    "name": "demand",
                    "version": "1",
                    "training_metrics": {"wape": 4.2, "mae": 120.0},
                }
            ]
        )
    )
    assert {trace.name for trace in available.data} == {"WAPE", "MAE"}


def test_alert_transitions_never_offer_a_no_op() -> None:
    for current, transitions in ALLOWED_TRANSITIONS.items():
        assert current not in transitions


def test_dashboard_source_has_no_database_or_hardcoded_result_dependencies() -> None:
    root = Path(__file__).parents[1] / "src" / "gridmind" / "dashboard"
    source = "\n".join(path.read_text() for path in root.rglob("*.py"))
    assert "import duckdb" not in source
    assert "from gridmind.data" not in source
    for production_metric in ("4.03", "4.23", "47.34", "17.72", "114,506.428"):
        assert production_metric not in source


def test_dashboard_imports_without_training_dependencies() -> None:
    code = """
import importlib.abc
import sys
blocked = {'catboost', 'lightgbm', 'mlflow', 'mlforecast', 'optuna', 'scipy', 'shap', 'duckdb'}
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise ImportError(f'blocked optional dependency: {fullname}')
        return None
sys.meta_path.insert(0, Blocker())
import gridmind.dashboard.app
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
