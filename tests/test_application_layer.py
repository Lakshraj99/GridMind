"""Offline Milestone 6 API, services, authentication, metrics, and Docker tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from gridmind.alerts.lifecycle import AlertManager
from gridmind.alerts.storage import AlertStorage
from gridmind.anomalies.contracts import make_anomaly
from gridmind.anomalies.storage import AnomalyStorage
from gridmind.api.app import create_app
from gridmind.config import Settings
from gridmind.dashboard.api_client import (
    DashboardAPIError,
    DashboardAuthenticationError,
    DashboardConnectionError,
    GridMindAPIClient,
)
from gridmind.dashboard.formatting import megawatts, utc_label
from gridmind.dashboard.pages import alerts as alerts_page
from gridmind.dashboard.pages import anomalies as anomalies_page
from gridmind.dashboard.pages import battery as battery_page
from gridmind.dashboard.pages import forecasts as forecasts_page
from gridmind.dashboard.pages import models as models_page
from gridmind.dashboard.pages import overview as overview_page
from gridmind.data.duckdb_connection import connect_duckdb
from gridmind.data.target_storage import TargetForecastStorage
from gridmind.exceptions import ResourceNotFoundError, StorageError
from gridmind.observability.logging import JSONFormatter
from gridmind.optimization.storage import BatteryDispatchStorage
from gridmind.services.common import TTLCache
from gridmind.services.forecast_service import ForecastService
from gridmind.services.health_service import HealthService
from gridmind.services.model_service import ModelService


def _settings(tmp_path: Path, **values: object) -> Settings:
    defaults: dict[str, object] = {
        "DUCKDB_PATH": tmp_path / "gridmind.duckdb",
        "DATA_QUALITY_DIR": tmp_path / "quality",
        "MLFLOW_ENABLED": False,
        "API_KEY_ENABLED": False,
        "API_CACHE_TTL_SECONDS": 30,
        "LOG_FORMAT": "text",
        "_env_file": None,
    }
    defaults.update(values)
    return Settings(**defaults)


def _seed(path: Path) -> tuple[str, str]:
    origins = pd.Timestamp("2026-07-14T05:00:00Z")
    forecast_rows: list[dict[str, object]] = []
    for region, value in (("PJM", 1000.0), ("MISO", 2000.0)):
        for step in range(1, 4):
            forecast_rows.append(
                {
                    "region": region,
                    "target": "demand_mw",
                    "forecast_origin": origins,
                    "timestamp_utc": origins + pd.Timedelta(hours=step),
                    "forecast_step": step,
                    "predicted_value": value + step,
                    "model_name": "lightgbm_demand",
                    "model_version": "7",
                    "run_id": "safe-run-id",
                    "weather_mode": "realistic_forecast",
                    "created_at_utc": origins,
                }
            )
    TargetForecastStorage(path).upsert(pd.DataFrame(forecast_rows))
    event = make_anomaly(
        region="PJM",
        target="demand_mw",
        timestamp="2026-07-14T06:00:00Z",
        detector_name="rules",
        anomaly_type="demand_spike",
        anomaly_score=80,
        severity="critical",
        observed_value=1200,
        expected_value=1000,
        residual=200,
        explanation="Demand changed abruptly.",
    )
    AnomalyStorage(path).upsert(pd.DataFrame([event]))
    alerts = AlertStorage(path)
    AlertManager(alerts).process(pd.DataFrame([event]))
    alert_id = str(alerts.read_alerts().iloc[0]["alert_id"])
    _seed_dispatch(path, origins)
    return str(event["anomaly_id"]), alert_id


def _seed_dispatch(path: Path, origin: pd.Timestamp) -> None:
    BatteryDispatchStorage(path)
    with connect_duckdb(path) as connection:
        connection.execute(
            "INSERT INTO battery_dispatch_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
            "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                "dispatch-1",
                "PJM",
                "battery-1",
                "peak_shaving",
                origin,
                2.0,
                "scipy-milp-highs",
                "optimal",
                1.0,
                0.01,
                0.0,
                True,
                1100.0,
                1000.0,
                10.0,
                10.0,
                500.0,
                4.0,
                origin,
                json.dumps({"capacity_mwh": 100.0}),
                json.dumps({"model_alias": "champion"}),
                "/private/artifact",
                "mlflow-run",
                json.dumps({"peak_load_mw": 1000.0, "throughput_mwh": 20.0}),
            ],
        )
        for step in (1, 2):
            connection.execute(
                "INSERT INTO battery_dispatch_points VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    "dispatch-1",
                    "PJM",
                    "battery-1",
                    origin,
                    origin + pd.Timedelta(hours=step),
                    step,
                    1100.0,
                    100.0,
                    50.0,
                    150.0,
                    950.0,
                    5.0,
                    0.0,
                    -5.0,
                    50.0,
                    55.0,
                    955.0,
                    30.0,
                    1.0,
                    "charging",
                    "optimal",
                    origin,
                    "{}",
                ],
            )


def test_application_routes_filters_pagination_lifecycle_and_metrics(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    anomaly_id, alert_id = _seed(settings.duckdb_path)
    app = create_app(settings)
    with TestClient(app) as client:
        assert client.get("/health/live").json() == {"status": "alive"}
        ready = client.get("/health/ready").json()
        assert ready["status"] == "ready"
        assert client.get("/openapi.json").json()["info"]["version"] == "0.6.0"

        response = client.get(
            "/api/v1/forecasts",
            params={"region": "PJM", "limit": 2, "offset": 1},
            headers={"X-Request-ID": "request-123"},
        )
        assert response.headers["X-Request-ID"] == "request-123"
        body = response.json()
        assert body["pagination"] == {
            "limit": 2,
            "offset": 1,
            "returned": 2,
            "total": 3,
            "has_more": False,
        }
        assert {row["region"] for row in body["items"]} == {"PJM"}
        assert all(row["timestamp_utc"].endswith("Z") for row in body["items"])
        assert body["items"][0]["lineage"]["run_id"] == "safe-run-id"
        latest = client.get(
            "/api/v1/forecasts/latest",
            params={"region": "PJM", "target": "demand_mw", "horizon": 3},
        ).json()
        assert len(latest["items"]) == 3
        assert client.get("/api/v1/forecasts/summary").json()["total_rows"] == 6
        assert (
            client.get("/api/v1/forecasts", params={"region": "' OR 1=1 --"}).json()["pagination"][
                "total"
            ]
            == 0
        )

        anomalies = client.get("/api/v1/anomalies", params={"severity": "critical"}).json()
        assert anomalies["pagination"]["total"] == 1
        assert anomalies["items"][0]["review_status"].startswith("detection")
        assert client.get(f"/api/v1/anomalies/{anomaly_id}").status_code == 200
        assert client.get("/api/v1/anomalies/summary").json()["groups"][0]["count"] == 1

        alert = client.get(f"/api/v1/alerts/{alert_id}").json()
        assert (
            client.get("/api/v1/alerts", params={"status": "open"}).json()["pagination"]["total"]
            == 1
        )
        history_count = len(alert["history"])
        assert (
            client.patch(f"/api/v1/alerts/{alert_id}", json={"status": "acknowledged"}).status_code
            == 200
        )
        assert (
            client.patch(f"/api/v1/alerts/{alert_id}", json={"status": "acknowledged"}).status_code
            == 200
        )
        updated = client.get(f"/api/v1/alerts/{alert_id}").json()
        assert updated["status"] == "acknowledged"
        assert len(updated["history"]) == history_count + 1

        runs = client.get("/api/v1/dispatches", params={"region": "PJM"}).json()
        assert runs["pagination"]["total"] == 1
        assert "artifact_path" not in runs["items"][0]
        assert (
            client.get("/api/v1/dispatches/dispatch-1").json()["lineage"]["model_alias"]
            == "champion"
        )
        assert len(client.get("/api/v1/dispatches/dispatch-1/points").json()["items"]) == 2
        dispatch_summary = client.get("/api/v1/dispatches/dispatch-1/summary").json()
        assert dispatch_summary["peak_reduction_mw"] == 100
        assert "does not control" in dispatch_summary["disclaimer"]

        assert "gridmind_http_requests_total" in client.get("/metrics").text
        unknown = client.get("/does-not-exist").json()["error"]
        assert unknown["code"] == "http_error" and unknown["request_id"]
        invalid = client.get("/api/v1/forecasts", params={"limit": 999}).json()["error"]
        assert invalid["code"] == "http_error"
        missing = client.get("/api/v1/anomalies/missing").json()["error"]
        assert missing["code"] == "resource_not_found"


def test_authentication_public_liveness_readiness_and_secret_redaction(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    secret = "local-api-secret"
    settings = _settings(tmp_path, API_KEY_ENABLED=True, GRIDMIND_API_KEY=secret)
    _seed(settings.duckdb_path)
    with TestClient(create_app(settings)) as client:
        assert client.get("/health/live").status_code == 200
        missing = client.get("/api/v1/forecasts")
        assert missing.status_code == 401
        assert missing.headers["WWW-Authenticate"] == "ApiKey"
        assert client.get("/api/v1/forecasts", headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.get("/api/v1/forecasts", headers={"X-API-Key": secret}).status_code == 200
    assert secret not in capsys.readouterr().err


def test_readiness_reports_missing_database_and_table_without_network(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        body = client.get("/health/ready").json()
        assert body["status"] == "not_ready"
        assert body["components"]["duckdb"]["ready"] is False
    with connect_duckdb(settings.duckdb_path):
        pass
    with TestClient(create_app(settings)) as client:
        assert (
            "target_forecasts"
            in client.get("/health/ready").json()["components"]["duckdb"]["missing_tables"]
        )


def test_readiness_reports_invalid_mlflow_without_external_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path, MLFLOW_ENABLED=True)
    _seed(settings.duckdb_path)

    class BrokenClient:
        def __init__(self, **_values: object) -> None:
            pass

        def search_experiments(self, **_values: object) -> None:
            raise RuntimeError("backend unavailable")

    monkeypatch.setattr("gridmind.services.health_service.MlflowClient", BrokenClient)
    readiness = HealthService(settings).readiness()
    assert readiness["status"] == "not_ready"
    assert readiness["components"]["mlflow"] == {
        "ready": False,
        "enabled": True,
        "message": "MLflow backend is unavailable.",
    }


def test_latest_forecast_skips_newer_gapped_series(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _seed(settings.duckdb_path)
    origin = pd.Timestamp("2026-07-15T00:00:00Z")
    rows = []
    for step in (1, 3, 4):
        rows.append(
            {
                "region": "PJM",
                "target": "demand_mw",
                "forecast_origin": origin,
                "timestamp_utc": origin + pd.Timedelta(hours=step),
                "forecast_step": step,
                "predicted_value": 1000 + step,
                "model_name": "gapped-model",
                "model_version": "8",
                "run_id": "gapped-run",
                "weather_mode": "realistic_forecast",
                "created_at_utc": origin,
            }
        )
    TargetForecastStorage(settings.duckdb_path).upsert(pd.DataFrame(rows))
    service = ForecastService(settings.duckdb_path)
    latest = service.latest(region="PJM", target="demand_mw", horizon=3, model_alias="champion")
    assert latest["lineage"]["model_version"] == "7"
    with pytest.raises(ResourceNotFoundError, match="complete contiguous"):
        service.latest(region="PJM", target="demand_mw", horizon=4, model_alias="champion")


def test_api_config_validation_disabled_metrics_and_standard_validation(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="GRIDMIND_API_KEY"):
        _settings(tmp_path, API_KEY_ENABLED=True)
    with pytest.raises(ValidationError, match="API_MAX_PAGE_SIZE"):
        _settings(tmp_path, API_DEFAULT_PAGE_SIZE=10, API_MAX_PAGE_SIZE=5)
    with pytest.raises(ValidationError, match="Invalid CORS"):
        _settings(tmp_path, API_CORS_ORIGINS="file:///tmp")
    with pytest.raises(ValidationError):
        _settings(tmp_path, API_PORT=70000)
    settings = _settings(tmp_path, METRICS_ENABLED=False)
    with TestClient(create_app(settings)) as client:
        assert client.get("/metrics").status_code == 404
        error = client.get("/api/v1/forecasts", params={"offset": -1}).json()["error"]
        assert error["code"] == "validation_error"

    class Broken:
        def list(self, **_filters: object) -> None:
            raise StorageError("safe storage failure")

    app = create_app(settings)
    app.state.forecast_service = Broken()
    with TestClient(app, raise_server_exceptions=False) as client:
        failed = client.get("/api/v1/forecasts").json()["error"]
        assert failed["code"] == "gridmind_error"
        assert "Traceback" not in failed["message"]


def test_model_service_returns_only_safe_registry_metadata(tmp_path: Path) -> None:
    version = SimpleNamespace(
        version="2",
        status="READY",
        run_id="run-2",
        creation_timestamp=123,
        tags={"target": "demand_mw"},
    )
    model = SimpleNamespace(name="demand-model", aliases={"champion": "2"})

    class Client:
        def search_registered_models(self) -> list[Any]:
            return [model]

        def search_model_versions(self, query: str) -> list[Any]:
            assert query == "name='demand-model'"
            return [version]

        def get_run(self, run_id: str) -> Any:
            assert run_id == "run-2"
            return SimpleNamespace(data=SimpleNamespace(metrics={"wape": 0.1}))

    service = ModelService(_settings(tmp_path, MLFLOW_ENABLED=True), client=Client())  # type: ignore[arg-type]
    item = service.list()[0]
    assert item["aliases"] == ["champion"]
    assert item["training_metrics"] == {"wape": 0.1}
    assert not any("path" in key or "uri" in key for key in item)
    assert service.get("demand-model")["versions"][0]["run_id"] == "run-2"
    assert service.summary()["champion_versions"] == 1
    with pytest.raises(Exception, match="missing"):
        service.get("missing")

    settings = _settings(tmp_path)
    app = create_app(settings)
    app.state.model_service = service
    with TestClient(app) as api:
        assert api.get("/api/v1/models").json()["items"][0]["name"] == "demand-model"
        assert api.get("/api/v1/models/summary").json()["model_versions"] == 1
        assert api.get("/api/v1/models/demand-model").status_code == 200
        assert api.get("/api/v1/models/missing").status_code == 404


def test_dashboard_client_success_auth_errors_retry_and_patch() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/retry" and len([c for c in calls if c.url.path == "/retry"]) == 1:
            return httpx.Response(503, json={"error": {"message": "retry"}})
        if request.url.path == "/auth":
            return httpx.Response(401, json={})
        if request.url.path == "/error":
            return httpx.Response(400, json={"error": {"message": "bad request"}})
        return httpx.Response(200, json={"ok": True})

    client = GridMindAPIClient(
        "http://api", api_key="placeholder", transport=httpx.MockTransport(handler)
    )
    assert client.get("/ok", region="PJM") == {"ok": True}
    assert calls[0].headers["X-API-Key"] == "placeholder"
    assert client.get("/retry") == {"ok": True}
    assert client.patch("/alerts/1", {"status": "resolved"}) == {"ok": True}
    with pytest.raises(DashboardAuthenticationError):
        client.get("/auth")
    with pytest.raises(DashboardAPIError, match="bad request"):
        client.get("/error")
    with pytest.raises(DashboardAPIError, match="invalid response"):
        GridMindAPIClient(
            "http://api",
            transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=[])),
        ).get("/invalid")
    client.close()

    def timeout(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout")

    unavailable = GridMindAPIClient(
        "http://api", get_retries=1, transport=httpx.MockTransport(timeout)
    )
    with pytest.raises(DashboardConnectionError):
        unavailable.get("/timeout")
    with pytest.raises(DashboardConnectionError):
        unavailable.patch("/timeout", {})
    unavailable.close()


def test_cache_formatting_logging_and_docker_configuration(tmp_path: Path) -> None:
    now = [0.0]
    cache = TTLCache(10, clock=lambda: now[0])
    calls = [0]

    def build() -> int:
        calls[0] += 1
        return calls[0]

    assert cache.get_or_create("key", build) == (1, False)
    assert cache.get_or_create("key", build) == (1, True)
    now[0] = 11
    assert cache.get_or_create("key", build) == (2, False)
    cache.clear()
    assert utc_label("2026-01-01T05:30:00+05:30") == "2026-01-01T00:00:00Z"
    assert megawatts(1234.56) == "1,234.6 MW"
    record = __import__("logging").LogRecord("test", 20, "", 1, "key=%s", ("secret",), None)
    formatted = JSONFormatter(("secret",)).format(record)
    assert "secret" not in formatted and "[REDACTED]" in formatted
    root = Path(__file__).parents[1]
    compose = (root / "docker-compose.yml").read_text(encoding="utf-8")
    assert "gridmind-api:" in compose and "condition: service_healthy" in compose
    for dockerfile in ("Dockerfile.api", "Dockerfile.dashboard"):
        text = (root / dockerfile).read_text(encoding="utf-8")
        assert "USER gridmind" in text and "HEALTHCHECK" in text
    assert ".env" in (root / ".dockerignore").read_text(encoding="utf-8")
    __import__("gridmind.api.entrypoint")
    __import__("gridmind.dashboard.entrypoint")


def test_dashboard_pages_render_data_and_empty_states() -> None:
    class UI:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def __getattr__(self, name: str) -> Any:
            def record(*args: object, **_kwargs: object) -> Any:
                self.messages.append(f"{name}:{args[0] if args else ''}")
                if name == "columns":
                    return [self, self, self]
                if name == "text_input":
                    return args[1]
                if name == "selectbox":
                    return args[1][0]  # type: ignore[index]
                if name in {"checkbox", "button"}:
                    return False
                return None

            return record

    class Client:
        def get(self, path: str, **_params: object) -> dict[str, Any]:
            if path == "/api/v1/forecasts/summary":
                return {"total_rows": 6}
            if path == "/api/v1/forecasts/latest":
                return {
                    "items": [{"timestamp_utc": "2026-01-01T01:00:00Z", "predicted_value": 100}],
                    "lineage": {"model_version": "1"},
                }
            if path == "/api/v1/anomalies":
                return {
                    "items": [
                        {
                            "timestamp_utc": "2026-01-01T01:00:00Z",
                            "anomaly_score": 10,
                            "severity": "warning",
                            "target": "demand_mw",
                            "anomaly_type": "spike",
                            "detector_name": "rules",
                        }
                    ]
                }
            if path == "/api/v1/anomalies/summary":
                return {"calibration_warnings": ["review calibration"]}
            if path == "/api/v1/alerts":
                return {
                    "items": [{"alert_id": "alert-1", "status": "open"}],
                    "pagination": {"total": 1},
                }
            if path == "/api/v1/dispatches":
                return {"items": [{"dispatch_run_id": "dispatch-1"}]}
            if path.endswith("/points"):
                return {
                    "items": [
                        {
                            "timestamp_utc": "2026-01-01T01:00:00Z",
                            "net_load_before_battery_mw": 100,
                            "net_load_after_battery_mw": 90,
                            "soc_end_mwh": 50,
                        }
                    ]
                }
            if path.endswith("/summary") and "dispatches" in path:
                return {"peak_reduction_mw": 10.0}
            if path == "/api/v1/models":
                return {"items": [{"name": "model", "version": "1"}]}
            if path == "/api/v1/models/summary":
                return {"registered_models": 1}
            raise AssertionError(path)

        def patch(self, _path: str, _payload: dict[str, object]) -> dict[str, Any]:
            raise AssertionError("Unconfirmed status changes must not be sent.")

    ui = UI()
    client = Client()
    for renderer in (
        overview_page.render,
        forecasts_page.render,
        anomalies_page.render,
        alerts_page.render,
        battery_page.render,
        models_page.render,
    ):
        renderer(ui, client)
    assert any(message.startswith("plotly_chart") for message in ui.messages)
    assert any("human review" in message for message in ui.messages)

    class EmptyClient(Client):
        def get(self, path: str, **_params: object) -> dict[str, Any]:
            return {"items": [], "pagination": {"total": 0}}

    empty = EmptyClient()
    for renderer in (
        forecasts_page.render,
        anomalies_page.render,
        alerts_page.render,
        battery_page.render,
        models_page.render,
    ):
        renderer(ui, empty)
    assert sum(message.startswith("info:No") for message in ui.messages) >= 5
