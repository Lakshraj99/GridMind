"""GridMind FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from gridmind.api.errors import install_error_handlers
from gridmind.api.middleware import install_middleware
from gridmind.api.routers import alerts, anomalies, dispatch, forecasts, health, models
from gridmind.api.security import require_api_key
from gridmind.config import Settings
from gridmind.observability.logging import configure_application_logging
from gridmind.observability.metrics import ApplicationMetrics
from gridmind.services import (
    AlertService,
    AnomalyService,
    DispatchService,
    ForecastService,
    HealthService,
    ModelService,
)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create an isolated, dependency-injected GridMind API instance."""
    configured = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_application_logging(
            configured.log_level,
            configured.log_format,
            api_key=configured.gridmind_api_key,
            eia_key=configured.eia_api_key,
        )
        yield

    app = FastAPI(
        title=configured.api_title,
        version=configured.api_version,
        root_path=configured.api_root_path,
        lifespan=lifespan,
        description=(
            "Read-only decision support for GridMind forecasts, detections, and simulations."
        ),
    )
    app.state.settings = configured
    app.state.metrics = ApplicationMetrics()
    app.state.forecast_service = ForecastService(
        configured.duckdb_path,
        cache_ttl=configured.api_cache_ttl_seconds,
        metrics=app.state.metrics,
    )
    app.state.anomaly_service = AnomalyService(
        configured.duckdb_path,
        cache_ttl=configured.api_cache_ttl_seconds,
        maximum_rate=configured.anomaly_max_rate,
        metrics=app.state.metrics,
    )
    app.state.alert_service = AlertService(
        configured.duckdb_path, configured, cache_ttl=configured.api_cache_ttl_seconds
    )
    app.state.dispatch_service = DispatchService(configured.duckdb_path)
    app.state.model_service = ModelService(configured)
    app.state.health_service = HealthService(configured)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(configured.api_cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "PATCH"],
        allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
    )
    install_middleware(app)
    install_error_handlers(app)
    app.include_router(health.router)
    versioned = APIRouter(prefix="/api/v1", dependencies=[Depends(require_api_key)])
    for router in (
        forecasts.router,
        anomalies.router,
        alerts.router,
        dispatch.router,
        models.router,
    ):
        versioned.include_router(router)
    app.include_router(versioned)

    @app.get("/metrics", include_in_schema=True)
    def metrics() -> Response:
        if not configured.metrics_enabled:
            return Response(status_code=404)
        return Response(app.state.metrics.render(), media_type="text/plain; version=0.0.4")

    return app
