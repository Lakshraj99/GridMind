"""FastAPI dependency providers and pagination validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, cast

from fastapi import Depends, HTTPException, Query, Request

from gridmind.services import (
    AlertService,
    AnomalyService,
    DispatchService,
    ForecastService,
    HealthService,
    ModelService,
)


@dataclass(frozen=True)
class Pagination:
    limit: int
    offset: int


def pagination(
    request: Request,
    limit: Annotated[int | None, Query(ge=1)] = None,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Pagination:
    settings = request.app.state.settings
    selected = limit or settings.api_default_page_size
    if selected > settings.api_max_page_size:
        raise HTTPException(
            status_code=422,
            detail=f"limit must not exceed {settings.api_max_page_size}",
        )
    return Pagination(selected, offset)


PageDependency = Annotated[Pagination, Depends(pagination)]


def get_forecast_service(request: Request) -> ForecastService:
    return cast(ForecastService, request.app.state.forecast_service)


def get_anomaly_service(request: Request) -> AnomalyService:
    return cast(AnomalyService, request.app.state.anomaly_service)


def get_alert_service(request: Request) -> AlertService:
    return cast(AlertService, request.app.state.alert_service)


def get_dispatch_service(request: Request) -> DispatchService:
    return cast(DispatchService, request.app.state.dispatch_service)


def get_model_service(request: Request) -> ModelService:
    return cast(ModelService, request.app.state.model_service)


def get_health_service(request: Request) -> HealthService:
    return cast(HealthService, request.app.state.health_service)
