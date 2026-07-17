"""Read-only forecast routes."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from gridmind.api.dependencies import PageDependency, get_forecast_service
from gridmind.services.forecast_service import ForecastService

router = APIRouter(prefix="/forecasts", tags=["forecasts"])


@router.get("")
def list_forecasts(
    page: PageDependency,
    service: Annotated[ForecastService, Depends(get_forecast_service)],
    region: str | None = None,
    target: str | None = None,
    forecast_origin: datetime | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    model_name: str | None = None,
    model_version: str | None = None,
    model_alias: str | None = None,
    weather_mode: str | None = None,
) -> dict[str, object]:
    filters = locals().copy()
    filters.pop("page")
    filters.pop("service")
    result = service.list(limit=page.limit, offset=page.offset, **filters)
    return result.as_dict(filters=filters)


@router.get("/latest")
def latest_forecast(
    service: Annotated[ForecastService, Depends(get_forecast_service)],
    region: str,
    target: str,
    horizon: Annotated[int, Query(ge=1, le=500)] = 24,
    model_alias: str = "champion",
) -> dict[str, object]:
    return service.latest(region=region, target=target, horizon=horizon, model_alias=model_alias)


@router.get("/summary")
def forecast_summary(
    service: Annotated[ForecastService, Depends(get_forecast_service)],
) -> dict[str, object]:
    return service.summary()
