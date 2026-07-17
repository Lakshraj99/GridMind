"""Read-only simulated battery dispatch routes."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends

from gridmind.api.dependencies import PageDependency, get_dispatch_service
from gridmind.services.dispatch_service import DispatchService

router = APIRouter(prefix="/dispatches", tags=["battery dispatch"])


@router.get("")
def list_dispatches(
    page: PageDependency,
    service: Annotated[DispatchService, Depends(get_dispatch_service)],
    region: str | None = None,
    battery_id: str | None = None,
    objective_mode: str | None = None,
    solver_status: str | None = None,
    forecast_origin: datetime | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> dict[str, object]:
    filters = locals().copy()
    filters.pop("page")
    filters.pop("service")
    return service.list(limit=page.limit, offset=page.offset, **filters).as_dict(filters=filters)


@router.get("/{dispatch_run_id}")
def dispatch_detail(
    dispatch_run_id: str,
    service: Annotated[DispatchService, Depends(get_dispatch_service)],
) -> dict[str, object]:
    return service.get(dispatch_run_id)


@router.get("/{dispatch_run_id}/points")
def dispatch_points(
    dispatch_run_id: str,
    page: PageDependency,
    service: Annotated[DispatchService, Depends(get_dispatch_service)],
) -> dict[str, object]:
    return service.points(dispatch_run_id, limit=page.limit, offset=page.offset).as_dict()


@router.get("/{dispatch_run_id}/summary")
def dispatch_summary(
    dispatch_run_id: str,
    service: Annotated[DispatchService, Depends(get_dispatch_service)],
) -> dict[str, object]:
    return service.summary(dispatch_run_id)
