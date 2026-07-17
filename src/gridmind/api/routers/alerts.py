"""Alert listing, detail, and controlled lifecycle routes."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends

from gridmind.api.dependencies import PageDependency, get_alert_service
from gridmind.api.schemas import AlertUpdate
from gridmind.api.security import require_api_key
from gridmind.services.alert_service import AlertService

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("")
def list_alerts(
    page: PageDependency,
    service: Annotated[AlertService, Depends(get_alert_service)],
    region: str | None = None,
    target: str | None = None,
    severity: str | None = None,
    status: str | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> dict[str, object]:
    filters = locals().copy()
    filters.pop("page")
    filters.pop("service")
    return service.list(limit=page.limit, offset=page.offset, **filters).as_dict(filters=filters)


@router.get("/{alert_id}")
def alert_detail(
    alert_id: str, service: Annotated[AlertService, Depends(get_alert_service)]
) -> dict[str, object]:
    return service.get(alert_id)


@router.patch("/{alert_id}", dependencies=[Depends(require_api_key)])
def update_alert(
    alert_id: str,
    update: AlertUpdate,
    service: Annotated[AlertService, Depends(get_alert_service)],
) -> dict[str, object]:
    return service.update(alert_id, update.status)
