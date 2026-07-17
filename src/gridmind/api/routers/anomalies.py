"""Human-review anomaly routes."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends

from gridmind.api.dependencies import PageDependency, get_anomaly_service
from gridmind.services.anomaly_service import AnomalyService

router = APIRouter(prefix="/anomalies", tags=["anomalies"])


@router.get("")
def list_anomalies(
    page: PageDependency,
    service: Annotated[AnomalyService, Depends(get_anomaly_service)],
    region: str | None = None,
    target: str | None = None,
    severity: str | None = None,
    detector: str | None = None,
    anomaly_type: str | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> dict[str, object]:
    filters = locals().copy()
    filters.pop("page")
    filters.pop("service")
    return service.list(limit=page.limit, offset=page.offset, **filters).as_dict(filters=filters)


@router.get("/summary")
def anomaly_summary(
    service: Annotated[AnomalyService, Depends(get_anomaly_service)],
) -> dict[str, object]:
    return service.summary()


@router.get("/{anomaly_id}")
def anomaly_detail(
    anomaly_id: str, service: Annotated[AnomalyService, Depends(get_anomaly_service)]
) -> dict[str, object]:
    return service.get(anomaly_id)
