"""Public liveness and dependency-aware readiness routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from gridmind.api.dependencies import get_health_service
from gridmind.services.health_service import HealthService

router = APIRouter(tags=["health"])


@router.get("/health/live")
def live() -> dict[str, str]:
    return {"status": "alive"}


@router.get("/health/ready")
def ready(service: Annotated[HealthService, Depends(get_health_service)]) -> dict[str, object]:
    return service.readiness()
