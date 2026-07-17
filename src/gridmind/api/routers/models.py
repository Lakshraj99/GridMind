"""Read-only safe MLflow Registry metadata routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from gridmind.api.dependencies import PageDependency, get_model_service
from gridmind.services.model_service import ModelService

router = APIRouter(prefix="/models", tags=["models"])


@router.get("")
def list_models(
    page: PageDependency,
    service: Annotated[ModelService, Depends(get_model_service)],
) -> dict[str, object]:
    items = service.list()
    selected = items[page.offset : page.offset + page.limit]
    return {
        "items": selected,
        "pagination": {
            "limit": page.limit,
            "offset": page.offset,
            "returned": len(selected),
            "total": len(items),
            "has_more": page.offset + len(selected) < len(items),
        },
    }


@router.get("/summary")
def model_summary(
    service: Annotated[ModelService, Depends(get_model_service)],
) -> dict[str, object]:
    return service.summary()


@router.get("/{model_name}")
def model_detail(
    model_name: str, service: Annotated[ModelService, Depends(get_model_service)]
) -> dict[str, object]:
    return service.get(model_name)
