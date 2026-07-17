"""Consistent, traceback-free FastAPI error responses."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from gridmind.exceptions import (
    AlertLifecycleError,
    GridMindError,
    ResourceNotFoundError,
    ServiceUnavailableError,
)

LOGGER = logging.getLogger(__name__)


def response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    *,
    headers: Mapping[str, str] | None = None,
    **details: Any,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        headers=headers,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": getattr(request.state, "request_id", "unknown"),
                "details": details,
            }
        },
    )


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ResourceNotFoundError)
    async def not_found(request: Request, exc: ResourceNotFoundError) -> JSONResponse:
        return response(request, 404, "resource_not_found", str(exc))

    @app.exception_handler(AlertLifecycleError)
    async def invalid_transition(request: Request, exc: AlertLifecycleError) -> JSONResponse:
        return response(request, 409, "invalid_alert_transition", str(exc))

    @app.exception_handler(ServiceUnavailableError)
    async def unavailable(request: Request, exc: ServiceUnavailableError) -> JSONResponse:
        return response(request, 503, "service_unavailable", str(exc))

    @app.exception_handler(GridMindError)
    async def known(request: Request, exc: GridMindError) -> JSONResponse:
        return response(request, 400, "gridmind_error", str(exc))

    @app.exception_handler(RequestValidationError)
    async def validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        return response(
            request, 422, "validation_error", "Request validation failed.", errors=exc.errors()
        )

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        return response(
            request,
            exc.status_code,
            "http_error",
            str(exc.detail),
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception) -> JSONResponse:
        LOGGER.exception(
            "Unhandled API error",
            extra={"request_id": getattr(request.state, "request_id", "unknown")},
        )
        return response(request, 500, "internal_error", "An internal application error occurred.")
