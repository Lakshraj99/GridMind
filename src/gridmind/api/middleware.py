"""Request correlation, access logging, and HTTP metrics middleware."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response

LOGGER = logging.getLogger("gridmind.api.access")


def install_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        metrics = request.app.state.metrics
        started = time.perf_counter()
        metrics.active.inc()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            duration = time.perf_counter() - started
            metrics.active.dec()
            path = request.url.path
            metrics.requests.labels(request.method, path, str(status_code)).inc()
            metrics.latency.labels(request.method, path).observe(duration)
            if status_code >= 400:
                metrics.errors.labels(request.method, path, str(status_code)).inc()
            LOGGER.info(
                "request completed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": path,
                    "status_code": status_code,
                    "duration_ms": round(duration * 1000, 3),
                },
            )
