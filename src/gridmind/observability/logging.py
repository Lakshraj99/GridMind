"""Structured, secret-redacted application logging."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from gridmind.logging_config import SecretRedactionFilter, redact_sensitive_text


class JSONFormatter(logging.Formatter):
    """Render stable JSON log events without headers or secrets."""

    fields = ("request_id", "method", "path", "status_code", "duration_ms")

    def __init__(self, secrets: tuple[str, ...] = ()) -> None:
        super().__init__()
        self.secrets = secrets

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(
            {field: getattr(record, field) for field in self.fields if hasattr(record, field)}
        )
        return redact_sensitive_text(json.dumps(payload, default=str), self.secrets)


def configure_application_logging(
    level: str, log_format: str, *, api_key: str | None = None, eia_key: str | None = None
) -> None:
    """Configure every application handler with final-pass secret redaction."""
    secrets = tuple(value for value in (api_key, eia_key) if value)
    handler = logging.StreamHandler()
    handler.addFilter(SecretRedactionFilter(secrets))
    if log_format == "json":
        handler.setFormatter(JSONFormatter(secrets))
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
