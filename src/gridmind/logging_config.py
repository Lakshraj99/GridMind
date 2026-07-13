"""Application logging configuration with credential redaction."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import quote, quote_plus

REDACTED_MARKER = "[REDACTED]"
_API_KEY_PARAMETER = re.compile(r"(?i)(api_key\s*=\s*)([^&\s\"']*)")
_NOISY_HTTP_LOGGERS = ("httpx", "httpcore")

for _logger_name in _NOISY_HTTP_LOGGERS:
    logging.getLogger(_logger_name).setLevel(logging.WARNING)


def redact_sensitive_text(value: object, secrets: Iterable[str] = ()) -> str:
    """Return text with EIA query credentials and configured secret values removed."""
    redacted = _API_KEY_PARAMETER.sub(rf"\1{REDACTED_MARKER}", str(value))
    for secret in secrets:
        if not secret:
            continue
        for representation in {secret, quote(secret, safe=""), quote_plus(secret, safe="")}:
            if representation:
                redacted = redacted.replace(representation, REDACTED_MARKER)
    return redacted


def redact_sensitive_value(value: Any, secrets: Iterable[str] = ()) -> Any:
    """Recursively redact credentials in JSON-compatible values."""
    secret_values = tuple(secrets)
    if isinstance(value, Mapping):
        return {
            key: (
                REDACTED_MARKER
                if str(key).lower() == "api_key"
                else redact_sensitive_value(item, secret_values)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive_value(item, secret_values) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_sensitive_value(item, secret_values) for item in value)
    if isinstance(value, str):
        return redact_sensitive_text(value, secret_values)
    return value


class SecretRedactionFilter(logging.Filter):
    """Redact EIA credentials from log-record messages and arguments."""

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        super().__init__()
        self.secrets = tuple(secret for secret in secrets if secret)

    def filter(self, record: logging.LogRecord) -> bool:
        """Sanitize the record in place before any handler emits it."""
        record.msg = redact_sensitive_text(record.getMessage(), self.secrets)
        record.args = ()
        if record.exc_info is not None:
            _, exception, traceback = record.exc_info
            safe_exception = Exception(redact_sensitive_text(exception, self.secrets))
            record.exc_info = (Exception, safe_exception, traceback)
            record.exc_text = None
        if record.stack_info is not None:
            record.stack_info = redact_sensitive_text(record.stack_info, self.secrets)
        return True


class _RedactingFormatter(logging.Formatter):
    """Apply a final redaction pass, including formatted exception text."""

    def __init__(self, fmt: str, secrets: Iterable[str]) -> None:
        super().__init__(fmt)
        self.secrets = tuple(secrets)

    def format(self, record: logging.LogRecord) -> str:
        return redact_sensitive_text(super().format(record), self.secrets)


def configure_logging(level: str = "INFO", *, eia_api_key: str | None = None) -> None:
    """Configure concise, secret-safe process-wide logging."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=numeric_level, format=log_format)
    secrets = (eia_api_key,) if eia_api_key else ()
    redaction_filter = SecretRedactionFilter(secrets)
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.addFilter(redaction_filter)
        handler.setFormatter(_RedactingFormatter(log_format, secrets))
    for logger_name in ("gridmind", *_NOISY_HTTP_LOGGERS):
        logging.getLogger(logger_name).addFilter(redaction_filter)
    for logger_name in _NOISY_HTTP_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
