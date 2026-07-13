"""Credential-redaction tests for logs, exceptions, and raw artifacts."""

from __future__ import annotations

import logging
import traceback

import httpx
import pytest

from gridmind.data.eia_client import EIAClient
from gridmind.exceptions import EIANetworkError
from gridmind.logging_config import REDACTED_MARKER, configure_logging


def test_httpx_logs_redact_query_and_configured_key(caplog: pytest.LogCaptureFixture) -> None:
    secret = "configured/eia+secret"
    configure_logging("INFO", eia_api_key=secret)
    logger = logging.getLogger("httpx")

    logger.info("this noisy request should be suppressed: %s", secret)
    logger.warning(
        "HTTP Request: GET https://api.eia.gov/v2/data?api_key=%s credential=%s",
        secret,
        secret,
    )
    try:
        raise RuntimeError(f"transport failed for api_key={secret}")
    except RuntimeError:
        logger.exception("request failed")

    assert secret not in caplog.text
    assert "configured%2Feia%2Bsecret" not in caplog.text
    assert REDACTED_MARKER in caplog.text
    assert "this noisy request should be suppressed" not in caplog.text
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("httpcore").level == logging.WARNING


def test_eia_network_exception_does_not_retain_secret_url() -> None:
    secret = "exception-secret"

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"failed request {request.url}", request=request)

    client = EIAClient(
        secret,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    with pytest.raises(EIANetworkError) as raised:
        client.fetch_hourly_data("PJM", "2024-01-01", "2024-01-02")

    rendered = "".join(traceback.format_exception(raised.value))
    assert secret not in str(raised.value)
    assert secret not in rendered
    assert "api_key=" not in rendered
