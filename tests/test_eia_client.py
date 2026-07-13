"""Offline tests for EIA API communication and pagination."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from gridmind.config import Settings
from gridmind.data.eia_client import EIAClient
from gridmind.exceptions import (
    ConfigurationError,
    EIAAuthenticationError,
    EIAMalformedResponseError,
    EIANetworkError,
    EIARateLimitError,
)


def test_pagination_and_query_construction(eia_payload: dict[str, Any]) -> None:
    records = eia_payload["response"]["data"]
    offsets: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/electricity/rto/region-data/data/")
        assert request.url.params["facets[respondent][]"] == "PJM"
        assert request.url.params["sort[0][column]"] == "period"
        assert request.url.params["sort[1][column]"] == "type"
        offset = int(request.url.params["offset"])
        offsets.append(offset)
        page = records[offset : offset + 3]
        return httpx.Response(200, json={"response": {"total": "8", "data": page}})

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = EIAClient("secret", client=http_client, page_size=3, sleep=lambda _: None)
    result = client.fetch_hourly_data("PJM", "2024-01-01", "2024-01-02")

    assert len(result.records) == 8
    assert len(result.pages) == 3
    assert offsets == [0, 3, 6]
    assert "secret" not in client.data_url


def test_pagination_exact_page_boundary_does_not_lose_records() -> None:
    records = [
        {"period": f"2024-01-01T0{index}", "respondent": "PJM", "type": "D", "value": index}
        for index in range(6)
    ]
    offsets: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        offset = int(request.url.params["offset"])
        offsets.append(offset)
        return httpx.Response(
            200, json={"response": {"total": "6", "data": records[offset : offset + 3]}}
        )

    client = EIAClient(
        "secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        page_size=3,
    )
    result = client.fetch_hourly_data("PJM", "2024-01-01", "2024-01-02")
    assert result.records == records
    assert offsets == [0, 3, 6]


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (401, EIAAuthenticationError),
        (429, EIARateLimitError),
        (503, EIANetworkError),
    ],
)
def test_api_errors(status: int, error: type[Exception]) -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(status, json={"message": "no"}))
    client = EIAClient(
        "secret", client=httpx.Client(transport=transport), max_retries=0, sleep=lambda _: None
    )
    with pytest.raises(error):
        client.fetch_hourly_data("PJM", "2024-01-01", "2024-01-02")


def test_retry_then_success() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(500, json={})
        return httpx.Response(200, json={"response": {"total": 0, "data": []}})

    client = EIAClient(
        "secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=1,
        sleep=lambda _: None,
    )
    assert client.fetch_hourly_data("PJM", "2024-01-01", "2024-01-02").records == []
    assert calls == 2


def test_network_exception_is_wrapped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    client = EIAClient(
        "secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    with pytest.raises(EIANetworkError, match="after retries"):
        client.fetch_hourly_data("PJM", "2024-01-01", "2024-01-02")


@pytest.mark.parametrize(
    "payload",
    [{}, {"response": {"data": "wrong"}}, {"response": {"data": [], "total": "wrong"}}],
)
def test_malformed_response(payload: dict[str, Any]) -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    client = EIAClient("secret", client=httpx.Client(transport=transport))
    with pytest.raises(EIAMalformedResponseError):
        client.fetch_hourly_data("PJM", "2024-01-01", "2024-01-02")


def test_missing_api_key_is_actionable() -> None:
    settings = Settings(EIA_API_KEY=None, _env_file=None)
    with pytest.raises(ConfigurationError, match="EIA_API_KEY"):
        settings.require_eia_api_key()
