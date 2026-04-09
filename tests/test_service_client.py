from __future__ import annotations

import json
import os

import httpx
import pytest
from respx import MockRouter

import logits

BASE_URL = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


@pytest.mark.respx(base_url=BASE_URL)
def test_service_client_uses_non_tml_logits_key_as_header(respx_mock: MockRouter) -> None:
    create_session_route = respx_mock.post("/api/v1/create_session").mock(
        return_value=httpx.Response(200, json={"session_id": "test-session-id"})
    )

    service_client = logits.ServiceClient(base_url=BASE_URL, api_key="logits-test-key")
    service_client.holder.close()

    assert create_session_route.called
    assert create_session_route.calls[0].request.headers["X-API-Key"] == "logits-test-key"


@pytest.mark.respx(base_url=BASE_URL)
def test_service_client_prefers_logits_env_vars(
    respx_mock: MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    create_session_route = respx_mock.post("/api/v1/create_session").mock(
        return_value=httpx.Response(200, json={"session_id": "test-session-id"})
    )
    monkeypatch.setenv("LOGITS_API_KEY", "logits-env-key")
    monkeypatch.setenv("LOGITS_BASE_URL", BASE_URL)
    monkeypatch.setenv("TINKER_API_KEY", "tml-tinker-fallback")
    monkeypatch.setenv("TINKER_BASE_URL", "https://example.invalid")

    service_client = logits.ServiceClient()
    service_client.holder.close()

    assert create_session_route.called
    request = create_session_route.calls[0].request
    assert str(request.url).startswith(f"{BASE_URL}/api/v1/create_session")
    assert request.headers["X-API-Key"] == "logits-env-key"


@pytest.mark.respx(base_url=BASE_URL)
def test_create_service_client_returns_logits_service_client(respx_mock: MockRouter) -> None:
    respx_mock.post("/api/v1/create_session").mock(
        return_value=httpx.Response(200, json={"session_id": "test-session-id"})
    )

    service_client = logits.create_service_client(base_url=BASE_URL, api_key="tml-direct-key")
    service_client.holder.close()

    assert isinstance(service_client, logits.ServiceClient)
