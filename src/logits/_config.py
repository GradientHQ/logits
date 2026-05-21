from __future__ import annotations

from collections.abc import Mapping
import logging
import os
from typing import Any

import tinker.lib._auth_token_provider as _auth_provider
import tinker.lib.internal_client_holder as _internal_client_holder
import tinker._client as _tinker_client
from tinker import types as _tinker_types

LOGITS_API_KEY_ENV = "LOGITS_API_KEY"
LOGITS_BASE_URL_ENV = "LOGITS_BASE_URL"
TINKER_API_KEY_ENV = "TINKER_API_KEY"
TINKER_BASE_URL_ENV = "TINKER_BASE_URL"
API_KEY_HEADER = "X-API-Key"

_logger = logging.getLogger("logits")

DEFAULT_CLIENT_CONFIG: dict[str, Any] = {
    "pjwt_auth_enabled": False,
    "credential_default_source": "api_key",
    "sample_dispatch_bytes_semaphore_size": 1 << 30,
    "inflight_response_bytes_semaphore_size": 1 << 30,
    "parallel_fwdbwd_chunks": 1,
    "proto_write_fwdbwd": False,
    "billing_exception_max_pause_duration_sec": 0,
    "grpc_target": "",
    "enable_grpc_retrieve_future": False,
    "sample_no_retries": False,
}


class LogitsApiKeyAuthProvider(_auth_provider.AuthTokenProvider):
    """Tinker-compatible auth provider that accepts Logits API keys."""

    def __init__(self, api_key: str | None = None) -> None:
        resolved = api_key or _get_first_env(LOGITS_API_KEY_ENV, TINKER_API_KEY_ENV)
        if not resolved:
            raise _auth_provider.TinkerError(
                "The api_key client option must be set either by passing api_key to the client"
                f" or by setting the {LOGITS_API_KEY_ENV} environment variable"
            )
        self._token = resolved

    async def get_token(self) -> str | None:
        return self._token


def install_logits_auth_compat() -> None:
    _auth_provider.ApiKeyAuthProvider = LogitsApiKeyAuthProvider
    _internal_client_holder.ApiKeyAuthProvider = LogitsApiKeyAuthProvider
    _tinker_client.ApiKeyAuthProvider = LogitsApiKeyAuthProvider


_FETCH_CLIENT_CONFIG_PATCHED = False


def install_client_config_fallback() -> None:
    """Make `_fetch_client_config` fall back to defaults on transport failure.

    Some Logits deployments do not implement `/api/v1/client/config` yet. When
    the upstream tinker holder boots, it blocks on that endpoint before any
    other call. Wrap the holder method so a 404 / connect error returns the
    built-in defaults instead of aborting bootstrap. Once the backend ships
    the endpoint, real responses take over automatically.
    """
    global _FETCH_CLIENT_CONFIG_PATCHED
    if _FETCH_CLIENT_CONFIG_PATCHED:
        return

    original = _internal_client_holder.InternalClientHolder._fetch_client_config

    async def _fetch_client_config_with_fallback(
        self: _internal_client_holder.InternalClientHolder,
        auth: _auth_provider.AuthTokenProvider,
    ) -> _tinker_types.ClientConfigResponse:
        try:
            return await original(self, auth)
        except Exception as exc:  # noqa: BLE001 — narrow below
            if not _is_missing_endpoint_error(exc):
                raise
            _logger.info(
                "logits: /api/v1/client/config unavailable (%s); using built-in defaults",
                _summarize_exc(exc),
            )
            return _tinker_types.ClientConfigResponse.model_validate(DEFAULT_CLIENT_CONFIG)

    _internal_client_holder.InternalClientHolder._fetch_client_config = (  # type: ignore[method-assign]
        _fetch_client_config_with_fallback
    )
    _FETCH_CLIENT_CONFIG_PATCHED = True


def _is_missing_endpoint_error(exc: BaseException) -> bool:
    # httpx.HTTPStatusError exposes `.response`; tinker wraps 4xx into TinkerError.
    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) == 404:
        return True
    # Tinker raises NotFoundError (subclass of TinkerError) for 404 responses.
    if type(exc).__name__ == "NotFoundError":
        return True
    # Connection-level failures (DNS, refused, TLS) — treat as endpoint missing.
    if isinstance(exc, (ConnectionError, OSError)):
        return True
    return False


def _summarize_exc(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:200]


def _get_first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def resolve_api_key(
    api_key: str | None = None,
    *,
    default_headers: Mapping[str, str] | None = None,
) -> str | None:
    if api_key is not None:
        return api_key
    if default_headers is not None and default_headers.get(API_KEY_HEADER):
        return default_headers[API_KEY_HEADER]
    return _get_first_env(LOGITS_API_KEY_ENV, TINKER_API_KEY_ENV)


def resolve_base_url(base_url: str | None = None) -> str | None:
    if base_url is not None:
        return base_url
    return _get_first_env(LOGITS_BASE_URL_ENV, TINKER_BASE_URL_ENV)


def resolve_service_client_kwargs(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    default_headers: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    install_logits_auth_compat()
    install_client_config_fallback()
    headers = dict(default_headers or {})
    resolved_api_key = resolve_api_key(api_key, default_headers=headers)
    resolved_base_url = resolve_base_url(base_url)

    kwargs: dict[str, Any] = {}
    if resolved_base_url is not None:
        kwargs["base_url"] = resolved_base_url

    if resolved_api_key is None:
        if headers:
            kwargs["default_headers"] = headers
        return kwargs

    kwargs["api_key"] = resolved_api_key

    if headers:
        kwargs["default_headers"] = headers
    return kwargs
