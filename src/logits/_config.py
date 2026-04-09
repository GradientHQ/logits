from __future__ import annotations

from collections.abc import Mapping
import os
from typing import Any

LOGITS_API_KEY_ENV = "LOGITS_API_KEY"
LOGITS_BASE_URL_ENV = "LOGITS_BASE_URL"
TINKER_API_KEY_ENV = "TINKER_API_KEY"
TINKER_BASE_URL_ENV = "TINKER_BASE_URL"
API_KEY_HEADER = "X-API-Key"
COMPAT_API_KEY = "tml-logits-compat"


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

    if resolved_api_key.startswith("tml-"):
        kwargs["api_key"] = resolved_api_key
    else:
        kwargs["api_key"] = COMPAT_API_KEY
        headers[API_KEY_HEADER] = resolved_api_key

    if headers:
        kwargs["default_headers"] = headers
    return kwargs
