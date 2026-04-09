from __future__ import annotations

import typing as _t

import tinker as _tinker
from tinker import *  # noqa: F403

from . import types
from ._config import (
    LOGITS_API_KEY_ENV,
    LOGITS_BASE_URL_ENV,
    TINKER_API_KEY_ENV,
    TINKER_BASE_URL_ENV,
    resolve_api_key,
    resolve_base_url,
)
from ._service_client import ServiceClient, create_service_client
from ._training_client import TrainingClient
from ._version import __title__, __version__

LogitsError = TinkerError

__backend_sdk__ = "tinker"
__backend_sdk_version__ = _tinker.__version__

__all__ = [
    *[name for name in _tinker.__all__ if name not in ("ServiceClient", "TrainingClient")],
    "ServiceClient",
    "TrainingClient",
    "create_service_client",
    "LOGITS_API_KEY_ENV",
    "LOGITS_BASE_URL_ENV",
    "TINKER_API_KEY_ENV",
    "TINKER_BASE_URL_ENV",
    "resolve_api_key",
    "resolve_base_url",
    "LogitsError",
    "__title__",
    "__version__",
    "__backend_sdk__",
    "__backend_sdk_version__",
]

if not _t.TYPE_CHECKING:
    from tinker import resources as resources

__locals = locals()
for __name in __all__:
    if not __name.startswith("__"):
        try:
            __locals[__name].__module__ = "logits"
        except (TypeError, AttributeError):
            pass
