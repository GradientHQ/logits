from __future__ import annotations

LOGITS_URI_PREFIX = "logits://"
TINKER_URI_PREFIX = "tinker://"

_VALID_MODEL_PATH_PREFIXES = (LOGITS_URI_PREFIX, TINKER_URI_PREFIX)


def is_valid_model_path(path: str | None) -> bool:
    """Whether the URI looks like a supported weights path.

    Tinker SDK historically required ``tinker://``. Newer backends may return
    ``logits://`` instead, so we accept both prefixes at the SDK boundary.
    """

    if path is None:
        return False
    return path.startswith(_VALID_MODEL_PATH_PREFIXES)


def normalize_tinker_path(path: str) -> str:
    """Convert logits://<...> to tinker://<...> for local parsing only."""

    if path.startswith(LOGITS_URI_PREFIX):
        return TINKER_URI_PREFIX + path[len(LOGITS_URI_PREFIX) :]
    return path

