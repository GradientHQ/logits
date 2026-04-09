from __future__ import annotations

import importlib
import pkgutil
import sys

import tinker.types as _tinker_types
from tinker.types import *  # noqa: F403

for module_info in pkgutil.walk_packages(_tinker_types.__path__, prefix="tinker.types."):
    tinker_module_name = module_info.name
    logits_module_name = tinker_module_name.replace("tinker.", "logits.", 1)
    sys.modules[logits_module_name] = importlib.import_module(tinker_module_name)

__all__ = [
    name
    for name in globals()
    if not name.startswith("_")
    and name not in {"importlib", "pkgutil", "sys", "module_info", "tinker_module_name", "logits_module_name"}
]

__locals = locals()
for __name in __all__:
    try:
        __locals[__name].__module__ = "logits.types"
    except (TypeError, AttributeError):
        pass
