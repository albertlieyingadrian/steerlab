"""Model backends: every model behind the one `ModelBackend` protocol from `sandbagkit.core.backend`.

    HFBackend         a local Hugging Face model (needs the `probing` extra: torch + transformers)
    FakeBackend       scripted and offline, for tests and lessons (lives in core)

The names load lazily, so `import sandbagkit.backends` never imports torch. torch is imported the first time you
touch the class that needs it.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from sandbagkit.backends._common import MissingAPIKeyError, MissingExtraError

if TYPE_CHECKING:
    from sandbagkit.backends.hf import HFBackend, pick_device, pick_dtype

_LAZY = {
    "HFBackend": "sandbagkit.backends.hf",
    "pick_device": "sandbagkit.backends.hf",
    "pick_dtype": "sandbagkit.backends.hf",
}

__all__ = [
    "HFBackend",
    "MissingAPIKeyError",
    "MissingExtraError",
    "pick_device",
    "pick_dtype",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'sandbagkit.backends' has no attribute {name!r}")
