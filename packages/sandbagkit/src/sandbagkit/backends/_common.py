"""Small helpers the backends share: optional imports, API keys and stop strings."""

from __future__ import annotations

import importlib
import os
from collections.abc import Sequence
from types import ModuleType


class MissingExtraError(ImportError):
    """An optional dependency isn't installed. The message names the extra to install."""


class MissingAPIKeyError(RuntimeError):
    """No API key was found. Raised in the constructor, before any request could be sent."""


def require_module(module: str, extra: str) -> ModuleType:
    """Import `module`, or explain which sandbagkit extra provides it."""
    try:
        return importlib.import_module(module)
    except ImportError as error:
        raise MissingExtraError(
            f"`{module}` is not installed. Install it with: pip install 'sandbagkit[{extra}]'"
        ) from error


def resolve_api_key(api_key: str | None, env_vars: Sequence[str], provider: str) -> str:
    """The explicit key, else the first non-empty environment variable, else a clear error."""
    if api_key:
        return api_key
    for name in env_vars:
        value = os.environ.get(name)
        if value:
            return value
    names = " or ".join(env_vars)
    raise MissingAPIKeyError(
        f"No {provider} API key: pass api_key=... or set {names}. Nothing was sent. "
        f"Remember that every {provider} call is billed; the lessons use FakeBackend and local models instead."
    )


def truncate_at_stop(text: str, stop: Sequence[str]) -> tuple[str, bool]:
    """Cut `text` at the earliest stop string. Returns (text, whether a stop string was found)."""
    cut = len(text)
    for marker in stop:
        if not marker:
            continue
        position = text.find(marker)
        if position != -1:
            cut = min(cut, position)
    return text[:cut], cut < len(text)
