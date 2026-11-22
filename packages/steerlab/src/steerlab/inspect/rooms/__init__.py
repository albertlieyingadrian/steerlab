"""The rooms of `steerlab inspect`: one module per top-level page, found at startup.

A room is a module `steerlab/inspect/rooms/<id>.py` with two names:

    ROOM = {"id": "lens", "title": "Lens", "order": 20, "key": "l"}
    def register(app: FastAPI, ctx: RoomContext) -> None: ...      # adds routes under /api/<id>/...

and, on the page side, `static/rooms/<id>.js` (an ES module exporting `mount(el, api, route)` and `unmount()`, plus an
optional `update(route)` for a new deep link inside the same room) and an optional `static/rooms/<id>.css`.

`discover()` imports every module in this package whose name does not start with `_`, keeps those with a valid `ROOM`
and a `register`, and sorts them by `order`. A module that fails to import (a missing optional dependency, say) is
skipped with a warning and listed in `skipped`, so one broken room never takes the others down. Nothing else needs
registering: adding the file is enough.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from steerlab.inspect.core import Inspector

log = logging.getLogger(__name__)
T = TypeVar("T")
ROOM_KEYS = ("id", "title", "order", "key")
DEFAULT_CACHE = Path.home() / ".cache" / "steerlab" / "inspect"


@dataclass
class RoomContext:
    """What every room gets: the loaded model, the one lock around it, its id and a cache directory of its own.

    The model is not thread-safe and FastAPI runs sync routes in a thread pool, so every computation that touches
    `inspector` goes through `view` (or holds `lock` itself). `view` also turns a `ValueError` into a 400 with the
    reason, the convention every inspect route follows.
    """

    inspector: Inspector
    lock: threading.Lock = field(default_factory=threading.Lock)
    model_id: str = ""
    cache_dir: Path = DEFAULT_CACHE
    rooms: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.model_id:
            self.model_id = self.inspector.name

    def room_cache(self, room: str) -> Path:
        """A per-room, per-model cache directory (created on first use)."""
        path = self.cache_dir / room / self.model_id.replace("/", "--")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def view(self, compute: Callable[[], T]) -> T:
        from fastapi import HTTPException

        with self.lock:
            try:
                return compute()
            except ValueError as error:
                raise HTTPException(status_code=400, detail=str(error)) from None


@dataclass
class Room:
    meta: dict[str, Any]
    module: ModuleType

    @property
    def id(self) -> str:
        return str(self.meta["id"])


def valid_meta(meta: Any) -> bool:
    return (
        isinstance(meta, dict)
        and all(key in meta for key in ROOM_KEYS)
        and isinstance(meta["id"], str)
        and meta["id"].isidentifier()
        and isinstance(meta["key"], str)
        and len(meta["key"]) == 1
    )


def discover(package: str = __name__) -> tuple[list[Room], list[dict[str, str]]]:
    """Every room module in `package`, sorted by `order`, and the ones skipped (with why)."""
    found: list[Room] = []
    skipped: list[dict[str, str]] = []
    pkg = importlib.import_module(package)
    for info in pkgutil.iter_modules(pkg.__path__):
        if info.name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"{package}.{info.name}")
        except Exception as error:  # one broken room must not stop the app
            log.warning("steerlab inspect: room %s skipped (%s)", info.name, error)
            skipped.append({"id": info.name, "error": f"{type(error).__name__}: {error}"})
            continue
        meta: Any = getattr(module, "ROOM", None)
        if not valid_meta(meta) or not callable(getattr(module, "register", None)):
            skipped.append({"id": info.name, "error": "no valid ROOM dict and register(app, ctx)"})
            continue
        found.append(Room(dict(meta), module))
    ids = [room.id for room in found]
    keys = [room.meta["key"] for room in found]
    if len(set(ids)) != len(ids) or len(set(keys)) != len(keys):
        raise ValueError(f"rooms must have unique ids and keys: {ids} {keys}")
    found.sort(key=lambda room: (room.meta["order"], room.id))
    return found, skipped
