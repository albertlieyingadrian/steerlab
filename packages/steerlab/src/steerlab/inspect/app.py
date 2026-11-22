"""The HTTP side of `steerlab inspect`: one page, a nav of rooms, and a JSON route per view.

    GET  /, /<room>/...    the page (deep links: /trace?preset=france, /lens/..., /units/<model>/L8/n344+)
    GET  /api/info         model shape and the preset prompts        GET  /api/rooms   the rooms, in nav order
    POST /api/run, /api/layer, /api/neuron, /api/find, /api/edit, /api/dose      the Trace room (rooms/trace.py)
    /api/<room>/...        every other room's routes (rooms/<room>.py)

The model is not thread-safe, so every view runs under one lock (`RoomContext.view`; FastAPI runs these sync routes
in a thread pool). A bad request (unknown layer, prompt too long, empty word) is a 400 with the reason. As in
`steerlab serve`, `TrustedHostMiddleware` refuses unknown `Host` headers, and model output reaches the page as JSON
the page inserts with `textContent`, never as HTML.

The page is served with `<base href="/">` and `data-room="<room>"` added, so its relative `static/` and `api/` URLs
work from any deep link; the static export (`export.py`) copies the file without them.
"""

from __future__ import annotations

import html
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from steerlab.inspect.core import EDIT_MODES, Inspector
from steerlab.inspect.export import PRESETS
from steerlab.inspect.rooms import RoomContext, discover

STATIC = Path(__file__).parent / "static"
LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "[::1]")


def page_for(room: str) -> str:
    """index.html with a base URL and the starting room marked (read per request so edits show on reload)."""
    text = (STATIC / "index.html").read_text()
    text = text.replace('<html lang="en">', f'<html lang="en" data-room="{html.escape(room)}">', 1)
    return text.replace('<meta charset="utf-8">', '<meta charset="utf-8">\n<base href="/">', 1)


def create_app(
    inspector: Inspector, *, hosts: Iterable[str] = LOOPBACK_HOSTS, cache_dir: Path | None = None
) -> FastAPI:
    """The explorer around a loaded `Inspector`, with every room in `steerlab.inspect.rooms`."""
    ctx = RoomContext(inspector) if cache_dir is None else RoomContext(inspector, cache_dir=cache_dir)
    app = FastAPI(title="steerlab inspect", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(hosts))
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    app.state.rooms = ctx

    rooms, skipped = discover()
    ctx.rooms = [
        {
            **room.meta,
            "js": (STATIC / "rooms" / f"{room.id}.js").exists(),
            "css": (STATIC / "rooms" / f"{room.id}.css").exists(),
        }
        for room in rooms
    ]

    @app.get("/api/info")
    def info() -> dict[str, Any]:
        return {
            "model": inspector.name,
            "n_layers": inspector.n_layers,
            "n_heads": inspector.n_heads,
            "d_mlp": inspector.d_mlp,
            "modes": list(EDIT_MODES),
            "presets": PRESETS,
        }

    @app.get("/api/rooms")
    def list_rooms() -> dict[str, Any]:
        return {"model": ctx.model_id, "rooms": ctx.rooms, "skipped": skipped}

    for room in rooms:
        room.module.register(app, ctx)

    @app.get("/", include_in_schema=False)
    def page() -> HTMLResponse:
        return HTMLResponse(page_for("trace"))

    def room_page(room_id: str) -> Any:
        def serve(rest: str = "") -> HTMLResponse:
            return HTMLResponse(page_for(room_id))

        return serve

    for room in rooms:
        app.add_api_route(f"/{room.id}", room_page(room.id), methods=["GET"], include_in_schema=False)
        app.add_api_route(f"/{room.id}/{{rest:path}}", room_page(room.id), methods=["GET"], include_in_schema=False)

    return app
