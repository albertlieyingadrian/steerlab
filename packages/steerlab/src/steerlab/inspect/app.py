"""The HTTP side of `steerlab inspect`: one page and a JSON route per view.

    GET  /               the explorer page            GET  /api/info     model shape and the preset prompts
    POST /api/run        {text, target?, pos?}        POST /api/layer    {text, target?, pos?, layer}
    POST /api/neuron     {..., layer, neuron}         POST /api/find     {word, text?, pos?}
    POST /api/edit       {..., edits: [{layer, neuron, mode, value}]}   POST /api/dose  {..., layer, neuron}

The model is not thread-safe, so every view runs under one lock (FastAPI runs these sync routes in a thread pool). A
bad request (unknown layer, prompt too long, empty word) is a 400 with the reason. As in `steerlab serve`,
`TrustedHostMiddleware` refuses unknown `Host` headers, and model output reaches the page as JSON the page inserts
with `textContent`, never as HTML.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from steerlab.inspect.core import EDIT_MODES, EditMode, Inspector
from steerlab.inspect.core import Edit as CoreEdit
from steerlab.inspect.export import PRESETS

STATIC = Path(__file__).parent / "static"
LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "[::1]")
MAX_EDITS = 8


class Prompt(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    target: str | int | None = None
    pos: int | None = None


class LayerBody(Prompt):
    layer: int


class NeuronBody(LayerBody):
    neuron: int


class FindBody(BaseModel):
    word: str | int
    text: str | None = Field(default=None, max_length=1000)
    pos: int | None = None


class EditItem(BaseModel):
    layer: int
    neuron: int
    mode: EditMode = "off"
    value: float = Field(default=0.0, ge=-1000, le=1000)


class EditBody(Prompt):
    edits: list[EditItem] = Field(min_length=1, max_length=MAX_EDITS)
    max_new_tokens: int = Field(default=12, ge=1, le=40)


def create_app(inspector: Inspector, *, hosts: Iterable[str] = LOOPBACK_HOSTS) -> FastAPI:
    """The explorer around a loaded `Inspector`."""
    lock = threading.Lock()
    app = FastAPI(title="steerlab inspect", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(hosts))
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    def view(compute: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        with lock:
            try:
                return compute()
            except ValueError as error:
                raise HTTPException(status_code=400, detail=str(error)) from None

    @app.get("/", include_in_schema=False)
    def page() -> FileResponse:
        return FileResponse(STATIC / "index.html")

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

    @app.post("/api/run")
    def run(body: Prompt) -> dict[str, Any]:
        return view(lambda: inspector.run(body.text, body.target, body.pos))

    @app.post("/api/layer")
    def layer(body: LayerBody) -> dict[str, Any]:
        return view(lambda: inspector.layer(body.text, body.layer, body.target, body.pos))

    @app.post("/api/neuron")
    def neuron(body: NeuronBody) -> dict[str, Any]:
        return view(lambda: inspector.neuron(body.text, body.layer, body.neuron, body.target, body.pos))

    @app.post("/api/find")
    def find(body: FindBody) -> dict[str, Any]:
        return view(lambda: inspector.find(body.word, body.text, body.pos))

    @app.post("/api/edit")
    def edit(body: EditBody) -> dict[str, Any]:
        edits = [CoreEdit(item.layer, item.neuron, item.mode, item.value) for item in body.edits]
        return view(lambda: inspector.edit(body.text, edits, body.target, body.pos, body.max_new_tokens))

    @app.post("/api/dose")
    def dose(body: NeuronBody) -> dict[str, Any]:
        return view(lambda: inspector.dose(body.text, body.layer, body.neuron, body.target, body.pos))

    return app
