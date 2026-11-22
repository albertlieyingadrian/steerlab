"""The Trace room: today's layer explorer (prompt → layer stack → layer / neuron / find / edit / dose tabs).

Its routes keep the paths they had before rooms existed (`/api/run`, `/api/layer`, ...), so the page, the static
export and every saved link still work unchanged; new rooms put theirs under `/api/<room>/`.

Every view runs under the shared edit list (`app.state.edits`, kept by the Edit room) when it has enabled edits:
they are applied for the request and the weights restored after it. Each such response carries `shared_edits`,
the labels of what was on. The Edit tab's own neuron edits go on top of them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from steerlab.inspect.core import Edit as CoreEdit
from steerlab.inspect.core import EditMode

if TYPE_CHECKING:
    from fastapi import FastAPI

    from steerlab.inspect.rooms import RoomContext

ROOM = {"id": "trace", "title": "Trace", "order": 10, "key": "t"}
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


def register(app: FastAPI, ctx: RoomContext) -> None:
    inspector = ctx.inspector

    def view(compute: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        """`ctx.view`, with the shared edit list's enabled edits on for the duration of the request."""

        def under_edits() -> dict[str, Any]:
            stack = getattr(app.state, "edits", None)
            on = [r.label for r in stack.records if r.enabled] if stack is not None else []
            if stack is None or not on:
                return compute()
            with stack.applied(inspector.model, inspector.tokenizer):
                return {**compute(), "shared_edits": on}

        return ctx.view(under_edits)

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
