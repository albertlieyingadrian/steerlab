"""The Units room: an atlas of every neuron pole and attention head, with a page per unit.

    GET  /api/units/models                          catalog models, computed atlases, the loaded model, build jobs
    GET  /api/units/overview?model=&top=40           the atlas landing: each layer's easiest units and its heads
    GET  /api/units/unit?model=&address=L8/n344%2B   one unit's evidence card
    GET  /api/units/search?model=&q=&reads=0&kind=   word search (an address or "L8" is answered directly)
    GET  /api/units/graph?model=&address=&up=2&down=2&fan=4    the wiring walk around a unit
    POST /api/units/fire   {model, address, text}    run a prompt on the loaded model and read the unit per token
    POST /api/units/build  {model}                   compute the loaded model's atlas (background; poll /models)

Models in the stdlogits/neuron-lens catalog (Qwen3 0.6B-14B and their bases, Gemma 4 12B) are read from it without
loading them; any other model is computed from the loaded one (`steerlab.atlas.compute`), or from an atlas built on
Modal and copied into the cache. The heavy code is `steerlab.atlas`; this module only routes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from pydantic import BaseModel, Field

from steerlab.atlas.address import Address

if TYPE_CHECKING:
    from fastapi import FastAPI

    from steerlab.atlas.service import AtlasService
    from steerlab.inspect.rooms import RoomContext

ROOM = {"id": "units", "title": "Units", "order": 30, "key": "u"}
T = TypeVar("T")


class FireBody(BaseModel):
    model: str = Field(min_length=1, max_length=120)
    address: str = Field(min_length=2, max_length=40)
    text: str = Field(min_length=1, max_length=1000)


class BuildBody(BaseModel):
    model: str = Field(min_length=1, max_length=120)
    n_tokens: int = Field(default=24_000, ge=1_000, le=200_000)


def service_for(ctx: RoomContext) -> AtlasService:
    from steerlab.atlas.compute import Parts
    from steerlab.atlas.service import AtlasService

    inspector = ctx.inspector
    return AtlasService(
        ctx.cache_dir / "atlas",
        loaded=ctx.model_id,
        parts=lambda: Parts(inspector.model, inspector.tokenizer),
        lock=ctx.lock,
    )


def guarded(compute: Callable[[], T]) -> T:
    """A ValueError is a 400 with its reason; a missing catalog table or atlas file is a 404."""
    from fastapi import HTTPException

    try:
        return compute()
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from None
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from None


def _clamp(value: int, high: int) -> int:
    return max(0, min(value, high))


def register(app: FastAPI, ctx: RoomContext) -> None:
    atlas = service_for(ctx)
    app.state.atlas = atlas

    @app.get("/api/units/models")
    def models() -> dict[str, Any]:
        return guarded(atlas.models)

    @app.get("/api/units/overview")
    def overview(model: str, top: int = 40) -> dict[str, Any]:
        return guarded(lambda: atlas.overview(model, top=max(1, min(top, 200))))

    @app.get("/api/units/unit")
    def unit(model: str, address: str) -> dict[str, Any]:
        return guarded(lambda: atlas.unit(model, Address.parse(address)))

    @app.get("/api/units/search")
    def search(
        model: str, q: str, reads: bool = False, kind: Literal["neuron", "head"] | None = None, limit: int = 30
    ) -> dict[str, Any]:
        return guarded(lambda: atlas.search(model, q[:200], reads=reads, kind=kind, limit=max(1, min(limit, 100))))

    @app.get("/api/units/graph")
    def graph(model: str, address: str, up: int = 2, down: int = 2, fan: int = 4) -> dict[str, Any]:
        where = Address.parse
        return guarded(
            lambda: atlas.graph(model, where(address), _clamp(up, 3), _clamp(down, 3), max(1, _clamp(fan, 6)))
        )

    @app.post("/api/units/fire")
    def fire(body: FireBody) -> dict[str, Any]:
        return guarded(lambda: atlas.fire(body.model, Address.parse(body.address), body.text))

    @app.post("/api/units/build")
    def build(body: BuildBody) -> dict[str, Any]:
        return guarded(lambda: atlas.build(body.model, n_tokens=body.n_tokens))
