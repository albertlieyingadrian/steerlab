"""The Lens room: what the residual stream says at every layer and position, through the logit, J, R or S lens.

    GET  /api/lens/info    the lenses and their status (ready / downloadable / needs a fit), the model's slug
    POST /api/lens/load    {lens}: download or load a lens (rs-lenses from the Hub, or our own fit)
    POST /api/lens/grid    {text, lens, z}: the top word in every layer × position cell, and the model's real output
    POST /api/lens/cell    {text, lens, z, layer, pos, k}: one cell's top words through every loaded lens, raw and
                           layer-contrastive, and the neurons and heads of that layer that push its top word most

Reads use `steerlab.lens.jacobian.Reader`. A cell's "units" are layer L's neurons (activation × the lens's read of
the neuron's write, for the cell's top token) and heads (their output at that position, read the same way); each
links to the Units room at /units/<slug>/L<l>/n<i><pole> (pole + or -, the sign of the activation) or /h<i>.
"""

from __future__ import annotations

import importlib
import threading
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from fastapi import FastAPI

    from steerlab.inspect.rooms import RoomContext

ROOM = {"id": "lens", "title": "Lens", "order": 20, "key": "l"}
LensName = Literal["logit", "J", "R", "S"]
MAX_UNITS = 8


class GridBody(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    lens: LensName = "logit"
    z: bool = False


class CellBody(GridBody):
    layer: int
    pos: int
    k: int = Field(default=10, ge=1, le=40)


class LoadBody(BaseModel):
    lens: LensName


class LensState:
    """The reader for the loaded model, with lenses added as they are loaded."""

    def __init__(self, ctx: RoomContext) -> None:
        from steerlab.lens import jacobian

        self.ctx = ctx
        self.jacobian = jacobian
        self.reader = jacobian.Reader(ctx.inspector.model, jacobian.load(ctx.model_id, download=False))
        self.loading = threading.Lock()

    def info(self) -> dict[str, Any]:
        status = self.jacobian.availability(self.ctx.model_id)
        for lens in self.reader.lenses:
            status[lens]["status"] = "loaded"
        status["logit"]["status"] = "loaded"
        inspector = self.ctx.inspector
        return {
            "model": self.ctx.model_id,
            "slug": self.jacobian.slug(self.ctx.model_id),
            "n_layers": inspector.n_layers,
            "n_heads": inspector.n_heads,
            "lenses": status,
            "loaded": self.reader.available(),
        }

    def load(self, lens: str) -> dict[str, Any]:
        if lens != "logit" and lens not in self.reader.lenses:
            with self.loading:
                try:
                    self.reader.lenses[lens] = self.jacobian.load_one(self.ctx.model_id, lens, download=True)
                except OSError as error:
                    raise ValueError(f"could not load the {lens}-lens: {error}") from None
        return self.info()


def _scores(state: LensState, h: Any, layer: int, lens: str, z: bool) -> Any:
    reader = state.reader
    return reader.z_logits(h, layer, lens) if z else reader.logits(h, layer, lens)


def grid(state: LensState, body: GridBody) -> dict[str, Any]:
    import torch

    inspector = state.ctx.inspector
    ids = inspector.tokenize(body.text)
    trace = inspector.trace(ids)
    layers = []
    for layer in range(inspector.n_layers):
        scores = _scores(state, trace.resid[layer], layer, body.lens, body.z)  # [seq, vocab]
        if body.z or body.lens == "S":
            top = scores.max(-1)
            values = top.values
        else:
            probs = torch.softmax(scores, -1)
            top = probs.max(-1)
            values = top.values
        layers.append(
            {
                "layer": layer,
                "cells": [
                    {"id": int(i), "text": inspector.token_text(int(i)), "value": round(float(v), 4)}
                    for i, v in zip(top.indices, values, strict=True)
                ],
            }
        )
    out_probs = torch.softmax(trace.logits, -1).max(-1)
    return {
        "lens": body.lens,
        "z": body.z,
        "value": "z" if body.z else ("score" if body.lens == "S" else "p"),
        "tokens": [{"id": i, "text": inspector.token_text(i)} for i in ids],
        "layers": layers,
        "output": [
            {"id": int(i), "text": inspector.token_text(int(i)), "value": round(float(p), 4)}
            for i, p in zip(out_probs.indices, out_probs.values, strict=True)
        ],
    }


def unit_href(model_id: str, layer: int, index: int, pole: int | None = None) -> str:
    """The Units room's link to a neuron pole (pole ±1) or a head (no pole): `steerlab.atlas.unit_href` when the
    atlas is installed, else the same rule (`/units/<slug>/L<l>/n<i><+|->` or `/h<i>`)."""
    try:
        atlas_href = importlib.import_module("steerlab.atlas.address").unit_href
    except ImportError:
        from steerlab.lens.jacobian import slug

        tail = f"h{index}" if pole is None else f"n{index}{'+' if pole > 0 else '-'}"
        return f"/units/{slug(model_id)}/L{layer}/{tail}"
    return str(atlas_href(model_id, layer, index, pole))


def _direction(state: LensState, layer: int, lens: str, token: int) -> Any:
    """The vector r with `r · x` = the lens's (pre-norm, linear) score of `token` for a write x into layer's output."""
    reader = state.reader
    if lens == "S":
        u = reader.E[token]
        return reader.lenses["S"][layer] @ u  # c · K (E_v − Ē) = c · r
    u = reader.W_U[token] * reader.g
    return u if lens == "logit" else reader.lenses[lens][layer].T @ u


def cell(state: LensState, body: CellBody) -> dict[str, Any]:
    import torch

    inspector = state.ctx.inspector
    reader = state.reader
    ids, pos = inspector.resolve(body.text, body.pos)
    if not 0 <= body.layer < inspector.n_layers:
        raise ValueError(f"layer {body.layer} is outside 0..{inspector.n_layers - 1}")
    trace = inspector.trace(ids)
    h = trace.resid[body.layer][pos]

    def words(lens: str, z: bool) -> list[dict[str, Any]]:
        scores = _scores(state, h, body.layer, lens, z)
        if not z and lens != "S":
            scores = torch.softmax(scores, -1)
        vals, idx = torch.topk(scores, body.k)
        return [
            {"id": int(i), "text": inspector.token_text(int(i)), "value": round(float(v), 4)}
            for i, v in zip(idx, vals, strict=True)
        ]

    reads = {lens: {"raw": words(lens, False), "z": words(lens, True)} for lens in reader.available()}
    chosen = reads[body.lens]["z" if body.z else "raw"][0]
    r = _direction(state, body.layer, body.lens, chosen["id"])
    acts = trace.acts[body.layer][pos]
    neuron_scores = acts * (inspector.down(body.layer) @ r)
    head_scores = inspector.head_outputs(trace, body.layer, pos) @ r

    def neuron_rows(order: Any) -> list[dict[str, Any]]:
        rows = []
        for n in order.tolist():
            pole = "+" if float(acts[n]) >= 0 else "-"
            rows.append(
                {
                    "unit": f"L{body.layer} n{n}{pole}",
                    "href": unit_href(state.ctx.model_id, body.layer, n, 1 if pole == "+" else -1),
                    "act": round(float(acts[n]), 4),
                    "push": round(float(neuron_scores[n]), 4),
                }
            )
        return rows

    head_order = torch.argsort(head_scores.abs(), descending=True)[:MAX_UNITS]
    return {
        "layer": body.layer,
        "pos": pos,
        "token": {"id": ids[pos], "text": inspector.token_text(ids[pos])},
        "lens": body.lens,
        "z": body.z,
        "reads": reads,
        "top": chosen,
        "neurons": {
            "toward": neuron_rows(torch.topk(neuron_scores, MAX_UNITS).indices),
            "against": neuron_rows(torch.topk(-neuron_scores, MAX_UNITS).indices),
        },
        "heads": [
            {
                "unit": f"L{body.layer} h{int(i)}",
                "href": unit_href(state.ctx.model_id, body.layer, int(i)),
                "push": round(float(head_scores[i]), 4),
            }
            for i in head_order
        ],
    }


def register(app: FastAPI, ctx: RoomContext) -> None:
    states: list[LensState] = []

    def state() -> LensState:
        if not states:
            states.append(LensState(ctx))
        return states[0]

    @app.get("/api/lens/info")
    def info() -> dict[str, Any]:
        return ctx.view(lambda: state().info())

    @app.post("/api/lens/load")
    def load(body: LoadBody) -> dict[str, Any]:
        return ctx.view(lambda: state().load(body.lens))

    @app.post("/api/lens/grid")
    def lens_grid(body: GridBody) -> dict[str, Any]:
        return ctx.view(lambda: grid(state(), body))

    @app.post("/api/lens/cell")
    def lens_cell(body: CellBody) -> dict[str, Any]:
        return ctx.view(lambda: cell(state(), body))
