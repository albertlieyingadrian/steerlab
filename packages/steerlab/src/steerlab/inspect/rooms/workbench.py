"""The Workbench room: chat with the loaded model under steering, on a loom, with probes and artifacts beside it.

Pages: /workbench (the chat and its loom), /workbench/probes, /workbench/artifacts; /workbench/<loom id>/<node id>
deep-links a turn. The numerics live in `steerlab.workbench`; this module is only the HTTP side.

    GET  /api/workbench/state                     model, probe layer, fitting status, probes, steerables
    POST /api/workbench/generate                  {loom, parent, steer?, edits?, probes?, sampling?} -> NDJSON stream:
                                                  {"type":"token",...} per token, then {"type":"done","node":...}
    POST /api/workbench/cancel                    stop the reply being written
    POST /api/workbench/replay                    {loom, node, steer?} the probes on a reply's tokens, re-read
    GET/POST /api/workbench/looms                 list / new; GET|DELETE /looms/{id}; /looms/{id}/turn|fork|remove;
                                                  GET /looms/{id}/compare?a=&b=; POST /looms/import
    GET  /api/workbench/probes                    probes and the bundle; POST /probes/fit {name, positive, negative}
    POST /api/workbench/probes/read               {text} per-token readings + random band; DELETE /probes/{name}
    GET  /api/workbench/artifacts[?type=]         list; GET|DELETE /artifacts/{type}/{name}; POST /artifacts (save)
    POST /api/workbench/artifacts/export          {keys} -> a zip; POST /artifacts/import (zip or JSON body)
    POST /api/workbench/artifacts/from-probe      {probe, name} save a probe's direction as a direction artifact
    POST /v1/chat/completions, GET /v1/models     the OpenAI-compatible API (steerlab.workbench.openai_api)

Every route that touches the model runs under the shared model lock. On first launch the bundled probes are fitted in
a background thread (cached in ~/.cache/steerlab/inspect/artifacts/probe/, so later launches load them);
`STEERLAB_WORKBENCH_AUTOFIT=0` turns that off. Like every inspect route, these are reachable from loopback only
(the app's `TrustedHostMiddleware`, and the server binds 127.0.0.1).
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

from fastapi import Request
from pydantic import BaseModel, Field

from steerlab.workbench.artifacts import KINDS, ArtifactError, slug
from steerlab.workbench.loom import Loom, LoomError
from steerlab.workbench.openai_api import ChatRequest, chunk, completion_id, extras, finish_reason, models, response
from steerlab.workbench.probes import spec_from_examples
from steerlab.workbench.session import Sampling, SteerError, Workbench

if TYPE_CHECKING:
    from fastapi import FastAPI

    from steerlab.inspect.rooms import RoomContext

ROOM = {"id": "workbench", "title": "Workbench", "order": 50, "key": "w"}
log = logging.getLogger(__name__)
INLINE_LIMIT = 64
BAD_REQUEST = (ValueError, LoomError, ArtifactError, SteerError, KeyError)


class SamplingBody(BaseModel):
    max_tokens: int = Field(default=96, ge=1, le=512)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int | None = None

    def to_sampling(self) -> Sampling:
        return Sampling(max_tokens=self.max_tokens, temperature=self.temperature, top_p=self.top_p, seed=self.seed)


class GenerateBody(BaseModel):
    loom: str
    parent: str
    steer: str | list[dict[str, Any]] | None = None
    edits: list[str | dict[str, Any]] | None = None
    probes: list[str] | None = None
    sampling: SamplingBody = Field(default_factory=SamplingBody)
    units: bool = True
    label: str = ""


class ReplayBody(BaseModel):
    loom: str
    node: str
    steer: str | list[dict[str, Any]] | None = None
    edits: list[str | dict[str, Any]] | None = None


class NewLoom(BaseModel):
    title: str = Field(default="", max_length=120)
    system: str = Field(default="", max_length=4000)


class TurnBody(BaseModel):
    parent: str | None = None
    role: str = "user"
    content: str = Field(min_length=1, max_length=8000)


class ForkBody(BaseModel):
    node: str
    content: str = Field(min_length=1, max_length=8000)


class NodeBody(BaseModel):
    node: str


class FitBody(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    positive: list[str] = Field(min_length=2, max_length=64)
    negative: list[str] = Field(min_length=2, max_length=64)
    prompt: str = Field(default="Say something.", max_length=1000)
    positive_label: str = Field(default="positive", max_length=40)
    negative_label: str = Field(default="negative", max_length=40)
    layer: int | None = None


class ReadBody(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    prompt: str = Field(default="Say something.", max_length=1000)
    probes: list[str] | None = None


class SaveBody(BaseModel):
    meta: dict[str, Any]
    arrays: dict[str, list[Any]] | None = None


class ExportBody(BaseModel):
    keys: list[str] = Field(min_length=1, max_length=500)


class FromProbeBody(BaseModel):
    probe: str
    name: str = ""


def trim(meta: dict[str, Any]) -> dict[str, Any]:
    """An artifact's JSON for listing: long inline number lists (an edit's axis) shown as their length."""
    return {
        key: (f"[{len(value)} numbers]" if isinstance(value, list) and len(value) > INLINE_LIMIT else value)
        for key, value in meta.items()
    }


def register(app: FastAPI, ctx: RoomContext) -> None:
    from fastapi import HTTPException
    from fastapi.responses import JSONResponse, Response, StreamingResponse

    inspector = ctx.inspector
    bench = Workbench(inspector.model, inspector.tokenizer, ctx.model_id, ctx.cache_dir)
    bench.looms.root = ctx.cache_dir / "workbench" / ctx.model_id.replace("/", "--") / "looms"
    app.state.workbench = bench

    def guarded(compute: Callable[[], Any]) -> Any:
        with ctx.lock:
            try:
                return compute()
            except BAD_REQUEST as error:
                raise HTTPException(status_code=400, detail=str(error).strip("'\"")) from None

    def fit_in_background() -> None:
        try:
            with ctx.lock:
                bench.fit_bundle()
        except Exception as error:  # reported through /state; the room still works without probes
            log.warning("steerlab inspect: fitting the bundled probes failed (%s)", error)

    if os.environ.get("STEERLAB_WORKBENCH_AUTOFIT", "1") != "0":
        bench.fit_status["state"] = "queued"
        threading.Thread(target=fit_in_background, name="workbench-fit", daemon=True).start()

    def stream(job: Callable[[Callable[[dict[str, Any]], None]], Any]) -> Iterator[tuple[str, Any]]:
        """Run `job(on_token)` under the model lock in a thread; yield ("token", record)... then ("done", result)."""
        events: queue.Queue[tuple[str, Any]] = queue.Queue()
        finished = threading.Event()

        def worker() -> None:
            try:
                with ctx.lock:
                    result = job(lambda record: events.put(("token", record)))
                events.put(("done", result))
            except BAD_REQUEST as error:
                events.put(("error", str(error).strip("'\"")))
            except Exception as error:
                log.exception("workbench generation failed")
                events.put(("error", f"{type(error).__name__}: {error}"))
            finally:
                finished.set()

        threading.Thread(target=worker, name="workbench-generate", daemon=True).start()
        try:
            while True:
                kind, payload = events.get()
                yield kind, payload
                if kind in ("done", "error"):
                    return
        finally:
            if not finished.is_set():
                bench.cancel()  # the client went away mid-reply

    # State

    @app.get("/api/workbench/state")
    def state() -> dict[str, Any]:
        return {
            "model": ctx.model_id,
            "n_layers": bench.n_layers,
            "d_model": bench._d_model(),
            "probe_layer": bench.probe_layer,
            "fit": dict(bench.fit_status),
            "probes": [probe.summary() for probe in bench.bank.probes.values()],
            "steerables": bench.steerables(),
            "messages": bench.messages[-10:],
        }

    # Generation

    @app.post("/api/workbench/generate")
    def generate(body: GenerateBody) -> StreamingResponse:
        def job(on_token: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
            loom = bench.looms.load(body.loom)
            parent = loom.get(body.parent)
            if parent.role == "assistant":
                raise LoomError("a reply follows a user (or system) turn; add a user turn first")
            result = bench.generate(
                loom.messages(body.parent),
                steer=body.steer,
                edits=body.edits,
                probes=body.probes,
                sampling=body.sampling.to_sampling(),
                units=body.units,
                on_token=on_token,
            )
            loom = bench.looms.load(body.loom)  # re-read: the tree may have changed while the reply was written
            meta = {key: result[key] for key in ("steer", "steer_request", "edits", "edit_names", "sampling")}
            meta.update(
                tokens=result["tokens"],
                token_ids=result["token_ids"],
                finish_reason=result["finish_reason"],
                probes=result["probes"],
                messages=result["messages"],
                seconds=result["seconds"],
                label=body.label,
            )
            node = loom.add("assistant", result["text"], parent=body.parent, meta=meta)
            if loom.title in ("", "New conversation"):
                loom.title = loom.get(body.parent).content[:60]
            bench.looms.save(loom)
            return {"node": node.to_dict(), "loom": {"id": loom.id, "title": loom.title}}

        def lines() -> Iterator[str]:
            for kind, payload in stream(job):
                if kind == "token":
                    yield json.dumps({"type": "token", **payload}) + "\n"
                elif kind == "done":
                    yield json.dumps({"type": "done", **payload}) + "\n"
                else:
                    yield json.dumps({"type": "error", "detail": payload}) + "\n"

        return StreamingResponse(lines(), media_type="application/x-ndjson")

    @app.post("/api/workbench/cancel")
    def cancel() -> dict[str, Any]:
        bench.cancel()
        return {"ok": True}

    @app.post("/api/workbench/replay")
    def replay(body: ReplayBody) -> dict[str, Any]:
        def compute() -> dict[str, Any]:
            loom = bench.looms.load(body.loom)
            node = loom.get(body.node)
            if node.role != "assistant" or node.parent is None:
                raise LoomError("replay reads a model reply")
            ids = node.meta.get("token_ids", [])
            return bench.replay(loom.messages(node.parent), ids, steer=body.steer, edits=body.edits)

        result: dict[str, Any] = guarded(compute)
        return result

    # Looms

    @app.get("/api/workbench/looms")
    def looms() -> dict[str, Any]:
        return {"looms": bench.looms.list()}

    @app.post("/api/workbench/looms")
    def new_loom(body: NewLoom) -> dict[str, Any]:
        loom = bench.looms.new(body.title)
        if body.system.strip():
            loom.add("system", body.system.strip())
            bench.looms.save(loom)
        return loom.to_dict()

    @app.post("/api/workbench/looms/import")
    def import_loom(data: dict[str, Any]) -> dict[str, Any]:
        def compute() -> dict[str, Any]:
            loom = Loom.from_dict(data)
            fresh = bench.looms.new(loom.title or "Imported conversation")
            loom.id = fresh.id
            bench.looms.save(loom)
            return loom.to_dict()

        result: dict[str, Any] = guarded(compute)
        return result

    @app.get("/api/workbench/looms/{loom_id}")
    def get_loom(loom_id: str) -> dict[str, Any]:
        result: dict[str, Any] = guarded(lambda: bench.looms.load(loom_id).to_dict())
        return result

    @app.delete("/api/workbench/looms/{loom_id}")
    def delete_loom(loom_id: str) -> dict[str, Any]:
        guarded(lambda: bench.looms.delete(loom_id))
        return {"ok": True}

    def mutate(loom_id: str, change: Callable[[Loom], Any]) -> dict[str, Any]:
        def compute() -> dict[str, Any]:
            loom = bench.looms.load(loom_id)
            out = change(loom)
            bench.looms.save(loom)
            return {"loom": loom.to_dict(), "result": out}

        result: dict[str, Any] = guarded(compute)
        return result

    @app.post("/api/workbench/looms/{loom_id}/turn")
    def add_turn(loom_id: str, body: TurnBody) -> dict[str, Any]:
        return mutate(loom_id, lambda loom: loom.add(body.role, body.content, parent=body.parent).id)

    @app.post("/api/workbench/looms/{loom_id}/fork")
    def fork(loom_id: str, body: ForkBody) -> dict[str, Any]:
        return mutate(loom_id, lambda loom: loom.fork(body.node, body.content).id)

    @app.post("/api/workbench/looms/{loom_id}/remove")
    def remove(loom_id: str, body: NodeBody) -> dict[str, Any]:
        return mutate(loom_id, lambda loom: loom.remove(body.node))

    @app.get("/api/workbench/looms/{loom_id}/compare")
    def compare(loom_id: str, a: str, b: str) -> dict[str, Any]:
        result: dict[str, Any] = guarded(lambda: bench.looms.load(loom_id).compare(a, b))
        return result

    # Probes

    @app.get("/api/workbench/probes")
    def probes() -> dict[str, Any]:
        return {
            "layer": bench.probe_layer,
            "n_layers": bench.n_layers,
            "fit": dict(bench.fit_status),
            "probes": [probe.summary() for probe in bench.bank.probes.values()],
            "bundle": bench.bundle_names(),
        }

    @app.post("/api/workbench/probes/fit")
    def fit(body: FitBody) -> dict[str, Any]:
        def compute() -> dict[str, Any]:
            spec = spec_from_examples(
                body.name,
                body.positive,
                body.negative,
                prompt=body.prompt,
                positive_label=body.positive_label,
                negative_label=body.negative_label,
            )
            return bench.fit_custom(spec, body.layer).summary()

        result: dict[str, Any] = guarded(compute)
        return result

    @app.post("/api/workbench/probes/refit")
    def refit() -> dict[str, Any]:
        """Fit any bundled probe that is missing (after a delete, or with autofit off)."""
        fitted: list[str] = guarded(bench.fit_bundle)
        return {"fitted": fitted}

    @app.post("/api/workbench/probes/read")
    def read(body: ReadBody) -> dict[str, Any]:
        result: dict[str, Any] = guarded(lambda: bench.read_text(body.text, prompt=body.prompt, probes=body.probes))
        return result

    @app.delete("/api/workbench/probes/{name}")
    def delete_probe(name: str) -> dict[str, Any]:
        def compute() -> None:
            bench.bank.remove(name)
            bench.registry.directions.pop("probe:" + name, None)

        guarded(compute)
        return {"ok": True}

    # Artifacts

    @app.get("/api/workbench/artifacts")
    def artifacts(type: str | None = None) -> dict[str, Any]:
        rows: list[dict[str, Any]] = guarded(lambda: bench.artifacts.list(type))
        return {"kinds": list(KINDS), "artifacts": [trim(meta) for meta in rows]}

    @app.get("/api/workbench/artifacts/{kind}/{name}")
    def artifact(kind: str, name: str) -> dict[str, Any]:
        def compute() -> dict[str, Any]:
            meta, arrays = bench.artifacts.load(kind, name)
            return {"meta": trim(meta), "arrays": {k: list(v.shape) for k, v in arrays.items()}}

        result: dict[str, Any] = guarded(compute)
        return result

    @app.delete("/api/workbench/artifacts/{kind}/{name}")
    def delete_artifact(kind: str, name: str) -> dict[str, Any]:
        def compute() -> None:
            bench.artifacts.delete(kind, name)
            if kind == "direction":
                bench.registry.directions.pop(name, None)
            if kind == "probe":
                gone = [k for k in bench.bank.probes if bench.bank.key(k) == name]
                for key in gone:
                    bench.bank.probes.pop(key)
                    bench.registry.directions.pop("probe:" + key, None)

        guarded(compute)
        return {"ok": True}

    def after_change() -> None:
        bench.bank.load_cached()
        bench.refresh_directions()

    @app.post("/api/workbench/artifacts")
    def save_artifact(body: SaveBody) -> dict[str, Any]:
        def compute() -> dict[str, Any]:
            meta = dict(body.meta)
            meta.setdefault("model", ctx.model_id)
            stored = bench.artifacts.save(meta, body.arrays)
            after_change()
            return trim(stored)

        result: dict[str, Any] = guarded(compute)
        return result

    @app.post("/api/workbench/artifacts/from-probe")
    def from_probe(body: FromProbeBody) -> dict[str, Any]:
        def compute() -> dict[str, Any]:
            probe = bench.bank.probes.get(body.probe)
            if probe is None:
                raise ArtifactError(f"no probe named {body.probe!r}")
            meta = {
                "type": "direction",
                "name": slug(body.name or f"{probe.name}-direction"),
                "model": ctx.model_id,
                "layer": probe.layer,
                "scale": probe.class_gap,
                "concept": probe.name,
                "made_from": f"probe {probe.name} (difference of means, d' {probe.d_prime:.2f})",
            }
            stored = bench.artifacts.save(meta, {"vector": probe.direction})
            after_change()
            return trim(stored)

        result: dict[str, Any] = guarded(compute)
        return result

    @app.post("/api/workbench/artifacts/export")
    def export(body: ExportBody) -> Response:
        blob: bytes = guarded(lambda: bench.artifacts.export(body.keys))
        name = "steerlab-artifacts.zip" if len(body.keys) > 1 else body.keys[0].replace("/", "-") + ".zip"
        disposition = {"Content-Disposition": f'attachment; filename="{name}"'}
        return Response(blob, media_type="application/zip", headers=disposition)

    @app.post("/api/workbench/artifacts/import")
    async def import_artifacts(request: Request) -> dict[str, Any]:
        blob = await request.body()
        if not blob:
            raise HTTPException(status_code=400, detail="send a zip bundle or an artifact's JSON as the body")

        def compute() -> dict[str, Any]:
            keys = bench.artifacts.import_bundle(blob) if blob[:2] == b"PK" else [bench.artifacts.import_files(blob)]
            after_change()
            return {"imported": keys}

        result: dict[str, Any] = guarded(compute)
        return result

    # The OpenAI-compatible API

    @app.get("/v1/models")
    def v1_models() -> dict[str, Any]:
        return models(ctx.model_id)

    @app.post("/v1/chat/completions", response_model=None)
    def v1_chat(body: ChatRequest) -> Any:
        if body.model not in (None, "", ctx.model_id):
            return JSONResponse(
                status_code=404,
                content={
                    "error": {
                        "message": f"model {body.model!r} is not loaded; this server runs {ctx.model_id!r}",
                        "type": "invalid_request_error",
                        "code": "model_not_found",
                    }
                },
            )
        names = body.probe_names(list(bench.bank.probes))

        def job(on_token: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
            return bench.generate(
                body.chat_messages(),
                steer=body.steer,
                edits=body.edits,
                probes=names,
                sampling=body.sampling(),
                units=False,
                on_token=on_token,
            )

        if not body.stream:
            try:
                with ctx.lock:
                    result = job(lambda record: None)
            except BAD_REQUEST as error:
                return JSONResponse(
                    status_code=400,
                    content={"error": {"message": str(error).strip("'\""), "type": "invalid_request_error"}},
                )
            return response(body, result, ctx.model_id, names)

        cid = completion_id()

        def events() -> Iterator[str]:
            yield "data: " + json.dumps(chunk(cid, ctx.model_id, {"role": "assistant", "content": ""})) + "\n\n"
            for kind, payload in stream(job):
                if kind == "token":
                    if payload["text"]:
                        yield "data: " + json.dumps(chunk(cid, ctx.model_id, {"content": payload["text"]})) + "\n\n"
                elif kind == "done":
                    last = chunk(cid, ctx.model_id, {}, finish_reason(payload), steerlab=extras(payload, names))
                    yield "data: " + json.dumps(last) + "\n\n"
                else:
                    yield "data: " + json.dumps({"error": {"message": payload, "type": "server_error"}}) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")
