"""The Edit room: concept edits in the weights (mirror, project out, amplify along an axis), with their controls.

    GET    /api/edit/info               model id and its Units-room slug, shape, exactness, the example, the edit list
    POST   /api/edit/units              {axis, units}            the units an edit would touch, with their loadings
    POST   /api/edit/compare            {axis, units, k, ...}    before / after / random axis / random units
    POST   /api/edit/restore            weights back and checked: logits bit-identical to the startup fingerprint
    GET    /api/edit/stack              the shared edit list (Trace's neuron edits and the Edit room's weight edits)
    POST   /api/edit/stack              {axis, units, k, ...}    add a weight edit (units resolved now, so it is fixed)
    POST   /api/edit/stack/neurons      {edits: [{layer, neuron, mode, value}]}   add the Trace room's edits
    POST   /api/edit/stack/{id}/toggle  switch one record on or off
    DELETE /api/edit/stack/{id}, /api/edit/stack
    GET    /api/edit/stack/{id}/artifact   the record as an artifact JSON (schema: steerlab.edits.stack)
    POST   /api/edit/stack/{id}/save    {name}   save it to the shared artifact store (edit/<name>.json + .npz axes),
                                        where the Workbench chat and /v1/chat/completions `edits: [name]` find it
    GET    /api/edit/saved              the edit artifacts in that store
    POST   /api/edit/load               {name}   add a stored edit to the list
    POST   /api/edit/import             {artifact}               add an edit JSON (axes rebuilt from axis_source)

The shared list is `app.state.edits` (created when this room registers). The Trace room runs its views under it.

The model is never left edited between requests: every comparison applies the enabled records of the list plus
the edit being tried, and restores the weights before it returns. `restore` re-checks that with a fingerprint.

The numerics live in `steerlab.edits` (reflection, checks, stack); this module only parses requests.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

# Fails where steerlab.edits is not shipped (the public mirror), so discover() skips this room there.
import steerlab.edits.reflection  # noqa: F401

if TYPE_CHECKING:
    from fastapi import FastAPI

    from steerlab.edits.reflection import Axes, Units
    from steerlab.edits.stack import EditRecord, EditStack
    from steerlab.inspect.rooms import RoomContext

ROOM = {"id": "edit", "title": "Edit", "order": 40, "key": "e"}
MAX_PAIRS = 32
MAX_PROMPTS = 4
MAX_UNITS_PER_LAYER = 4096

EXAMPLE = {
    "title": "Mother ↔ father (neuron-reflection §2)",
    "axis": {
        "kind": "sentences",
        "pairs": [
            ["I visited my mother last week.", "I visited my father last week."],
            ["We bought a gift for our mother.", "We bought a gift for our father."],
            ["She talked to her mother about school.", "She talked to her father about school."],
            ["The children ran to their mother.", "The children ran to their father."],
            ["He wrote a letter to his mother.", "He wrote a letter to his father."],
            ["My mother lives near the sea.", "My father lives near the sea."],
        ],
    },
    "units": {"selection": "top", "kind": "neurons", "per_layer": 100},
    "k": 2.0,
    "prompts": [
        "Who gives birth to a baby, the mother or the father? Answer briefly.",
        "Who was the mother of Jesus? One sentence.",
    ],
    "checks": [
        {
            "kind": "choice",
            "question": "Who gives birth to a baby? Answer with one word: Mother or Father.",
            "a": "Mother",
            "b": "Father",
        },
        {
            "kind": "choice",
            "question": "Who was the mother of Jesus? Answer with one word: Mary or Joseph.",
            "a": "Mary",
            "b": "Joseph",
        },
        {"kind": "words", "source": ["mother", "mom", "mothers"], "target": ["father", "dad", "fathers"]},
        {"kind": "probe"},
    ],
}


class AxisSpec(BaseModel):
    kind: Literal["pairs", "sentences", "template"] = "pairs"
    method: Literal["activations", "embed", "lens"] = "activations"
    pairs: list[tuple[str, str]] = Field(min_length=1, max_length=MAX_PAIRS)

    def source(self) -> dict[str, Any]:
        src: dict[str, Any] = {"kind": self.kind, "pairs": [list(p) for p in self.pairs]}
        if self.kind == "pairs":
            src["method"] = self.method
        return src


class UnitSpec(BaseModel):
    selection: Literal["top", "ids", "all"] = "top"
    kind: Literal["neurons", "heads", "both"] = "neurons"
    per_layer: int = Field(default=30, ge=1, le=MAX_UNITS_PER_LAYER)
    heads_per_layer: int = Field(default=2, ge=1, le=256)
    layers: list[int] | None = None
    ids: list[str] = Field(default_factory=list, max_length=20000)


class EditSpec(BaseModel):
    axis: AxisSpec
    units: UnitSpec = Field(default_factory=UnitSpec)
    k: float = Field(default=2.0, ge=-8.0, le=8.0)
    sides: Literal["both", "reads", "writes"] = "both"
    parts: str = Field(default="qkvo", pattern=r"^[qkvo]{1,4}$")
    label: str = Field(default="", max_length=120)


class CompareBody(EditSpec):
    prompts: list[str] = Field(min_length=1, max_length=MAX_PROMPTS)
    checks: list[dict[str, Any]] = Field(default_factory=list, max_length=8)
    controls: bool = True
    chat: bool = True
    sample: bool = False
    temperature: float = Field(default=0.8, gt=0.0, le=2.0)
    seed: int = 0
    max_new_tokens: int = Field(default=40, ge=1, le=120)


class NeuronItem(BaseModel):
    layer: int
    neuron: int
    mode: Literal["off", "flip", "scale", "set"] = "off"
    value: float = Field(default=0.0, ge=-1000, le=1000)


class NeuronsBody(BaseModel):
    edits: list[NeuronItem] = Field(default_factory=list, max_length=32)
    replace: bool = True


class ImportBody(BaseModel):
    artifact: dict[str, Any]


class RoomState:
    """The room's memory: the editor around the shared model, built axes (by source), the shared edit list and the
    startup fingerprint the restore check compares against."""

    def __init__(self, ctx: RoomContext, stack: EditStack) -> None:
        from steerlab.edits.reflection import Editor
        from steerlab.workbench.artifacts import ArtifactStore

        self.ctx = ctx
        self.store = ArtifactStore(ctx.cache_dir / "artifacts")  # the Workbench's store
        self.model = ctx.inspector.model
        self.tokenizer = ctx.inspector.tokenizer
        self.editor = Editor(self.model)
        self.stack = stack
        self._axes: dict[str, Axes] = {}
        ids = self.tokenizer("The capital of France is", add_special_tokens=False).input_ids[:16]
        self.fingerprint_ids = ids or [0]
        self.start = self._fingerprint()

    def _fingerprint(self) -> Any:
        from steerlab.edits.checks import fingerprint

        return fingerprint(self.model, self.fingerprint_ids)

    def axes(self, spec: AxisSpec) -> Axes:
        from steerlab.edits.reflection import build_axes

        source = spec.source()
        key = hashlib.sha1(json.dumps(source, sort_keys=True).encode()).hexdigest()
        if key not in self._axes:
            if len(self._axes) > 16:
                self._axes.clear()
            self._axes[key] = build_axes(self.model, self.tokenizer, source)
        return self._axes[key]

    def units(self, axes: Axes, spec: UnitSpec, sides: str, parts: str) -> Units:
        from steerlab.edits.reflection import Units, select_top

        n = self.editor.arch.n_layers
        if spec.layers is not None:
            bad = [layer for layer in spec.layers if not 0 <= layer < n]
            if bad:
                raise ValueError(f"layers {bad} are outside 0..{n - 1}")
        if spec.selection == "ids":
            units = Units.parse(spec.ids, self.ctx.model_id)
            if not units.ids():
                raise ValueError("list at least one unit, like L8.N344 or L8.H3")
            self.editor.check(units)
            return units
        if spec.selection == "all":
            scope = spec.layers if spec.layers is not None else range(n)
            units = Units()
            if spec.kind in ("neurons", "both"):
                units.neurons = {layer: list(range(self.editor.arch.d_mlp(layer))) for layer in scope}
            if spec.kind in ("heads", "both"):
                units.heads = {layer: list(range(self.editor.arch.n_heads)) for layer in scope}
            return units
        units = Units()
        side: Any = sides
        if spec.kind in ("neurons", "both"):
            units.neurons = select_top(self.editor, axes, spec.per_layer, layers=spec.layers, sides=side).neurons
        if spec.kind in ("heads", "both"):
            picked = select_top(self.editor, axes, spec.heads_per_layer, layers=spec.layers, kind="heads", parts=parts)
            units.heads = picked.heads
        return units

    def summary(self, axes: Axes, units: Units) -> dict[str, Any]:
        """Per layer: how many units, and their median loading against the layer's median neuron (how much more than
        a typical neuron of that layer these weights point along the axis)."""
        rows = []
        for layer in units.layers():
            row: dict[str, Any] = {
                "layer": layer,
                "neurons": len(units.neurons.get(layer, [])),
                "heads": len(units.heads.get(layer, [])),
            }
            if units.neurons.get(layer):
                load = self.editor.neuron_shares(layer, *axes.neuron_axes(layer)).max(0).values
                chosen = load[units.neurons[layer]]
                row["share"] = float(chosen.median())
                row["vs_layer"] = float(chosen.median() / load.median().clamp_min(1e-12))
            rows.append(row)
        from steerlab.edits.reflection import unit_href

        model = self.ctx.model_id
        links = [
            {"id": f"L{layer}.N{n}", "href": unit_href(model, "neuron", layer, n)}
            for layer in sorted(units.neurons)
            for n in units.neurons[layer]
        ]
        links += [
            {"id": f"L{layer}.H{h}", "href": unit_href(model, "head", layer, h)}
            for layer in sorted(units.heads)
            for h in units.heads[layer]
        ]
        ids = units.ids()
        return {
            "count": units.count(),
            "layers": rows,
            "ids": ids[:60],
            "links": links[:60],
            "more": max(0, len(ids) - 60),
        }

    def restore_check(self) -> dict[str, Any]:
        import torch

        self.editor.restore()
        now = self._fingerprint()
        return {
            "restored": bool(torch.equal(now, self.start)),
            "max_abs_diff": float((now - self.start).abs().max()),
            "exact": self.editor.exact,
        }


def _record(body: EditSpec, axes: Axes, units: Units) -> EditRecord:
    from steerlab.edits.stack import EditRecord

    scope = body.units.model_dump()
    scope.pop("ids", None)
    return EditRecord("reflect", body.label, units, dict(axes.source), body.k, body.sides, body.parts, scope=scope)


class SaveBody(BaseModel):
    name: str = Field(default="", max_length=80)


class LoadBody(BaseModel):
    name: str = Field(min_length=1, max_length=80)


def register(app: FastAPI, ctx: RoomContext) -> None:
    from steerlab.edits.stack import EditStack

    holder: dict[str, RoomState] = {}
    shared = EditStack()
    app.state.edits = shared  # the shared list, for the Trace and Workbench rooms

    def room() -> RoomState:
        if "state" not in holder:
            holder["state"] = RoomState(ctx, shared)
        return holder["state"]

    def add_loaded(s: RoomState, meta: dict[str, Any], arrays: dict[str, Any] | None) -> None:
        from steerlab.edits.stack import load_artifact

        model = meta.get("model")
        if model and model != ctx.model_id:
            raise ValueError(f"this edit was made on {model}; the loaded model is {ctx.model_id}")
        record, axes = load_artifact(meta, arrays, s.editor.arch.d_model)
        s.editor.check(record.units)
        s.stack.add(record, axes)
        if record.kind == "reflect" and axes is None:
            try:
                s.stack.axes_for(record, s.model, s.tokenizer)  # rebuild now, so a bad source fails here
            except Exception:
                s.stack.remove(record.id)
                raise

    def stack_json() -> dict[str, Any]:
        return {"stack": room().stack.to_json()}

    @app.get("/api/edit/info")
    def info() -> dict[str, Any]:
        from steerlab.edits.reflection import lens_available, model_slug

        def compute() -> dict[str, Any]:
            s = room()
            arch = s.editor.arch
            return {
                "model": ctx.model_id,
                "slug": model_slug(ctx.model_id),  # how the Units room names it (its ?model= links)
                "family": arch.family,
                "n_layers": arch.n_layers,
                "n_heads": arch.n_heads,
                "d_mlp": arch.d_mlp(0),
                "exact": s.editor.exact,
                "lens": lens_available(),
                "example": EXAMPLE,
                "stack": s.stack.to_json(),
            }

        return ctx.view(compute)

    @app.post("/api/edit/units")
    def units(body: EditSpec) -> dict[str, Any]:
        def compute() -> dict[str, Any]:
            s = room()
            axes = s.axes(body.axis)
            return s.summary(axes, s.units(axes, body.units, body.sides, body.parts))

        return ctx.view(compute)

    @app.post("/api/edit/compare")
    def compare(body: CompareBody) -> dict[str, Any]:
        from steerlab.edits.checks import compare as run_compare

        def compute() -> dict[str, Any]:
            s = room()
            axes = s.axes(body.axis)
            chosen = s.units(axes, body.units, body.sides, body.parts)
            sides: Any = body.sides
            result = run_compare(
                s.editor,
                s.tokenizer,
                axes,
                chosen,
                prompts=body.prompts,
                checks=body.checks,
                k=body.k,
                sides=sides,
                parts=body.parts,
                controls=body.controls,
                seed=body.seed,
                chat=body.chat,
                max_new_tokens=body.max_new_tokens,
                sample=body.sample,
                temperature=body.temperature,
                base=s.stack.base(s.model, s.tokenizer, s.editor),
            )
            result["summary"] = s.summary(axes, chosen)
            result["base"] = [r.label for r in s.stack.records if r.enabled]
            result["artifact"] = _record(body, axes, chosen).to_artifact(ctx.model_id)
            return result

        return ctx.view(compute)

    @app.post("/api/edit/restore")
    def restore() -> dict[str, Any]:
        return ctx.view(lambda: room().restore_check())

    @app.get("/api/edit/stack")
    def get_stack() -> dict[str, Any]:
        return ctx.view(stack_json)

    @app.post("/api/edit/stack")
    def add(body: EditSpec) -> dict[str, Any]:
        def compute() -> dict[str, Any]:
            s = room()
            axes = s.axes(body.axis)
            s.stack.add(_record(body, axes, s.units(axes, body.units, body.sides, body.parts)), axes)
            return stack_json()

        return ctx.view(compute)

    @app.post("/api/edit/stack/neurons")
    def add_neurons(body: NeuronsBody) -> dict[str, Any]:
        def compute() -> dict[str, Any]:
            s = room()
            n_layers, d_mlp = s.editor.arch.n_layers, s.editor.arch.d_mlp(0)
            for item in body.edits:
                if not 0 <= item.layer < n_layers or not 0 <= item.neuron < d_mlp:
                    raise ValueError(f"L{item.layer}.N{item.neuron} is not a neuron of this model")
            if body.replace:
                for record in [r for r in s.stack.records if r.kind == "neuron"]:
                    s.stack.remove(record.id)
            for item in body.edits:
                s.stack.add_neuron(item.layer, item.neuron, item.mode, item.value)
            return stack_json()

        return ctx.view(compute)

    @app.post("/api/edit/stack/{edit_id}/toggle")
    def toggle(edit_id: str) -> dict[str, Any]:
        def compute() -> dict[str, Any]:
            record = room().stack.get(edit_id)
            record.enabled = not record.enabled
            return stack_json()

        return ctx.view(compute)

    @app.delete("/api/edit/stack/{edit_id}")
    def remove(edit_id: str) -> dict[str, Any]:
        def compute() -> dict[str, Any]:
            room().stack.remove(edit_id)
            return stack_json()

        return ctx.view(compute)

    @app.delete("/api/edit/stack")
    def clear() -> dict[str, Any]:
        def compute() -> dict[str, Any]:
            room().stack.clear()
            return stack_json()

        return ctx.view(compute)

    @app.get("/api/edit/stack/{edit_id}/artifact")
    def artifact(edit_id: str) -> dict[str, Any]:
        return ctx.view(lambda: room().stack.get(edit_id).to_artifact(ctx.model_id))

    @app.post("/api/edit/stack/{edit_id}/save")
    def save(edit_id: str, body: SaveBody) -> dict[str, Any]:
        from steerlab.edits.stack import slug

        def compute() -> dict[str, Any]:
            s = room()
            record = s.stack.get(edit_id)
            name = slug(body.name or record.label)
            meta, arrays = s.stack.artifact(record, ctx.model_id, name, model_obj=s.model, tokenizer=s.tokenizer)
            stored = s.store.save(meta, arrays or None)
            return {"saved": f"edit/{name}", "arrays": stored["arrays"]}

        return ctx.view(compute)

    @app.get("/api/edit/saved")
    def saved() -> dict[str, Any]:
        def compute() -> dict[str, Any]:
            rows = room().store.list("edit")
            keep = ("name", "label", "model", "kind", "mode", "k", "created")
            return {"saved": [{key: row.get(key) for key in keep} for row in rows]}

        return ctx.view(compute)

    @app.post("/api/edit/load")
    def load(body: LoadBody) -> dict[str, Any]:
        def compute() -> dict[str, Any]:
            s = room()
            meta, arrays = s.store.load("edit", body.name)
            add_loaded(s, meta, arrays)
            return stack_json()

        return ctx.view(compute)

    @app.post("/api/edit/import")
    def import_artifact(body: ImportBody) -> dict[str, Any]:
        def compute() -> dict[str, Any]:
            s = room()
            add_loaded(s, dict(body.artifact), None)
            return stack_json()

        return ctx.view(compute)
