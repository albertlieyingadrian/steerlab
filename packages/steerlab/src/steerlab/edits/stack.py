"""One edit list for every room: the Trace room's neuron edits, the Edit room's weight edits, and their artifact JSON.

Two kinds of record:

    reflect   a weight edit (steerlab.edits.reflection): an axis source, resolved units, k, sides, head parts
    neuron    the Trace room's activation edit on one neuron: off, flip, scale by `value`, or set to `value`,
              applied with a hook on the MLP's down projection at every position (same as steerlab.inspect.core.Edit)

    stack = EditStack()
    stack.add_reflection(axes, units, k=2.0, label="mother ↔ father")
    stack.add_neuron(20, 1639, "off")
    with stack.applied(model, tokenizer, editor):    # every enabled record, in order; restored on exit
        ...
    meta, arrays = stack.artifact(stack.records[0], "Qwen/Qwen3-0.6B", "mother-father")   # for ArtifactStore.save

The artifact (one schema for the Edit room, the Workbench and `/v1/chat/completions`; saved through
`steerlab.workbench.artifacts.ArtifactStore` as `edit/<name>.json` + `edit/<name>.npz`):

    {"type": "edit", "name": "mother-father", "model": "Qwen/Qwen3-0.6B",
     "kind": "reflect",                                  # or "neuron" (a Trace activation edit)
     "mode": "reflect" | "ablate" | "amplify" | "dial",  # a name for k: 2, 1, < 0, anything else
     "k": 2.0, "sides": "both" | "reads" | "writes", "parts": "qkvo",
     "units": [{"kind": "neuron", "layer": 8, "index": 344}, {"kind": "head", "layer": 3, "index": 5}],
     "axis_source": {"kind": "sentences" | "pairs" | "template", "pairs": [[a, b], ...], "method": ...},
     "axis_layers": [0, 1, ...],                         # rows of the npz arrays "reads" and "writes" [L, d_model]
     "op": "off" | "flip" | "scale" | "set", "value": 0.0,   # kind "neuron" only
     "label": str, "scope": {...}, "created": unix seconds}

The axes travel as npz arrays `reads` and `writes` (per layer: the MLP-input and layer-output directions). Without
them they are rebuilt from `axis_source` (that needs the tokenizer). The Workbench's first shape is still read: a
single `axis` (inline list or npz array) and `mode`/`k` act on the units' writes only (down_proj columns, o_proj
blocks), which is `sides="writes"`, `parts="o"` with that axis at every layer. `load_artifact` reads all of them.

Importing this module does not import torch.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from steerlab.edits.reflection import HEAD_PARTS, SIDES, Axes, Editor, Units, build_axes

NEURON_MODES = ("off", "flip", "scale", "set")
Kind = Literal["reflect", "neuron"]
_ids = itertools.count(1)


def now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class EditRecord:
    kind: Kind
    label: str = ""
    units: Units = field(default_factory=Units)
    axis_source: dict[str, Any] | None = None
    k: float | None = None
    sides: str = "both"
    parts: str = HEAD_PARTS
    mode: str | None = None
    value: float | None = None
    scope: dict[str, Any] = field(default_factory=dict)
    created: str = field(default_factory=now)
    enabled: bool = True
    id: str = field(default_factory=lambda: f"e{next(_ids)}")

    def __post_init__(self) -> None:
        if self.kind == "reflect":
            if self.sides not in SIDES:
                raise ValueError(f"sides must be one of {SIDES}")
            if not self.parts or not set(self.parts) <= set(HEAD_PARTS):
                raise ValueError(f"head parts must be letters of {HEAD_PARTS!r}")
            if self.k is None:
                self.k = 2.0
        elif self.kind == "neuron":
            if self.mode not in NEURON_MODES:
                raise ValueError(f"neuron edit mode must be one of {NEURON_MODES}")
            if len(self.units.ids()) != 1 or self.units.heads:
                raise ValueError("a neuron edit names exactly one neuron")
            self.value = float(self.value or 0.0)
        else:
            raise ValueError(f"unknown edit kind {self.kind!r}")
        if not self.label:
            self.label = self.default_label()

    @property
    def neuron(self) -> tuple[int, int]:
        layer, idx = next(iter(self.units.neurons.items()))
        return layer, idx[0]

    def default_label(self) -> str:
        if self.kind == "neuron":
            layer, n = self.neuron
            name = f"L{layer}.N{n}"
            return {"off": f"{name} off", "flip": f"{name} flipped", "scale": f"{name} ×{self.value:g}"}.get(
                str(self.mode), f"{name} = {self.value:g}"
            )
        src = self.axis_source or {}
        pairs = src.get("pairs") or []
        concept = " / ".join(str(x) for x in pairs[0][:2]) if pairs else str(src.get("kind", "axis"))
        verb = {2.0: "mirror", 1.0: "project out"}.get(float(self.k or 0), f"k={self.k:g}")
        count = self.units.count()
        return f"{verb} {concept} · {count['neurons']} neurons, {count['heads']} heads"

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "enabled": self.enabled,
            "units": self.units.to_json(),
            "count": self.units.count(),
            "axis_source": self.axis_source,
            "k": self.k,
            "sides": self.sides if self.kind == "reflect" else None,
            "parts": self.parts if self.kind == "reflect" else None,
            "mode": self.mode,
            "value": self.value,
            "scope": self.scope,
            "created": self.created,
        }

    def to_artifact(self, model: str, name: str | None = None) -> dict[str, Any]:
        """The record's artifact JSON (no arrays; `EditStack.artifact` adds the axes)."""
        meta: dict[str, Any] = {
            "type": "edit",
            "name": name or slug(self.label),
            "model": model,
            "kind": self.kind,
            "units": units_list(self.units),
            "label": self.label,
            "scope": self.scope,
            "created": iso_to_epoch(self.created),
        }
        if self.kind == "reflect":
            k = float(self.k or 0.0)
            meta.update(mode=mode_for_k(k), k=k, sides=self.sides, parts=self.parts, axis_source=self.axis_source)
        else:
            meta.update(op=self.mode, value=self.value)
        return meta

    @classmethod
    def from_artifact(cls, data: dict[str, Any]) -> EditRecord:
        """A record from an artifact JSON of either shape (axes, if any, are read by `load_artifact`)."""
        return load_artifact(data)[0]


class EditStack:
    """The ordered edit list the rooms share. Reflection records keep their axes in memory (rebuilt from
    `axis_source` when imported); `applied` puts every enabled record on the model for the duration of a block."""

    def __init__(self) -> None:
        self.records: list[EditRecord] = []
        self._axes: dict[str, Axes] = {}

    def __len__(self) -> int:
        return len(self.records)

    def get(self, edit_id: str) -> EditRecord:
        for record in self.records:
            if record.id == edit_id:
                return record
        raise ValueError(f"no edit {edit_id!r} in the list")

    def add(self, record: EditRecord, axes: Axes | None = None) -> EditRecord:
        self.records.append(record)
        if axes is not None:
            self._axes[record.id] = axes
        return record

    def add_reflection(
        self,
        axes: Axes,
        units: Units,
        k: float = 2.0,
        sides: str = "both",
        parts: str = HEAD_PARTS,
        scope: dict[str, Any] | None = None,
        label: str = "",
    ) -> EditRecord:
        record = EditRecord("reflect", label, units, dict(axes.source), k, sides, parts, scope=dict(scope or {}))
        return self.add(record, axes)

    def add_neuron(self, layer: int, neuron: int, mode: str = "off", value: float = 0.0) -> EditRecord:
        return self.add(EditRecord("neuron", units=Units(neurons={layer: [neuron]}), mode=mode, value=value))

    def remove(self, edit_id: str) -> None:
        record = self.get(edit_id)
        self.records.remove(record)
        self._axes.pop(record.id, None)

    def clear(self) -> None:
        self.records.clear()
        self._axes.clear()

    def axes_for(self, record: EditRecord, model: Any, tokenizer: Any) -> Axes:
        if record.id not in self._axes:
            if not record.axis_source:
                raise ValueError(f"edit {record.id} has no axis")
            self._axes[record.id] = build_axes(model, tokenizer, record.axis_source)
        return self._axes[record.id]

    def neuron_edits(self) -> list[tuple[int, int, str, float]]:
        """The enabled neuron records as (layer, neuron, mode, value): what `steerlab.inspect.core.Edit` takes."""
        out = []
        for record in self.records:
            if record.kind == "neuron" and record.enabled:
                layer, n = record.neuron
                out.append((layer, n, str(record.mode), float(record.value or 0.0)))
        return out

    @contextmanager
    def applied(
        self, model: Any, tokenizer: Any, editor: Editor | None = None, *, skip: tuple[str, ...] = ()
    ) -> Iterator[Editor]:
        """Every enabled record except `skip`, in order: weight edits through an Editor of their own, neuron edits as
        down-projection hooks. Everything is restored (weights bit-identical) and every hook removed on exit."""
        own = Editor(model)
        if editor is not None:
            own._gains = editor._gains
        handles = []
        try:
            for record in self.records:
                if not record.enabled or record.id in skip:
                    continue
                if record.kind == "reflect":
                    axes = self.axes_for(record, model, tokenizer)
                    sides: Any = record.sides
                    own.reflect(axes, record.units, float(record.k or 0.0), sides, record.parts)
                else:
                    layer, n = record.neuron
                    down = own.arch.mlp(layer)[2]
                    hook = neuron_hook(n, str(record.mode), float(record.value or 0.0))
                    handles.append(down.register_forward_pre_hook(hook))
            yield own
        finally:
            for handle in handles:
                handle.remove()
            own.restore()

    def base(
        self, model: Any, tokenizer: Any, editor: Editor | None = None, skip: tuple[str, ...] = ()
    ) -> Callable[[], Any]:
        """A zero-argument context-manager factory for `checks.compare(base=...)`."""
        return lambda: self.applied(model, tokenizer, editor, skip=skip)

    def to_json(self) -> list[dict[str, Any]]:
        return [record.to_json() for record in self.records]

    def artifact(
        self, record: EditRecord, model: str, name: str | None = None, *, model_obj: Any = None, tokenizer: Any = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """(meta, arrays) for `ArtifactStore.save`: the JSON plus the axes as npz arrays `reads` and `writes`."""
        import numpy as np

        meta = record.to_artifact(model, name)
        if record.kind != "reflect":
            return meta, {}
        if record.id not in self._axes and model_obj is not None:
            self.axes_for(record, model_obj, tokenizer)
        axes = self._axes.get(record.id)
        if axes is None:
            return meta, {}
        layers = axes.layers()
        meta["axis_layers"] = layers
        arrays = {
            "reads": np.stack([axes.reads[layer].float().numpy() for layer in layers]),
            "writes": np.stack([axes.writes[layer].float().numpy() for layer in layers]),
        }
        return meta, arrays


def neuron_hook(neuron: int, mode: str, value: float) -> Callable[..., tuple[Any, ...]]:
    """A forward pre-hook on `mlp.down_proj` that changes one neuron's activation at every position."""

    def hook(module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
        acts = args[0].clone()
        col = acts[..., neuron]
        if mode == "off":
            acts[..., neuron] = 0.0
        elif mode == "flip":
            acts[..., neuron] = -col
        elif mode == "scale":
            acts[..., neuron] = col * value
        else:
            acts[..., neuron] = value
        return (acts, *args[1:])

    return hook


def from_inspect_edit(edit: Any) -> EditRecord:
    """A `steerlab.inspect.core.Edit` (layer, neuron, mode, value) as a stack record."""
    units = Units(neurons={int(edit.layer): [int(edit.neuron)]})
    return EditRecord("neuron", units=units, mode=str(edit.mode), value=float(edit.value))


MODE_K: dict[str, float] = {"reflect": 2.0, "ablate": 1.0, "amplify": -1.0}


def mode_for_k(k: float) -> str:
    """The Workbench's names for the dial: reflect (k = 2), ablate (k = 1), amplify (k < 0), else dial."""
    if k == 2.0:
        return "reflect"
    if k == 1.0:
        return "ablate"
    return "amplify" if k < 0 else "dial"


def slug(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip()).strip("-.")
    return out[:80] or "edit"


def iso_to_epoch(value: Any) -> float:
    if isinstance(value, int | float):
        return float(value)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return datetime.now(UTC).timestamp()


def units_list(units: Units) -> list[dict[str, Any]]:
    out = [
        {"kind": "neuron", "layer": layer, "index": i} for layer in sorted(units.neurons) for i in units.neurons[layer]
    ]
    out += [{"kind": "head", "layer": layer, "index": h} for layer in sorted(units.heads) for h in units.heads[layer]]
    return out


def units_from(raw: Any) -> Units:
    """Units from a list of {kind, layer, index} (the artifact shape) or {neurons: {...}, heads: {...}}."""
    if isinstance(raw, dict):
        return Units.from_json(raw)
    units = Units()
    for item in raw or []:
        kind = item.get("kind", "neuron")
        if kind not in ("neuron", "head"):
            raise ValueError(f"unit kind must be neuron or head, not {kind!r}")
        index = int(item.get("index", item.get("neuron", item.get("head", -1))))
        table = units.neurons if kind == "neuron" else units.heads
        table.setdefault(int(item.get("layer", -1)), []).append(index)
    for table in (units.neurons, units.heads):
        for layer in table:
            table[layer] = sorted(set(table[layer]))
    return units


def load_artifact(
    meta: dict[str, Any], arrays: dict[str, Any] | None = None, d_model: int | None = None
) -> tuple[EditRecord, Axes | None]:
    """A record and, when the artifact carries them, its axes. Reads the current shape (axis_source + npz reads /
    writes) and the Workbench's first one (one `axis`, writes only). Raises ValueError on anything malformed."""
    import numpy as np

    if meta.get("type", "edit") != "edit":
        raise ValueError(f"expected an edit artifact, got type {meta.get('type')!r}")
    arrays = arrays or {}
    units = units_from(meta.get("units"))
    created = str(meta.get("created") or now())
    if isinstance(meta.get("created"), int | float):
        created = datetime.fromtimestamp(float(meta["created"]), UTC).replace(microsecond=0).isoformat()
        created = created.replace("+00:00", "Z")
    label = str(meta.get("label") or meta.get("name") or "")
    op = meta.get("op") or (meta.get("mode") if meta.get("mode") in NEURON_MODES else None)
    if meta.get("kind") == "neuron" or op:
        value = meta.get("value")
        record = EditRecord("neuron", label, units, mode=op, value=None if value is None else float(value))
        return record, None
    if meta.get("k") is not None:
        k = float(meta["k"])
    else:
        mode = str(meta.get("mode", "reflect"))
        if mode not in MODE_K:
            raise ValueError(f"edit mode must be one of {sorted(MODE_K)}, not {mode!r}")
        k = MODE_K[mode]
    if not units.ids():
        raise ValueError(f"edit {meta.get('name', '')!r} names no units")
    scope = dict(meta.get("scope") or {})
    legacy = meta.get("axis") if meta.get("axis") is not None else arrays.get("axis")
    if legacy is not None and "reads" not in arrays:
        import torch

        from steerlab.edits.reflection import unit

        axis = np.asarray(legacy, dtype=np.float32).reshape(-1)
        if d_model is not None and axis.shape[0] != d_model:
            raise ValueError(f"the axis has {axis.shape[0]} numbers but the model's stream is {d_model} wide")
        if not np.any(axis):
            raise ValueError("the axis is all zeros")
        d = unit(torch.as_tensor(axis))
        layers = units.layers()
        axes = Axes({layer: d for layer in layers}, {layer: d for layer in layers}, {"kind": "vector"})
        record = EditRecord("reflect", label, units, {"kind": "vector"}, k, "writes", "o", scope=scope, created=created)
        return record, axes
    source = meta.get("axis_source")
    carried: Axes | None = None
    if "reads" in arrays and "writes" in arrays:
        import torch

        reads, writes = np.asarray(arrays["reads"], np.float32), np.asarray(arrays["writes"], np.float32)
        layers = [int(x) for x in meta.get("axis_layers", range(reads.shape[0]))]
        if reads.shape != writes.shape or reads.shape[0] != len(layers):
            raise ValueError("the reads / writes arrays do not match axis_layers")
        if d_model is not None and reads.shape[1] != d_model:
            raise ValueError(f"the axes are {reads.shape[1]} wide but the model's stream is {d_model} wide")
        carried = Axes(
            {layer: torch.from_numpy(reads[i].copy()) for i, layer in enumerate(layers)},
            {layer: torch.from_numpy(writes[i].copy()) for i, layer in enumerate(layers)},
            dict(source or {}),
        )
    elif not source:
        raise ValueError("an edit artifact needs its axes: npz reads/writes, an axis_source, or one axis")
    sides = str(meta.get("sides") or "both")
    parts = str(meta.get("parts") or HEAD_PARTS)
    record = EditRecord("reflect", label, units, source, k, sides, parts, scope=scope, created=created)
    return record, carried
