"""Steering and neuron-edit hooks for de-slopping at runtime, in steerlab's conventions.

    from deslop.steer import load_direction, steering, neuron_edits, NeuronEdit
    direction = load_direction("register")            # steerlab's format: register.npz + register.json
    with steering(model, direction, strength=-1.0), neuron_edits(model, [NeuronEdit(32, 4852, -1.0)]):
        model.generate(...)

A direction is a unit vector at one layer plus its class gap (the distance between the two classes' means along it),
as steerlab's `build_direction` saves it. Strength 1.0 adds one class gap to the layer's output at every position;
negative strengths subtract. A neuron edit multiplies one MLP neuron's activation (the input to `down_proj`) by
`scale`: 0 switches it off and -1 flips its sign, as steerlab inspect's off and flip do. `bake_neuron_edits` writes
the same edit into the weights, so a released model needs no hook.

Needs the `models` extra (torch); numpy comes with it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LAYER_PATHS = ("model.layers", "model.language_model.layers", "base_model.model.model.layers")


@dataclass(frozen=True)
class Direction:
    unit: Any  # numpy float array [d_model], length 1
    layer: int
    class_gap: float
    label: str = "direction"


@dataclass(frozen=True)
class NeuronEdit:
    layer: int
    neuron: int
    scale: float = -1.0

    @classmethod
    def parse(cls, spec: str) -> NeuronEdit:
        """`L32.N4852` (flip), `L32.N4852*0` (off) or `L32.N4852*0.5` (scale)."""
        name, _, scale = spec.partition("*")
        layer, neuron = name.upper().removeprefix("L").split(".N")
        return cls(int(layer), int(neuron), float(scale) if scale else -1.0)


def load_direction(path: str | Path) -> Direction:
    """A direction saved by steerlab (`<path>.npz` with `vector`, `<path>.json` with layer and class gap)."""
    import numpy as np

    base = Path(path)
    base = base.with_suffix("") if base.suffix in (".npz", ".json") else base
    vector = np.load(base.with_name(base.name + ".npz"))["vector"].astype(np.float32)
    meta = json.loads(base.with_name(base.name + ".json").read_text(encoding="utf-8"))
    meta = meta.get("metadata", meta)
    gap = float(meta.get("class_gap") or 1.0)
    label = str(meta.get("concept") or meta.get("label") or base.name)
    return Direction(unit=vector / np.linalg.norm(vector), layer=int(meta["layer"]), class_gap=gap, label=label)


def decoder_layers(model: Any) -> Any:
    for path in LAYER_PATHS:
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        return obj
    raise TypeError(f"can't find the decoder layers of {type(model).__name__}; tried {', '.join(LAYER_PATHS)}")


@contextmanager
def steering(model: Any, direction: Direction, strength: float) -> Iterator[None]:
    """Add `strength` class gaps of `direction` to its layer's output at every position, inside the block."""
    import torch

    layer = decoder_layers(model)[direction.layer]
    param = next(model.parameters())
    delta = torch.as_tensor(direction.unit * direction.class_gap * strength, device=param.device, dtype=param.dtype)

    def hook(module: Any, inputs: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            return (output[0] + delta, *output[1:])
        return output + delta

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def neuron_edits(model: Any, edits: Sequence[NeuronEdit]) -> Iterator[None]:
    """Scale MLP neurons' activations (the input to each layer's `down_proj`) inside the block."""
    by_layer: dict[int, list[NeuronEdit]] = {}
    for edit in edits:
        by_layer.setdefault(edit.layer, []).append(edit)
    layers = decoder_layers(model)
    with ExitStack() as stack:
        for index, layer_edits in by_layer.items():
            neurons = [e.neuron for e in layer_edits]
            scales = [e.scale for e in layer_edits]

            def pre(module: Any, args: tuple[Any, ...], ns: list[int] = neurons, ss: list[float] = scales) -> Any:
                acts = args[0].clone()
                for n, s in zip(ns, ss, strict=True):
                    acts[..., n] = acts[..., n] * s
                return (acts, *args[1:])

            handle = layers[index].mlp.down_proj.register_forward_pre_hook(pre)
            stack.callback(handle.remove)
        yield


def bake_neuron_edits(model: Any, edits: Sequence[NeuronEdit]) -> None:
    """Write the edits into `down_proj` weights (column `neuron` times `scale`): the same effect, no hook."""
    import torch

    layers = decoder_layers(model)
    with torch.no_grad():
        for edit in edits:
            layers[edit.layer].mlp.down_proj.weight[:, edit.neuron] *= edit.scale
