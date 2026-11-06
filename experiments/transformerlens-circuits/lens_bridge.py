"""The small bridge between a steering direction and TransformerLens: read it, attribute it, patch along it.

    direction = unit_direction(np.load("results/sandbagging_direction_l14.npz"))
    project(cache, direction)                        # [layer, batch, pos]: how much of the direction is in the stream
    attribute(cache, direction, pos=-1)              # which heads and MLPs wrote it at one position
    get_act_patch_resid_direction(model, corrupted, clean_cache, metric, unit=direction.unit, mode="swap")

Nothing here re-implements TransformerLens. Hooks go through `bridge.add_hook`; attribution through
`ActivationCache.stack_head_results` and `decompose_resid` (direct logit attribution with a unit vector in place of an
unembedding column); the patching sweep through `transformer_lens.patching.generic_activation_patch`, unchanged, with
one new patch setter that touches only the component along the direction:

    zero   h  ->  h - (h·u) u                      remove the direction from the corrupted run
    swap   h  ->  h - (h·u) u + (h_clean·u) u      give the corrupted run the clean run's amount of the direction
    add    h  ->  h + (h_clean·u) u                add the clean run's amount on top

Layer convention: `layer` is the block whose OUTPUT the direction was built on, so it lives at
`blocks.{layer}.hook_resid_post` (the same tensor as `blocks.{layer + 1}.hook_resid_pre`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import torch

MODES = ("zero", "swap", "add")


@dataclass(frozen=True)
class Direction:
    """A unit vector in the residual stream at one layer."""

    unit: torch.Tensor
    layer: int
    name: str = "direction"

    @property
    def d_model(self) -> int:
        return int(self.unit.shape[0])

    @property
    def hook_name(self) -> str:
        return f"blocks.{self.layer}.hook_resid_post"


def unit_direction(npz: Any, name: str = "sandbagging direction") -> Direction:
    """A Direction from a loaded .npz with `vector` and `layer` (the vector is normalised to length 1)."""
    vector = torch.as_tensor(npz["vector"], dtype=torch.float32)
    return Direction(unit=vector / vector.norm(), layer=int(npz["layer"]), name=name)


# --- Reading and steering -------------------------------------------------------------------------------------------
def project(cache: Any, direction: Direction) -> torch.Tensor:
    """The direction's reading at every layer's `hook_resid_post`: shape [n_layers, batch, pos]."""
    stack = cache.stack_activation("resid_post")  # [n_layers, batch, pos, d_model]
    return stack @ direction.unit.to(device=stack.device, dtype=stack.dtype)


@dataclass
class DirectionHook:
    """What `add_direction_hook` returns: how often the hook fired, and `remove()`."""

    name: str
    calls: int = 0
    seen: int = 0
    _bridge: Any = field(default=None, repr=False)
    _fn: Any = field(default=None, repr=False)

    def remove(self) -> None:
        from transformer_lens.model_bridge.bridge_core import build_alias_to_canonical_map

        hook_dict = self._bridge.hook_dict
        point = hook_dict[build_alias_to_canonical_map(hook_dict).get(self.name, self.name)]
        for handle in list(point.fwd_hooks):
            if handle.user_hook is self._fn:
                handle.hook.remove()
                point.fwd_hooks.remove(handle)


def add_direction_hook(bridge: Any, direction: Direction, strength: float, prompt_len: int = 0) -> DirectionHook:
    """Add `strength × unit` at the direction's hook point on positions >= prompt_len (absolute positions, so the same
    hook steers a full pass and a KV-cached decode loop). Strength 0 adds an exact zero."""
    state = DirectionHook(name=direction.hook_name)
    delta = float(strength) * direction.unit

    def hook(activation: torch.Tensor, hook: Any) -> torch.Tensor:
        state.calls += 1
        seq = int(activation.shape[1])
        start = 0 if seq > 1 else state.seen
        state.seen = start + seq
        first = max(prompt_len - start, 0)
        if first >= seq:
            return activation
        out = activation.clone()
        out[:, first:] += delta.to(device=activation.device, dtype=activation.dtype)
        return out

    state._bridge, state._fn = bridge, hook
    bridge.add_hook(state.name, hook)
    return state


# --- Attribution ----------------------------------------------------------------------------------------------------
@dataclass
class DirectionAttribution:
    """Each component's contribution to the direction's reading at one layer and position. `values[i, b]` is component
    `labels[i]` for batch element `b`; `total[b]` is the whole residual's reading, so `total - values.sum(0)` is ~0."""

    labels: list[str]
    values: torch.Tensor
    total: torch.Tensor


def attribute(cache: Any, direction: Direction, pos: int = -1) -> DirectionAttribution:
    """Split the reading at `blocks.{layer}.hook_resid_post`, position `pos`, into: `embed` (and `pos_embed` when the
    model has one), every head `L{l}H{h}` with l <= layer, each layer's attention bias `{l}_attn_bias` (the part of
    attn_out no head owns), and each `{l}_mlp_out`."""
    layer = direction.layer
    upto = layer + 1
    n_heads = int(cache.model.cfg.n_heads)
    resid_full = cache[("resid_post", layer)]
    pos = pos % int(resid_full.shape[1])
    heads, head_labels = cache.stack_head_results(layer=upto, pos_slice=pos, return_labels=True)
    parts, part_labels = cache.decompose_resid(layer=upto, pos_slice=pos, return_labels=True)
    components, labels = [], []
    for label, part in zip(part_labels, parts, strict=True):
        if label.endswith("_attn_out"):
            index = int(label.split("_")[0])
            these = heads[index * n_heads : (index + 1) * n_heads]
            components.extend(these)
            labels.extend(head_labels[index * n_heads : (index + 1) * n_heads])
            components.append(part - these.sum(dim=0))
            labels.append(f"{index}_attn_bias")
        else:
            components.append(part)
            labels.append(label)
    stack = torch.stack(components)
    unit = direction.unit.to(device=stack.device, dtype=stack.dtype)
    return DirectionAttribution(
        labels=labels, values=(stack @ unit).detach(), total=(resid_full[:, pos] @ unit).detach()
    )


# --- Patching along a direction -------------------------------------------------------------------------------------
def layer_pos_direction_patch_setter(
    corrupted_activation: torch.Tensor,
    index: Sequence[int],
    clean_activation: torch.Tensor,
    *,
    unit: torch.Tensor,
    mode: str = "swap",
) -> torch.Tensor:
    """Patch setter for index = [layer, pos] that replaces only the component along `unit` (activation axes
    [batch, pos, d_model]); same positional signature as TransformerLens's `layer_pos_patch_setter`."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    _layer, pos = index
    u = unit.to(device=corrupted_activation.device, dtype=corrupted_activation.dtype)
    h = corrupted_activation[:, pos, ...]
    clean_along = (clean_activation[:, pos, ...] @ u)[..., None] * u
    if mode == "zero":
        patched = h - (h @ u)[..., None] * u
    elif mode == "swap":
        patched = h - (h @ u)[..., None] * u + clean_along
    else:
        patched = h + clean_along
    corrupted_activation[:, pos, ...] = patched
    return corrupted_activation


def get_act_patch_resid_direction(
    model: Any,
    corrupted_tokens: torch.Tensor,
    clean_cache: Any,
    patching_metric: Any,
    *,
    unit: torch.Tensor,
    mode: str = "swap",
    **kwargs: Any,
) -> Any:
    """`get_act_patch_resid_pre`, restricted to one direction: a [n_layers, pos] grid of the metric (or a
    PatchingResult with axes ["layer", "pos"] when `baseline=True` or `reduce="none"` is passed through). Pass
    `index_df` to sweep only chosen cells; the result is then flat, one value per row of `index_df`."""
    from transformer_lens.patching import generic_activation_patch

    if "index_df" not in kwargs:
        kwargs["index_axis_names"] = ("layer", "pos")
    return generic_activation_patch(
        model,
        corrupted_tokens,
        clean_cache,
        patching_metric,
        patch_setter=partial(layer_pos_direction_patch_setter, unit=unit, mode=mode),
        activation_name="resid_pre",
        **kwargs,
    )
