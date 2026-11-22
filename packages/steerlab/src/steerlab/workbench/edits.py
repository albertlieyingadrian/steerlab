"""Concept edits in chat: apply edit artifacts to the model's weights for one reply, then restore them exactly.

The implementation is `steerlab.edits` (the Edit room's): this module only turns artifacts into edit-list records and
reports errors as `EditError`. An edit names per-layer axes and some units (neurons, attention heads) and moves their
weights through H = I − k d dᵀ:

- reflect (k = 2): the unit hears and writes the mirror image along d;
- ablate (k = 1): the unit no longer reads or writes anything along d;
- amplify (k < 0): the d-component is stretched by 1 − k.

    with applied(model, [meta], [arrays], tokenizer=tok) as labels:
        ...generate...                  # the model with the edits
    # every edited tensor is back to its saved copy, bit for bit

The artifact shape is documented in `steerlab.edits.stack` (`load_artifact` reads it): `units` as
[{kind, layer, index}], `mode` or `k`, and the axes as npz `reads`/`writes` (or rebuilt from `axis_source` with the
tokenizer). The first Workbench shape, one `axis` (inline list or npz "axis") acting on the units' writes only, still
works. torch is imported inside the functions.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from typing import Any

from steerlab.edits.stack import MODE_K, EditStack, load_artifact, mode_for_k

MAX_UNITS = 200_000
__all__ = ["MAX_UNITS", "MODE_K", "EditError", "applied", "edit_k"]


class EditError(ValueError):
    """An edit artifact that cannot be applied to this model."""


def edit_k(edit: Mapping[str, Any]) -> float:
    if edit.get("k") is not None:
        return float(edit["k"])
    mode = str(edit.get("mode", "reflect"))
    if mode not in MODE_K:
        raise EditError(f"edit mode must be one of {sorted(MODE_K)}, not {mode!r}")
    return MODE_K[mode]


@contextmanager
def applied(
    model: Any,
    edits: Sequence[Mapping[str, Any]],
    arrays: Sequence[Mapping[str, Any] | None] | None = None,
    *,
    tokenizer: Any = None,
) -> Iterator[list[str]]:
    """Apply `edits` (artifact dicts) in order for the duration of the block; restore every touched weight after.

    Yields one label per edit. A malformed edit raises EditError before the block runs, with nothing left applied.
    """
    if not edits:
        yield []
        return
    from steerlab.edits.reflection import Arch

    try:
        d_model = Arch(model).d_model
        stack = EditStack()
        labels = []
        for i, edit in enumerate(edits):
            record, axes = load_artifact(dict(edit), dict(arrays[i] or {}) if arrays else None, d_model)
            count = record.units.count()
            n = count["neurons"] + count["heads"]
            if n > MAX_UNITS:
                raise EditError(f"an edit can touch at most {MAX_UNITS} units")
            if record.kind == "reflect" and axes is None and tokenizer is None:
                raise EditError(
                    "this edit carries no axes (npz reads/writes) and no tokenizer was given to rebuild them"
                )
            stack.add(record, axes)
            name = edit.get("name", "edit")
            if record.kind == "reflect":
                k = float(record.k or 0.0)
                labels.append(f"{name}: {edit.get('mode') or mode_for_k(k)} k={k:g} on {n} units")
            else:
                labels.append(f"{name}: {record.label}")
    except ValueError as error:
        raise EditError(str(error)) from None

    with ExitStack() as scope:
        try:
            scope.enter_context(stack.applied(model, tokenizer))
        except ValueError as error:
            raise EditError(str(error)) from None
        yield labels
