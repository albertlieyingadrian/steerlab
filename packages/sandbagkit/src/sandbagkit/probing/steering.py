"""Steering: a direction in activation space, and a context manager that pushes a layer along it.

    vec = steering_vector(X, labels, groups=pair_ids, train_groups=train_pairs, test_groups=test_pairs, layer=14)
    vec.vector            mean(label-1 rows) − mean(label-0 rows), over TRAINING groups only
    with inject(model, vec, strength=-2.0):
        model.generate(...)       layer 14 outputs  original + (−2) × vector  at every position
    # leaving the block removes the hook, even after an error

The train-only guard: the toolkit built its steering vector from all 32 pairs, including the 7 it then
tested the probe on. Any number measured on those test pairs afterwards has seen its own answer. Here,
`steering_vector` refuses (LeakageError) if a test group would contribute a single row.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import numpy.typing as npt

from sandbagkit.probing.hooks import layer_output_tensor, register_hooks, replace_layer_output, resolve_layers
from sandbagkit.probing.io import load_arrays, save_arrays

if TYPE_CHECKING:
    import torch

FloatArray = npt.NDArray[np.float64]


class LeakageError(ValueError):
    """Test data would have been used to build something that is later evaluated on it."""


def mean_difference(X: npt.ArrayLike, labels: npt.ArrayLike) -> FloatArray:
    """mean(rows with label 1) − mean(rows with label 0). No guard: see `steering_vector` for that."""
    matrix = np.asarray(X, dtype=np.float64)
    y = np.asarray(labels)
    if matrix.ndim != 2 or y.shape != (matrix.shape[0],):
        raise ValueError(f"X must be [n, d] and labels [n]; got {matrix.shape} and {y.shape}")
    positive, negative = matrix[y == 1], matrix[y == 0]
    if len(positive) == 0 or len(negative) == 0:
        raise ValueError("need at least one row of each label")
    difference: FloatArray = positive.mean(axis=0) - negative.mean(axis=0)
    return difference


@dataclass(eq=False)
class SteeringVector:
    """A direction in one layer's activation space, with a record of exactly what it was built from."""

    vector: FloatArray
    layer: int | None = None
    n_positive: int = 0
    n_negative: int = 0
    train_groups: tuple[int, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    KIND: ClassVar[str] = "steering_vector"

    @property
    def norm(self) -> float:
        return float(np.linalg.norm(self.vector))

    def unit(self) -> FloatArray:
        """The vector scaled to length 1 (so `strength` is in activation units, not "multiples of the gap")."""
        unit: FloatArray = self.vector / self.norm
        return unit

    def save(self, path: str | Path, *, metadata: Mapping[str, Any] | None = None) -> tuple[Path, Path]:
        arrays: dict[str, npt.NDArray[Any]] = {"vector": self.vector}
        if self.train_groups is not None:
            arrays["train_groups"] = np.asarray(self.train_groups, dtype=np.int64)
        meta = {
            **self.metadata,
            **dict(metadata or {}),
            "layer": self.layer,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "has_train_groups": self.train_groups is not None,
        }
        return save_arrays(path, arrays, kind=self.KIND, metadata=meta)

    @classmethod
    def load(cls, path: str | Path) -> SteeringVector:
        arrays, meta = load_arrays(path, kind=cls.KIND)
        layer, n_positive, n_negative = meta.pop("layer"), meta.pop("n_positive"), meta.pop("n_negative")
        has_groups = meta.pop("has_train_groups", False)
        groups = tuple(int(g) for g in arrays["train_groups"]) if has_groups else None
        return cls(arrays["vector"].astype(np.float64), layer, n_positive, n_negative, groups, meta)


def steering_vector(
    X: npt.ArrayLike,
    labels: npt.ArrayLike,
    *,
    groups: npt.ArrayLike | None = None,
    train_groups: npt.ArrayLike | None = None,
    test_groups: npt.ArrayLike | None = None,
    layer: int | None = None,
) -> SteeringVector:
    """The mean-difference direction, built from training groups only.

    - `groups`: one group id per row (the pair each prompt came from).
    - `train_groups`: the groups allowed in. Defaults to every group not in `test_groups`.
    - `test_groups`: groups that must NOT contribute. Overlap with `train_groups` raises LeakageError.

    Without `groups` every row is used, and the result records `train_groups=None` ("unguarded").
    """
    matrix = np.asarray(X, dtype=np.float64)
    y = np.asarray(labels)
    if groups is None:
        if train_groups is not None or test_groups is not None:
            raise ValueError("train_groups/test_groups need groups=[one id per row]")
        return SteeringVector(mean_difference(matrix, y), layer, int((y == 1).sum()), int((y == 0).sum()), None)

    group_ids = np.asarray(groups)
    if group_ids.shape[0] != matrix.shape[0]:
        raise ValueError(f"{matrix.shape[0]} rows but {group_ids.shape[0]} group ids")
    test = np.asarray([] if test_groups is None else test_groups, dtype=group_ids.dtype)
    train = np.setdiff1d(np.unique(group_ids), test) if train_groups is None else np.asarray(train_groups)
    overlap = np.intersect1d(train, test)
    if overlap.size:
        raise LeakageError(f"groups {overlap.tolist()[:5]} are in both train_groups and test_groups")
    use = np.isin(group_ids, train)
    if np.isin(group_ids[use], test).any():  # the guard itself, kept even though the overlap check implies it
        raise LeakageError("a test group's row was about to be used to build the steering vector")
    vector = mean_difference(matrix[use], y[use])
    return SteeringVector(
        vector, layer, int((y[use] == 1).sum()), int((y[use] == 0).sum()), tuple(int(g) for g in np.unique(train))
    )


@dataclass
class Injection:
    """What `inject` yields: the settings, and how many forward passes were steered so far."""

    layer: int
    strength: float
    calls: int = 0


def _as_tensor(vector: SteeringVector | npt.ArrayLike | torch.Tensor) -> torch.Tensor:
    import torch

    if isinstance(vector, SteeringVector):
        vector = vector.vector
    if isinstance(vector, torch.Tensor):
        return vector.detach()
    return torch.as_tensor(np.asarray(vector, dtype=np.float32))


@contextmanager
def inject(
    model: Any,
    vector: SteeringVector | npt.ArrayLike | torch.Tensor,
    *,
    layer: int | None = None,
    strength: float = 1.0,
    layers_path: str | None = None,
) -> Iterator[Injection]:
    """Inside the block, decoder layer `layer` outputs `original + strength × vector` at every token position.

    `layer` defaults to the SteeringVector's own layer. The vector is moved to the model's device and dtype
    once, on entry. Strength 0 leaves every output bit-for-bit unchanged (x + 0 = x), which is the sanity
    check the tests run. The hook is removed on exit, even if the block raised.
    """
    if layer is None and isinstance(vector, SteeringVector):
        layer = vector.layer
    if layer is None:
        raise ValueError("which layer? pass layer=..., or a SteeringVector that records one")
    resolved = resolve_layers(model, layer, layers_path)[0]
    parameter = next(model.parameters())
    delta = _as_tensor(vector).to(device=parameter.device, dtype=parameter.dtype)
    hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)  # None for a non-HF module
    if delta.ndim != 1 or (hidden_size is not None and delta.shape[0] != hidden_size):
        raise ValueError(f"the vector has shape {tuple(delta.shape)}, but the model's hidden size is {hidden_size}")
    delta = strength * delta  # the same arithmetic as the lessons' steer(): original + strength * vector
    state = Injection(layer=resolved, strength=float(strength))

    def hook(module: Any, inputs: Any, output: Any) -> Any:
        state.calls += 1
        return replace_layer_output(output, layer_output_tensor(output) + delta)

    with register_hooks(model, {resolved: hook}, layers_path=layers_path):
        yield state
