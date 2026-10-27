"""Named difference-of-means directions per model, with the quality numbers that say if they should steer.

    model, tokenizer, name, layer = load_model("qwen")                 Qwen3-1.7B, thinking off
    vec = build_direction(model, tokenizer, load_concept("chinese"), layer=layer, model_name=name)
    vec.save(direction_path(out_dir, "qwen", "chinese"))               <out>/qwen/chinese.npz + .json
    to_feature_direction(vec)                                           unit vector + scale for the steering hook

How a direction is built: every contrast pair is two (prompt, reply) examples. The reply's tokens at one layer are
averaged (sandbagkit's `response_mean`, first reply token excluded), the positive mean minus the negative mean is the
direction, and it is stored as a UNIT vector with the class gap kept beside it. Strength 1.0 then means "add one class
gap along the direction", the same unit the SAE features use after calibration.

Quality numbers (Braun et al., "Understanding refusal/steering"):
- class gap: the two class means projected on the unit direction, positive minus negative; the norm of the raw mean
  difference.
- cosine agreement: the mean cosine between each pair's own difference and the mean difference. Near 1, every pair
  points the same way and steering is expected to work; near 0, the mean is an average of disagreeing arrows.
- d′: the class gap over the pooled standard deviation of the projections. Large means the classes separate cleanly
  along the direction. None when there is no spread at all (only possible on synthetic data).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt

from steerlab import defaults, hub
from steerlab.data import CONCEPTS, Concept, list_concepts, load_concept

if TYPE_CHECKING:
    from sandbagkit.probing import SteeringVector

    from steerlab.saes.common import FeatureDirection

DIRECTION_RELEASE = "sandbagkit.difference_of_means"
"""What `FeatureDirection.release` says for a direction (an SAE feature says its Hub release instead)."""

METADATA_KEYS: tuple[str, ...] = ("model", "layer", "concept", "class_gap", "cosine_agreement", "d_prime")
"""The keys every saved direction's JSON metadata carries."""

DEFAULT_OUTPUT_DIR = defaults.directions_dir()
"""Where `directions build` writes by default (`<model>/<concept>`): $STEERLAB_DIRECTIONS, else `outputs/directions`."""

TINY_MODEL_KEY = "tiny"
"""sandbagkit's 2-layer random model, for offline smoke runs: `steerlab directions build --model tiny`."""

MODEL_KEYS: tuple[str, ...] = (*defaults.MODELS, *defaults.EXTRA_MODELS, TINY_MODEL_KEY)

DEFAULT_LAYERS: dict[str, int] = {
    "qwen": defaults.QWEN_SCOPE_LAYER,
    "gemma": defaults.GEMMA_SCOPE_LAYER,
    **defaults.EXTRA_LAYERS,
    TINY_MODEL_KEY: 1,
}

POSITION: Literal["response_mean"] = "response_mean"
SKIP_FIRST_RESPONSE_TOKEN = True


def template_kwargs_for(model_name: str) -> dict[str, object]:
    """Extra chat-template variables per model. The rule lives in `defaults.template_kwargs_for`."""
    return defaults.template_kwargs_for(model_name)


def check_model_key(model_key: str) -> str:
    if model_key not in MODEL_KEYS:
        raise KeyError(f"unknown model {model_key!r}; choose one of {', '.join(MODEL_KEYS)}")
    return model_key


def default_layer(model_key: str) -> int:
    return DEFAULT_LAYERS[check_model_key(model_key)]


def direction_path(root: str | Path, model_key: str, concept: str) -> Path:
    """`<root>/<model>/<concept>` without a suffix; sandbagkit adds `.npz` and `.json`."""
    return Path(root) / model_key / concept


def build_command(model_key: str, concept: str, root: str | Path) -> str:
    """The command that creates one missing direction, as `directions list` prints it."""
    return f"steerlab directions build --model {model_key} --concept {concept} --out {root}"


@dataclass(frozen=True)
class DirectionQuality:
    class_gap: float
    cosine_agreement: float
    d_prime: float | None
    n_pairs: int


def direction_quality(
    X: npt.ArrayLike, labels: npt.ArrayLike, groups: npt.ArrayLike, unit: npt.ArrayLike
) -> DirectionQuality:
    """The three quality numbers of `unit` on rows `X` (labels 1/0, one group id per row = the pair)."""
    matrix = np.asarray(X, dtype=np.float64)
    y = np.asarray(labels)
    g = np.asarray(groups)
    u = np.asarray(unit, dtype=np.float64)
    u = u / np.linalg.norm(u)
    positive, negative = matrix[y == 1], matrix[y == 0]

    projections = matrix @ u
    class_gap = float(projections[y == 1].mean() - projections[y == 0].mean())

    diffs = []
    for group in np.unique(g):
        rows_pos = matrix[(g == group) & (y == 1)]
        rows_neg = matrix[(g == group) & (y == 0)]
        if len(rows_pos) and len(rows_neg):
            diffs.append(rows_pos.mean(axis=0) - rows_neg.mean(axis=0))
    pair_diffs = np.asarray(diffs)
    norms = np.linalg.norm(pair_diffs, axis=1)
    cosines = (pair_diffs @ u) / np.where(norms == 0, 1.0, norms)
    cosine_agreement = float(cosines.mean()) if len(cosines) else 0.0

    d_prime: float | None = None
    if len(positive) > 1 and len(negative) > 1:
        pooled = np.sqrt((projections[y == 1].var(ddof=1) + projections[y == 0].var(ddof=1)) / 2)
        if pooled > 0:
            d_prime = float(class_gap / pooled)
    return DirectionQuality(class_gap, cosine_agreement, d_prime, len(pair_diffs))


def load_model(model_key: str) -> tuple[Any, Any, str, int]:
    """(model, tokenizer, model name, default layer) for a model key: `qwen`, `gemma`, an extra model or `tiny`.

    Real models load through sandbagkit's `HFBackend.from_pretrained` (device and dtype policy). A gated model that
    is not yet in the Hub cache is only fetched after `hub.ensure_access` says this account may; the error
    it raises names the licence page, never the token.
    """
    check_model_key(model_key)
    if model_key == TINY_MODEL_KEY:
        from sandbagkit.backends.tiny import tiny_backend

        backend = tiny_backend(seed=0)
        return backend.model, backend.tokenizer, backend.name, default_layer(model_key)
    from sandbagkit.backends.hf import HFBackend

    name = defaults.MODELS.get(model_key) or defaults.EXTRA_MODELS[model_key]
    if name in defaults.GATED_REPOS and not hub.cached_files(name):
        hub.ensure_access(name)
    backend = HFBackend.from_pretrained(name)
    return backend.model, backend.tokenizer, name, default_layer(model_key)


def build_direction(
    model: Any,
    tokenizer: Any,
    concept: Concept,
    *,
    layer: int,
    model_name: str,
    template_kwargs: Mapping[str, object] | None = None,
    batch_size: int = 8,
    max_length: int = 512,
    model_key: str | None = None,
) -> SteeringVector:
    """The unit difference-of-means direction for `concept` at `layer`, with its quality numbers in `metadata`.

    `template_kwargs` defaults to `template_kwargs_for(model_name)`. The pair ids go to `steering_vector` as
    `groups`, so the result records which pairs built it and the leakage guard applies when some are held out later.
    """
    import dataclasses

    from sandbagkit.probing import extract_activations, steering_vector

    kwargs = dict(template_kwargs_for(model_name) if template_kwargs is None else template_kwargs)
    texts, responses, labels, groups = concept.as_rows()
    acts = extract_activations(
        model,
        tokenizer,
        texts,
        [layer],
        responses=responses,
        position=POSITION,
        chat_template=True,
        template_kwargs=kwargs,
        skip_first_response_token=SKIP_FIRST_RESPONSE_TOKEN,
        batch_size=batch_size,
        max_length=max_length,
    )[layer]
    raw = steering_vector(acts, labels, groups=groups, layer=layer)
    unit = raw.unit()
    quality = direction_quality(acts, labels, groups, unit)
    metadata: dict[str, Any] = {
        "model": model_name,
        "model_key": model_key,
        "layer": layer,
        "concept": concept.name,
        "label": concept.label,
        "class_gap": quality.class_gap,
        "cosine_agreement": quality.cosine_agreement,
        "d_prime": quality.d_prime,
        "n_pairs": quality.n_pairs,
        "position": POSITION,
        "skip_first_response_token": SKIP_FIRST_RESPONSE_TOKEN,
        "template_kwargs": kwargs,
        "release": DIRECTION_RELEASE,
        "unit_norm": True,
    }
    return dataclasses.replace(raw, vector=unit, metadata=metadata)


@dataclass(frozen=True)
class BuiltDirection:
    concept: str
    path: Path
    vector: SteeringVector


def check_concepts(names: Iterable[str]) -> list[str]:
    known = list_concepts()
    chosen = list(names)
    unknown = [name for name in chosen if name not in known]
    if unknown:
        raise KeyError(f"unknown concept(s) {', '.join(unknown)}; the shipped concepts are {', '.join(known)}")
    return chosen


def build_directions(
    model_key: str,
    concepts: Sequence[str] = CONCEPTS,
    *,
    out_dir: str | Path = DEFAULT_OUTPUT_DIR,
    layer: int | None = None,
    batch_size: int = 8,
    max_length: int = 512,
    loaded: tuple[Any, Any, str, int] | None = None,
) -> list[BuiltDirection]:
    """Build and save one direction per concept for `model_key`. `loaded` reuses an already-loaded model."""
    names = check_concepts(concepts)
    model, tokenizer, model_name, model_layer = loaded or load_model(model_key)
    chosen_layer = model_layer if layer is None else layer
    built = []
    for name in names:
        vec = build_direction(
            model,
            tokenizer,
            load_concept(name),
            layer=chosen_layer,
            model_name=model_name,
            batch_size=batch_size,
            max_length=max_length,
            model_key=model_key,
        )
        path = direction_path(out_dir, model_key, name)
        vec.save(path)
        built.append(BuiltDirection(name, path, vec))
    return built


def load_direction(path: str | Path) -> SteeringVector:
    """A saved direction (`<path>.npz` + `.json`), vector and metadata as `build_direction` stored them."""
    from sandbagkit.probing import SteeringVector

    return SteeringVector.load(path)


def to_feature_direction(vec: SteeringVector, *, default_label: str | None = None) -> FeatureDirection:
    """The direction in the shape the steering hook consumes: a float32 unit vector plus `scale` = the class gap.

    `feature` is 0: a direction has no feature index; `release` and `label` say what it is instead. The label is the
    saved concept or label, else `default_label` (the CLI passes the `--direction NAME`), else "direction".
    """
    import torch

    from steerlab.saes.common import FeatureDirection

    if vec.layer is None:
        raise ValueError("this SteeringVector records no layer")
    unit = torch.as_tensor(np.asarray(vec.vector, dtype=np.float32))
    unit = unit / unit.norm()
    return FeatureDirection(
        unit=unit,
        layer=int(vec.layer),
        release=DIRECTION_RELEASE,
        feature=0,
        label=str(vec.metadata.get("concept") or vec.metadata.get("label") or default_label or "direction"),
        scale=float(vec.metadata["class_gap"]) if "class_gap" in vec.metadata else None,
    )


@dataclass(frozen=True)
class DirectionStatus:
    concept: str
    path: Path
    present: bool
    metadata: dict[str, Any]


def direction_status(
    model_key: str, concepts: Sequence[str] = CONCEPTS, *, out_dir: str | Path = DEFAULT_OUTPUT_DIR
) -> list[DirectionStatus]:
    """Which of `concepts` exist under `<out_dir>/<model>/`, with their saved metadata. Reads only the JSON files."""
    import json

    statuses = []
    for name in check_concepts(concepts):
        base = direction_path(out_dir, model_key, name)
        npz, meta_file = base.with_suffix(".npz"), base.with_suffix(".json")
        present = npz.is_file() and meta_file.is_file()
        metadata: dict[str, Any] = {}
        if present:
            description = json.loads(meta_file.read_text(encoding="utf-8"))
            metadata = dict(description.get("metadata", {}))
        statuses.append(DirectionStatus(name, base, present, metadata))
    return statuses


def format_quality(metadata: Mapping[str, Any]) -> str:
    """`class gap 12.3, cosine agreement 0.81, d' 3.2` from a direction's metadata."""
    d_prime = metadata.get("d_prime")
    d_text = "n/a" if d_prime is None else f"{float(d_prime):.2f}"
    return (
        f"class gap {float(metadata['class_gap']):.3g}, "
        f"cosine agreement {float(metadata['cosine_agreement']):.2f}, d' {d_text}"
    )
