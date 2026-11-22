"""Concept probes: a direction per concept, read on every token, always next to a random-direction baseline.

A probe here is a ruler laid across the residual stream. Take example replies from two poles of a concept (formal vs
casual), run them through the model, average each reply's activations at one layer, and the ruler is the line from
the casual average to the formal average. Reading a token = where its activation falls along that ruler, in units
of the examples' own spread (a z-score: 0 is halfway between the poles, about +d'/2 is a typical formal example).

Every number needs a control: a random direction read the same way. Sixteen random unit directions, z-scored on the
same examples, give the "how big is a reading by chance" band; the UI draws it in grey under every probe.

    bundle = load_bundle()                                      # the 16 bundled concepts (concepts.json)
    X = response_means(model, tokenizer, examples, layers=[14]) # {14: [n_examples, d_model]}
    probe = fit_probe("formal.casual", X_pos, X_neg, layer=14, model_id="Qwen/...")
    probe.read(h)                                               # z-scores, one per row of h
    probe.random_band(h)                                        # the chance-sized reading for each row

Fitting is the difference of means (no optimiser), and the random directions come from a seed derived from the
concept, model and layer, so the same examples always give the same probe, bit for bit.

numpy at import; torch only inside `response_means`.
"""

from __future__ import annotations

import json
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from importlib import resources
from typing import Any

import numpy as np
import numpy.typing as npt

from steerlab.workbench.artifacts import ArtifactStore, slug

FloatArray = npt.NDArray[np.float32]
N_RANDOM = 16
MAX_EXAMPLES = 64
MAX_EXAMPLE_CHARS = 2000


@dataclass(frozen=True)
class Example:
    """One assistant reply (`response`) to a user `prompt`; its reply tokens are what the probe averages."""

    prompt: str
    response: str


@dataclass(frozen=True)
class ConceptSpec:
    name: str
    category: str
    positive_label: str
    negative_label: str
    positive: tuple[Example, ...]
    negative: tuple[Example, ...]
    source: str = "bundle"


def _bundle_data() -> dict[str, Any]:
    text = resources.files("steerlab.workbench").joinpath("concepts.json").read_text()
    data: dict[str, Any] = json.loads(text)
    return data


def load_bundle() -> list[ConceptSpec]:
    """The bundled concepts. `source: steerlab.data:<name>` entries reuse that concept's contrast pairs."""
    out = []
    for raw in _bundle_data()["concepts"]:
        source = raw.get("source", "bundle")
        if source.startswith("steerlab.data:"):
            from steerlab.data import load_concept

            concept = load_concept(source.split(":", 1)[1])
            positive = tuple(Example(p.positive.prompt, p.positive.response) for p in concept.pairs)
            negative = tuple(Example(p.negative.prompt, p.negative.response) for p in concept.pairs)
        else:
            prompt = raw.get("prompt", "Say something.")
            positive = tuple(Example(prompt, text) for text in raw["positive"])
            negative = tuple(Example(prompt, text) for text in raw["negative"])
        out.append(
            ConceptSpec(
                name=raw["name"],
                category=raw.get("category", "other"),
                positive_label=raw.get("positive_label", "positive"),
                negative_label=raw.get("negative_label", "negative"),
                positive=positive,
                negative=negative,
                source=source,
            )
        )
    return out


def spec_from_examples(
    name: str,
    positive: Sequence[str],
    negative: Sequence[str],
    *,
    prompt: str = "Say something.",
    positive_label: str = "positive",
    negative_label: str = "negative",
) -> ConceptSpec:
    """A concept from user-typed examples (the Probes room's "fit a new probe")."""
    pos = [text.strip() for text in positive if text.strip()]
    neg = [text.strip() for text in negative if text.strip()]
    if len(pos) < 2 or len(neg) < 2:
        raise ValueError("give at least two examples for each pole")
    if len(pos) > MAX_EXAMPLES or len(neg) > MAX_EXAMPLES:
        raise ValueError(f"at most {MAX_EXAMPLES} examples per pole")
    if any(len(text) > MAX_EXAMPLE_CHARS for text in [*pos, *neg]):
        raise ValueError(f"an example is longer than {MAX_EXAMPLE_CHARS} characters")
    return ConceptSpec(
        name=slug(name),
        category="custom",
        positive_label=positive_label or "positive",
        negative_label=negative_label or "negative",
        positive=tuple(Example(prompt, text) for text in pos),
        negative=tuple(Example(prompt, text) for text in neg),
        source="user",
    )


def seed_for(name: str, model_id: str, layer: int) -> int:
    return zlib.crc32(f"{name}|{model_id}|{layer}".encode())


def random_directions(d_model: int, seed: int, k: int = N_RANDOM) -> FloatArray:
    """k random unit directions in the residual stream, the same ones for the same seed."""
    rng = np.random.default_rng(seed)
    dirs = rng.standard_normal((k, d_model)).astype(np.float32)
    out: FloatArray = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    return out


def _separation(pos: FloatArray, neg: FloatArray) -> tuple[float, float, float]:
    """(midpoint, pooled sd, d') of two sets of 1-D projections."""
    mid = float((pos.mean() + neg.mean()) / 2)
    pooled = float(np.sqrt((pos.var(ddof=1) + neg.var(ddof=1)) / 2)) if len(pos) > 1 and len(neg) > 1 else 0.0
    sd = pooled if pooled > 1e-8 else 1.0
    return mid, sd, float((pos.mean() - neg.mean()) / sd)


@dataclass
class Probe:
    """A fitted concept probe at one layer, with its random-direction baseline."""

    name: str
    model: str
    layer: int
    direction: FloatArray  # unit vector, [d_model]
    mid: float
    sd: float
    class_gap: float
    d_prime: float
    accuracy: float
    random_dirs: FloatArray  # [k, d_model]
    random_mid: FloatArray  # [k]
    random_sd: FloatArray  # [k]
    random_d_prime: FloatArray  # [k]
    positive_label: str = "positive"
    negative_label: str = "negative"
    category: str = "other"
    source: str = "bundle"
    n_examples: tuple[int, int] = (0, 0)
    d_prime_by_layer: dict[int, float] = field(default_factory=dict)

    def read(self, h: npt.ArrayLike) -> FloatArray:
        """z-scores of activations `h` ([rows, d_model] or [d_model]) along the probe."""
        x = np.asarray(h, dtype=np.float32)
        out: FloatArray = ((x @ self.direction) - self.mid) / self.sd
        return out

    def read_random(self, h: npt.ArrayLike) -> FloatArray:
        """The same reading along each random direction: [rows, k] (or [k])."""
        x = np.asarray(h, dtype=np.float32)
        out: FloatArray = ((x @ self.random_dirs.T) - self.random_mid) / self.random_sd
        return out

    def random_band(self, h: npt.ArrayLike) -> FloatArray:
        """The 90th percentile of |random reading| per row: a reading inside this band is chance-sized."""
        out: FloatArray = np.quantile(np.abs(self.read_random(h)), 0.9, axis=-1).astype(np.float32)
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "layer": self.layer,
            "category": self.category,
            "source": self.source,
            "positive_label": self.positive_label,
            "negative_label": self.negative_label,
            "d_prime": round(self.d_prime, 4),
            "random_d_prime": round(float(np.quantile(np.abs(self.random_d_prime), 0.9)), 4),
            "accuracy": round(self.accuracy, 4),
            "class_gap": round(self.class_gap, 4),
            "n_examples": list(self.n_examples),
            "d_prime_by_layer": {str(k): round(v, 4) for k, v in self.d_prime_by_layer.items()},
        }

    # As an artifact

    def to_artifact(self) -> tuple[dict[str, Any], dict[str, npt.NDArray[Any]]]:
        meta = {"type": "probe", **self.summary(), "mid": self.mid, "sd": self.sd, "scale": self.class_gap}
        arrays = {
            "direction": self.direction,
            "random_dirs": self.random_dirs,
            "random_mid": self.random_mid,
            "random_sd": self.random_sd,
            "random_d_prime": self.random_d_prime,
        }
        return meta, arrays

    @classmethod
    def from_artifact(cls, meta: dict[str, Any], arrays: dict[str, npt.NDArray[Any]]) -> Probe:
        counts = meta.get("n_examples", (0, 0))
        return cls(
            name=str(meta.get("concept", meta["name"])),
            model=str(meta.get("model", "")),
            layer=int(meta["layer"]),
            direction=arrays["direction"].astype(np.float32),
            mid=float(meta["mid"]),
            sd=float(meta["sd"]),
            class_gap=float(meta.get("class_gap", meta.get("scale", 1.0))),
            d_prime=float(meta.get("d_prime", 0.0)),
            accuracy=float(meta.get("accuracy", 0.0)),
            random_dirs=arrays["random_dirs"].astype(np.float32),
            random_mid=arrays["random_mid"].astype(np.float32),
            random_sd=arrays["random_sd"].astype(np.float32),
            random_d_prime=arrays["random_d_prime"].astype(np.float32),
            positive_label=str(meta.get("positive_label", "positive")),
            negative_label=str(meta.get("negative_label", "negative")),
            category=str(meta.get("category", "other")),
            source=str(meta.get("source", "bundle")),
            n_examples=(int(counts[0]), int(counts[1])),
            d_prime_by_layer={int(k): float(v) for k, v in meta.get("d_prime_by_layer", {}).items()},
        )


def fit_probe(
    name: str,
    X_pos: npt.ArrayLike,
    X_neg: npt.ArrayLike,
    *,
    layer: int,
    model_id: str,
    spec: ConceptSpec | None = None,
    d_prime_by_layer: dict[int, float] | None = None,
) -> Probe:
    """Difference of means plus the random baseline. Deterministic: same inputs, same probe."""
    pos = np.asarray(X_pos, dtype=np.float32)
    neg = np.asarray(X_neg, dtype=np.float32)
    if pos.ndim != 2 or neg.ndim != 2 or pos.shape[1] != neg.shape[1]:
        raise ValueError("X_pos and X_neg must be [n, d_model] with the same d_model")
    if len(pos) < 2 or len(neg) < 2:
        raise ValueError("a probe needs at least two examples per pole")
    diff = pos.mean(0) - neg.mean(0)
    gap = float(np.linalg.norm(diff))
    if gap == 0.0:
        raise ValueError("the two poles have identical mean activations; nothing to probe")
    unit = (diff / gap).astype(np.float32)
    mid, sd, d_prime = _separation(pos @ unit, neg @ unit)
    accuracy = float(((pos @ unit > mid).sum() + (neg @ unit <= mid).sum()) / (len(pos) + len(neg)))
    rand = random_directions(pos.shape[1], seed_for(name, model_id, layer))
    stats = np.array([_separation(pos @ r, neg @ r) for r in rand], dtype=np.float32)
    return Probe(
        name=name,
        model=model_id,
        layer=layer,
        direction=unit,
        mid=mid,
        sd=sd,
        class_gap=gap,
        d_prime=d_prime,
        accuracy=accuracy,
        random_dirs=rand,
        random_mid=stats[:, 0],
        random_sd=stats[:, 1],
        random_d_prime=stats[:, 2],
        positive_label=spec.positive_label if spec else "positive",
        negative_label=spec.negative_label if spec else "negative",
        category=spec.category if spec else "other",
        source=spec.source if spec else "user",
        n_examples=(len(pos), len(neg)),
        d_prime_by_layer=dict(d_prime_by_layer or {}),
    )


def response_means(
    model: Any, tokenizer: Any, examples: Sequence[Example], layers: Sequence[int]
) -> dict[int, FloatArray]:
    """Each example's reply tokens, averaged, at each of `layers`: {layer: [n_examples, d_model]}.

    The reply is rendered with the chat template after the user prompt, exactly where a generated reply sits, and the
    first reply token is skipped when there are others (as sandbagkit's `response_mean` does).
    """
    import torch
    from sandbagkit.probing.hooks import capture_layers

    from steerlab.decode import render_messages

    model_type = str(getattr(model.config, "model_type", "") or "")
    device = next(model.parameters()).device
    rows: dict[int, list[FloatArray]] = {layer: [] for layer in layers}
    with torch.no_grad(), capture_layers(model, list(layers)) as captured:
        for example in examples:
            head = render_messages(tokenizer, [{"role": "user", "content": example.prompt}], model_type=model_type)
            head_ids = tokenizer(head, add_special_tokens=False).input_ids
            full_ids = tokenizer(head + example.response, add_special_tokens=False).input_ids
            start = min(len(head_ids), len(full_ids) - 1)
            if len(full_ids) - start > 1:
                start += 1
            model(input_ids=torch.tensor([full_ids], device=device), use_cache=False)
            for layer in layers:
                rows[layer].append(captured[layer][0, start:].float().mean(0).cpu().numpy())
    return {layer: np.stack(values).astype(np.float32) for layer, values in rows.items()}


def default_layer(n_layers: int) -> int:
    """Where the bundled probes read: about 60% of the way up, where concepts are usually linear and steerable."""
    return max(0, min(n_layers - 1, round(0.6 * (n_layers - 1))))


def fit_concept(
    model: Any, tokenizer: Any, spec: ConceptSpec, *, model_id: str, layer: int, profile: bool = True
) -> Probe:
    """Run the examples, fit at `layer`, and record d' at every layer (the probe's layer-by-layer profile)."""
    from sandbagkit.probing.hooks import num_layers

    layers = list(range(num_layers(model))) if profile else [layer]
    if layer not in layers:
        raise ValueError(f"layer {layer} is outside 0..{len(layers) - 1}")
    pos = response_means(model, tokenizer, spec.positive, layers)
    neg = response_means(model, tokenizer, spec.negative, layers)
    profile_scores = {}
    for each in layers:
        diff = pos[each].mean(0) - neg[each].mean(0)
        norm = float(np.linalg.norm(diff))
        if norm > 0:
            unit = diff / norm
            profile_scores[each] = _separation(pos[each] @ unit, neg[each] @ unit)[2]
    return fit_probe(
        spec.name, pos[layer], neg[layer], layer=layer, model_id=model_id, spec=spec, d_prime_by_layer=profile_scores
    )


def model_slug(model_id: str) -> str:
    return slug(model_id.replace("/", "--").lower())


class ProbeBank:
    """The probes for one model, cached as `probe` artifacts named `<model slug>.<concept>`."""

    def __init__(self, store: ArtifactStore, model_id: str) -> None:
        self.store = store
        self.model_id = model_id
        self.probes: dict[str, Probe] = {}

    def key(self, name: str) -> str:
        return slug(f"{model_slug(self.model_id)}.{name}")

    def load_cached(self) -> list[str]:
        """Every cached probe for this model. Returns the names loaded."""
        loaded = []
        for meta in self.store.list("probe"):
            if meta.get("model") != self.model_id:
                continue
            try:
                probe = Probe.from_artifact(*self.store.load("probe", meta["name"]))
            except (KeyError, ValueError):
                continue
            self.probes[probe.name] = probe
            loaded.append(probe.name)
        return loaded

    def add(self, probe: Probe) -> None:
        meta, arrays = probe.to_artifact()
        meta["concept"] = probe.name
        meta["name"] = self.key(probe.name)
        self.store.save(meta, arrays)
        self.probes[probe.name] = probe

    def remove(self, name: str) -> None:
        self.probes.pop(name, None)
        self.store.delete("probe", self.key(name))

    def ensure_bundle(
        self,
        model: Any,
        tokenizer: Any,
        *,
        layer: int,
        on_progress: Callable[[str, int, int], None] | None = None,
        bundle: Sequence[ConceptSpec] | None = None,
    ) -> list[str]:
        """Fit every bundled concept not cached yet. Returns the names fitted now."""
        self.load_cached()
        specs = list(bundle if bundle is not None else load_bundle())
        fitted = []
        for index, spec in enumerate(specs):
            if spec.name in self.probes:
                continue
            if on_progress is not None:
                on_progress(spec.name, index, len(specs))
            self.add(fit_concept(model, tokenizer, spec, model_id=self.model_id, layer=layer))
            fitted.append(spec.name)
        if on_progress is not None:
            on_progress("", len(specs), len(specs))
        return fitted

    def read(self, h_by_layer: dict[int, FloatArray], names: Sequence[str] | None = None) -> dict[str, Any]:
        """{name: {"z": [...], "band": [...]}} for activations `{layer: [rows, d_model]}`."""
        out = {}
        for name in names if names is not None else list(self.probes):
            probe = self.probes.get(name)
            if probe is None or probe.layer not in h_by_layer:
                continue
            h = h_by_layer[probe.layer]
            out[name] = {"z": probe.read(h).tolist(), "band": probe.random_band(h).tolist()}
        return out
