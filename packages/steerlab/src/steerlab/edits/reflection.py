"""Concept edits in the weights: mirror, project out or amplify an axis in chosen MLP neurons and attention heads.

Adapted from stdlogits/neuron-reflection (MIT, https://github.com/stdlogits/neuron-reflection, `neuron_reflection/
edit.py` and `axes.py`): the dial, the gain folding, the head parts and the word-position rule are theirs; the
architecture adapter, the unit parsing, the controls and the stack integration are ours.

The dial. One unit-length axis `d` and one number `k` give the matrix H = I − k d dᵀ:

    k = 2   a mirror (reflection): the d-component flips sign, everything else stays. H is its own inverse.
    k = 1   a projection: the d-component is removed (directional ablation).
    k = 0   nothing changes.
    k < 0   the d-component is stretched by 1 − k (amplify).

An MLP neuron in a gated (SwiGLU / GeGLU) layer is three vectors: a gate row and an up row that READ the normed
residual `g ⊙ x / rms(x)`, and a down column that WRITES into the residual. The edit puts the read rows through
H_read (with the norm gain `g` folded in, so it acts on what the row really sees) and the write column through
H_write. Reads and writes take their own axes: the read axis lives at the MLP input, the write axis at the layer
output. For k = 2 the edited neuron computes exactly H_write · N(H_read · x), because a mirror keeps rms(x)
(tests/edits/test_reflection.py checks this on tiny random models).

Attention heads take the same edit: Q, K and V rows are reads (input-norm gain folded in), O columns are writes.
K and V are shared by a group of query heads (GQA), so editing one head's K or V edits its whole group. QK-norm and
RoPE act after the projection, inside the head, so they are untouched.

Restore. The first time an edit touches a weight tensor, the Editor keeps a CPU copy of it; `restore()` copies the
originals back, so logits after a restore are bit-identical to before (tested with torch.equal).

Architectures: Qwen2 / Qwen2.5 / Qwen3, Llama, Mistral (SwiGLU, pre-norm only), and Gemma 2 / 3 / 4 text decoders
(GeGLU, with extra post-norms). Gemma's post-attention and post-feedforward norms rescale a layer's summed output;
the write edit folds their gain in, but their 1/rms depends on every unit of the layer, so on Gemma the identity
H_write · N(H_read · x) holds only approximately (`Editor.exact` says which).

Axis sources (every builder returns `Axes`, one read and one write direction per layer):
    sentence_axes   paired sentences that differ in the concept word: residual difference at the word
    pair_axes       single words through neutral frames (activations), or the embedding / unembedding ("embed")
    template_axes   paired chat prompts: residual difference over the chat template after the user turn
    random_axes_like   the control: random unit directions on the same layers

Unit selection: `select_top` (the units whose weights load most on the axis, per layer), `Units.parse` (ids such as
"L8.N344", "L8/n344+" from the Units room basket, "L8.H3"), and `random_units_like` (the control: as many units per
layer, at random).

Importing this module does not import torch; each function imports it when called.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import torch

Sides = Literal["both", "reads", "writes"]
SIDES: tuple[str, ...] = ("both", "reads", "writes")
HEAD_PARTS = "qkvo"
UNIT_ID = re.compile(
    r"^\s*l(?:ayer)?\s*(\d+)\s*[/·.\s_-]*\s*(n(?:euron)?|h(?:ead)?)\s*(\d+)\s*([+\-−]|plus|minus|pos|neg)?\s*$",
    re.IGNORECASE,
)
UNITS_URL = re.compile(r"^(?:[a-z]+://[^/]+)?/?units/(?P<slug>[^/]+)/(?P<rest>.+)$", re.IGNORECASE)
"""The Units room's address (`/units/<model slug>/L8/n344+`), full URL or path."""


def model_slug(model_id: str) -> str:
    """The Units room's model slug: `Qwen/Qwen3-0.6B` -> `qwen3-0.6b` (`steerlab.atlas.model_slug` when installed)."""
    try:
        import importlib

        return str(importlib.import_module("steerlab.atlas").model_slug(model_id))
    except (ImportError, AttributeError):
        name = model_id.rstrip("/").split("/")[-1].lower()
        return re.sub(r"[^a-z0-9.]+", "-", name).strip("-") or "model"


def unit_href(model_id: str, kind: str, layer: int, index: int) -> str:
    """The Units room's link to a unit: `/units/<slug>/L8/n344+` (a neuron, linked at its + pole; an edit moves both
    poles' weights) or `/units/<slug>/L3/h5`. Uses `steerlab.atlas.unit_href` when installed, so links never drift."""
    try:
        import importlib

        atlas_href = importlib.import_module("steerlab.atlas").unit_href
    except (ImportError, AttributeError):
        tail = f"L{layer}/h{index}" if kind == "head" else f"L{layer}/n{index}+"
        return f"/units/{model_slug(model_id)}/{tail}"
    return str(atlas_href(model_id, layer, index) if kind == "head" else atlas_href(model_id, layer, index, 1))


# ---------------------------------------------------------------------------------------------------------------------
# Units and axes


@dataclass
class Units:
    """Which units an edit touches: `{layer: [neuron, ...]}` and `{layer: [query head, ...]}`."""

    neurons: dict[int, list[int]] = field(default_factory=dict)
    heads: dict[int, list[int]] = field(default_factory=dict)

    def count(self) -> dict[str, int]:
        return {"neurons": sum(map(len, self.neurons.values())), "heads": sum(map(len, self.heads.values()))}

    def layers(self) -> list[int]:
        return sorted(set(self.neurons) | set(self.heads))

    def ids(self) -> list[str]:
        out = [f"L{layer}.N{n}" for layer in sorted(self.neurons) for n in self.neurons[layer]]
        return out + [f"L{layer}.H{h}" for layer in sorted(self.heads) for h in self.heads[layer]]

    def to_json(self) -> dict[str, dict[str, list[int]]]:
        return {
            "neurons": {str(layer): sorted(v) for layer, v in sorted(self.neurons.items()) if v},
            "heads": {str(layer): sorted(v) for layer, v in sorted(self.heads.items()) if v},
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Units:
        def table(raw: dict[str, Any] | None) -> dict[int, list[int]]:
            return {int(layer): sorted({int(i) for i in v}) for layer, v in (raw or {}).items() if v}

        return cls(table(data.get("neurons")), table(data.get("heads")))

    @classmethod
    def parse(cls, ids: Sequence[str], model_id: str | None = None) -> Units:
        """Unit ids from the Trace and Units rooms: "L8.N344", "L8/n344+", "L8·n344−", "L8 n344 -" (a pole; both
        poles are one neuron's weights, so the sign is dropped), "L8.H3" / "L20h17" for query heads, and the Units
        room's address "/units/<slug>/L8/n344+" (full URL or path). With `model_id`, an address whose slug names
        another model is refused."""
        units = cls()
        for raw in ids:
            text = raw.strip().replace("%2B", "+").replace("%2b", "+")
            url = UNITS_URL.match(text)
            if url:
                if model_id is not None and url["slug"].lower() != model_slug(model_id):
                    raise ValueError(f"{raw!r} is a unit of {url['slug']}, not of the loaded {model_slug(model_id)}")
                text = url["rest"]
            match = UNIT_ID.match(text)
            if not match:
                raise ValueError(f"unit id {raw!r} is not like L8.N344, L8/n344+, L8.H3 or /units/<model>/L8/n344+")
            layer, kind, index = int(match[1]), match[2][0].lower(), int(match[3])
            table = units.neurons if kind == "n" else units.heads
            if index not in table.setdefault(layer, []):
                table[layer].append(index)
        for table in (units.neurons, units.heads):
            for layer in table:
                table[layer].sort()
        return units


@dataclass
class Axes:
    """Per layer, a unit read direction (at the MLP input) and a unit write direction (at the layer output), float32
    on the CPU, plus a JSON description of where they came from (enough to rebuild them)."""

    reads: dict[int, torch.Tensor]
    writes: dict[int, torch.Tensor]
    source: dict[str, Any] = field(default_factory=dict)

    def layers(self) -> list[int]:
        return sorted(self.reads)

    def neuron_axes(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.reads[layer], self.writes[layer]

    def head_axes(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """A head at layer L reads the residual the layer before wrote (its write axis; layer 0 uses its own read
        axis) and writes the residual its own MLP reads (layer L's read axis)."""
        read = self.writes[layer - 1] if layer and (layer - 1) in self.writes else self.reads[layer]
        return read, self.reads[layer]


def unit(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(1e-12)  # type: ignore[no-any-return,unused-ignore]


# ---------------------------------------------------------------------------------------------------------------------
# The architecture adapter


def find_layers(model: Any) -> Any:
    """The decoder layers of a causal LM: `model.model.layers` (Qwen, Llama, Mistral, Gemma text), or the language
    model inside a multimodal wrapper (Gemma 3/4 conditional generation)."""
    paths = ("model.layers", "model.language_model.layers", "language_model.model.layers", "model.text_model.layers")
    for path in paths:
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise ValueError("cannot find the decoder layers (expected model.model.layers)")


def _norm_gain(norm: Any) -> torch.Tensor:
    """The per-channel gain an RMSNorm applies after dividing by rms: `weight` (Qwen, Llama, Mistral, Gemma 4) or
    `1 + weight` (Gemma 2/3). Decided by running the norm on a constant vector, whose rms is known."""
    import torch

    weight = getattr(norm, "weight", None)
    d = int(weight.shape[0]) if weight is not None else int(norm.normalized_shape[0])
    eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)))
    dtype = weight.dtype if weight is not None else torch.float32
    device = weight.device if weight is not None else "cpu"
    with torch.no_grad():
        probe = norm(torch.ones(1, d, dtype=dtype, device=device))[0].float().cpu() * math.sqrt(1 + eps)
    if weight is None:
        return probe  # type: ignore[no-any-return,unused-ignore]
    w = weight.detach().float().cpu()
    candidates = [w, 1 + w]
    return min(candidates, key=lambda c: float((c - probe).abs().max()))  # type: ignore[no-any-return,unused-ignore]


class Arch:
    """Where each piece lives in one decoder: MLP projections, attention projections, the norms around them."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.layers = find_layers(model)
        self.n_layers = len(self.layers)
        first = self.layers[0]
        if not all(hasattr(first.mlp, name) for name in ("gate_proj", "up_proj", "down_proj")):
            raise ValueError("the Edit room needs gated MLPs (gate_proj / up_proj / down_proj): SwiGLU or GeGLU")
        config = getattr(model.config, "text_config", None) or model.config
        self.n_heads = int(config.num_attention_heads)
        self.n_kv = int(getattr(config, "num_key_value_heads", None) or self.n_heads)
        self.d_model = int(config.hidden_size)
        self.gemma = hasattr(first, "pre_feedforward_layernorm")
        self.family = type(model).__name__

    def mlp(self, layer: int) -> tuple[Any, Any, Any]:
        m = self.layers[layer].mlp
        return m.gate_proj, m.up_proj, m.down_proj

    def d_mlp(self, layer: int) -> int:
        return int(self.layers[layer].mlp.up_proj.weight.shape[0])

    def mlp_in_norm(self, layer: int) -> Any:
        block = self.layers[layer]
        return block.pre_feedforward_layernorm if self.gemma else block.post_attention_layernorm

    def mlp_out_norm(self, layer: int) -> Any | None:
        return getattr(self.layers[layer], "post_feedforward_layernorm", None) if self.gemma else None

    def attn_in_norm(self, layer: int) -> Any:
        return self.layers[layer].input_layernorm

    def attn_out_norm(self, layer: int) -> Any | None:
        return self.layers[layer].post_attention_layernorm if self.gemma else None

    def attn(self, layer: int) -> tuple[Any, Any | None, Any | None, Any]:
        a = self.layers[layer].self_attn
        return a.q_proj, getattr(a, "k_proj", None), getattr(a, "v_proj", None), a.o_proj

    def head_dim(self, layer: int) -> int:
        return int(self.layers[layer].self_attn.q_proj.weight.shape[0]) // self.n_heads

    def kv_heads(self, layer: int) -> int:
        k = self.attn(layer)[1]
        return int(k.weight.shape[0]) // self.head_dim(layer) if k is not None else self.n_kv


# ---------------------------------------------------------------------------------------------------------------------
# The editor


class Editor:
    """Applies reflection edits to a model's weights in place and undoes them exactly. Edits compose: each call edits
    the current weights; `restore()` brings back the originals of everything touched since the last restore."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.arch = Arch(model)
        self._saved: dict[int, tuple[Any, Any]] = {}
        self._gains: dict[tuple[str, int], torch.Tensor | None] = {}
        self.log: list[dict[str, Any]] = []

    @property
    def edited(self) -> bool:
        return bool(self._saved)

    @property
    def exact(self) -> bool:
        """Whether k = 2 edits conjugate units exactly (no post-norms between a unit's write and the residual)."""
        return not self.arch.gemma

    def _gain(self, which: str, layer: int) -> torch.Tensor | None:
        key = (which, layer)
        if key not in self._gains:
            norm = {
                "mlp_in": self.arch.mlp_in_norm,
                "mlp_out": self.arch.mlp_out_norm,
                "attn_in": self.arch.attn_in_norm,
                "attn_out": self.arch.attn_out_norm,
            }[which](layer)
            self._gains[key] = None if norm is None else _norm_gain(norm)
        return self._gains[key]

    def _touch(self, param: Any) -> None:
        if id(param) not in self._saved:
            self._saved[id(param)] = (param, param.detach().cpu().clone())

    def restore(self) -> None:
        """Copy every touched weight back from its saved original (bit-identical)."""
        import torch

        with torch.no_grad():
            for param, original in self._saved.values():
                param.copy_(original.to(param.device))
        self._saved.clear()
        self.log.clear()

    @contextmanager
    def applied(
        self, axes: Axes, units: Units, k: float = 2.0, sides: Sides = "both", parts: str = HEAD_PARTS
    ) -> Iterator[dict[str, Any]]:
        """Apply one edit for the duration of the block, then restore everything touched."""
        try:
            yield self.reflect(axes, units, k, sides, parts)
        finally:
            self.restore()

    # loadings

    def neuron_shares(self, layer: int, read: torch.Tensor, write: torch.Tensor) -> torch.Tensor:
        """`[3, d_mlp]`: the share of each neuron's gain-folded gate row and up row along `read`, and of its down
        column (post-norm gain folded, on Gemma) along `write`. Squared cosines, between 0 and 1."""
        import torch

        gate, up, down = self.arch.mlp(layer)
        g = self._gain("mlp_in", layer)
        gp = self._gain("mlp_out", layer)
        with torch.no_grad():
            rows = [p.weight.detach().float().cpu() * g for p in (gate, up)]
            col = down.weight.detach().float().cpu()
            if gp is not None:
                col = gp[:, None] * col
            shares = [(r @ read).pow(2) / r.pow(2).sum(1).clamp_min(1e-30) for r in rows]
            shares.append((write @ col).pow(2) / col.pow(2).sum(0).clamp_min(1e-30))
        return torch.stack(shares)

    def head_shares(self, layer: int, read: torch.Tensor, write: torch.Tensor) -> torch.Tensor:
        """`[4, n_heads]`: per query head, the share along `read` of its gain-folded Q rows, of its KV group's K and V
        rows, and the share along `write` of its O columns."""
        import torch

        q, k, v, o = self.arch.attn(layer)
        hd, nh = self.arch.head_dim(layer), self.arch.n_heads
        nkv = self.arch.kv_heads(layer)
        g = self._gain("attn_in", layer)
        gp = self._gain("attn_out", layer)

        def rows(proj: Any, n: int) -> torch.Tensor:
            W = (proj.weight.detach().float().cpu() * g).view(n, hd, -1)
            return (W @ read).pow(2).sum(1) / W.pow(2).sum((1, 2)).clamp_min(1e-30)  # type: ignore[no-any-return,unused-ignore]

        with torch.no_grad():
            sq = rows(q, nh)
            sk = rows(k, nkv).repeat_interleave(nh // nkv) if k is not None else torch.zeros(nh)
            sv = rows(v, nkv).repeat_interleave(nh // nkv) if v is not None else torch.zeros(nh)
            cols = o.weight.detach().float().cpu()
            if gp is not None:
                cols = gp[:, None] * cols
            cols = cols.view(-1, nh, hd)
            so = torch.einsum("d,dhe->he", write, cols).pow(2).sum(1) / cols.pow(2).sum((0, 2)).clamp_min(1e-30)
        return torch.stack([sq, sk, sv, so])

    # the edit

    def reflect(
        self, axes: Axes, units: Units, k: float = 2.0, sides: Sides = "both", parts: str = HEAD_PARTS
    ) -> dict[str, Any]:
        """Put the chosen neurons' and heads' reads through H_read = I − k r rᵀ and their writes through
        H_write = I − k w wᵀ, with each layer's axes. `sides` limits neurons to their reads or writes; `parts`
        names the head parts edited (q, k, v reads; o writes)."""
        import torch

        if sides not in SIDES:
            raise ValueError(f"sides must be one of {SIDES}, not {sides!r}")
        if not parts or not set(parts) <= set(HEAD_PARTS):
            raise ValueError(f"head parts must be letters of {HEAD_PARTS!r}, not {parts!r}")
        self.check(units)
        missing = [layer for layer in units.layers() if layer not in axes.reads]
        if missing:
            raise ValueError(f"the axis has no direction at layers {missing}")
        read_share, write_share = [], []
        with torch.no_grad():
            for layer, idx in sorted(units.neurons.items()):
                if not idx:
                    continue
                read, write = axes.neuron_axes(layer)
                sh = self.neuron_shares(layer, read, write)[:, idx]
                read_share.append(sh[:2].max(0).values)
                write_share.append(sh[2])
                self._reflect_neurons(layer, idx, read, write, k, sides)
            for layer, heads in sorted(units.heads.items()):
                if not heads:
                    continue
                read, write = axes.head_axes(layer)
                sh = self.head_shares(layer, read, write)[:, heads]
                read_share.append(sh[:3].max(0).values)
                write_share.append(sh[3])
                self._reflect_heads(layer, heads, read, write, k, parts)
        info = {
            **units.count(),
            "k": k,
            "sides": sides,
            "parts": parts,
            "exact": self.exact,
            "median_read_share": float(torch.cat(read_share).median()) if read_share else 0.0,
            "median_write_share": float(torch.cat(write_share).median()) if write_share else 0.0,
        }
        self.log.append(info)
        return info

    def check(self, units: Units) -> None:
        """Refuse layers, neurons or heads the model does not have (ValueError)."""
        for layer, idx in units.neurons.items():
            if not 0 <= layer < self.arch.n_layers:
                raise ValueError(f"layer {layer} is outside 0..{self.arch.n_layers - 1}")
            n = self.arch.d_mlp(layer)
            bad = [i for i in idx if not 0 <= i < n]
            if bad:
                raise ValueError(f"neurons {bad[:5]} are outside layer {layer}'s 0..{n - 1}")
        for layer, heads in units.heads.items():
            if not 0 <= layer < self.arch.n_layers:
                raise ValueError(f"layer {layer} is outside 0..{self.arch.n_layers - 1}")
            bad = [h for h in heads if not 0 <= h < self.arch.n_heads]
            if bad:
                raise ValueError(f"heads {bad} are outside 0..{self.arch.n_heads - 1}")

    def _reflect_rows(self, param: Any, rows: torch.Tensor, gain: torch.Tensor, axis: torch.Tensor, k: float) -> None:
        """Rows that read the normed input: the gain-folded row r ⊙ g goes through H, stored back divided by g. Only
        the axis component changes: r' = r − k ((r ⊙ g) · d) (d / g)."""
        self._touch(param)
        dev = param.device
        g = gain.to(dev)
        d = axis.to(dev)
        W = param[rows].float()
        coef = (W * g) @ d
        W = W - k * coef[:, None] * (d / _safe(g))[None, :]
        param[rows] = W.to(param.dtype)

    def _reflect_cols(
        self, param: Any, cols: torch.Tensor, gain: torch.Tensor | None, axis: torch.Tensor, k: float
    ) -> None:
        """Columns that write the residual (through a post-norm gain on Gemma): w' = w − k d (d · (g ⊙ w)) / g."""
        self._touch(param)
        dev = param.device
        d = axis.to(dev)
        W = param[:, cols].float()
        if gain is None:
            W = W - k * d[:, None] * (d @ W)[None, :]
        else:
            g = gain.to(dev)
            W = W - k * (d / _safe(g))[:, None] * (d @ (g[:, None] * W))[None, :]
        param[:, cols] = W.to(param.dtype)

    def _reflect_neurons(self, layer: int, idx: list[int], read: Any, write: Any, k: float, sides: str) -> None:
        import torch

        gate, up, down = self.arch.mlp(layer)
        rows = torch.tensor(sorted(set(idx)), dtype=torch.long, device=gate.weight.device)
        if sides in ("both", "reads"):
            g = self._gain("mlp_in", layer)
            assert g is not None
            for proj in (gate, up):
                self._reflect_rows(proj.weight, rows, g, read, k)
        if sides in ("both", "writes"):
            self._reflect_cols(down.weight, rows.to(down.weight.device), self._gain("mlp_out", layer), write, k)

    def _reflect_heads(self, layer: int, heads: list[int], read: Any, write: Any, k: float, parts: str) -> None:
        import torch

        q, kp, v, o = self.arch.attn(layer)
        hd = self.arch.head_dim(layer)
        per = self.arch.n_heads // self.arch.kv_heads(layer)
        heads_t = sorted(set(heads))
        groups = sorted({h // per for h in heads_t})

        def span(ids: list[int], device: Any) -> torch.Tensor:
            return torch.cat([torch.arange(u * hd, (u + 1) * hd) for u in ids]).to(device)

        g = self._gain("attn_in", layer)
        assert g is not None
        for letter, proj, ids in (("q", q, heads_t), ("k", kp, groups), ("v", v, groups)):
            if letter in parts and proj is not None:
                self._reflect_rows(proj.weight, span(ids, proj.weight.device), g, read, k)
        if "o" in parts:
            self._reflect_cols(o.weight, span(heads_t, o.weight.device), self._gain("attn_out", layer), write, k)


def _safe(g: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """`g` with entries smaller than `eps` in size pushed to ±eps (a zero gain cannot be divided by)."""
    import torch

    sign = torch.where(g < 0, -1.0, 1.0).to(g.dtype)
    return torch.where(g.abs() < eps, eps * sign, g)


# ---------------------------------------------------------------------------------------------------------------------
# Unit selection and controls


def select_top(
    editor: Editor,
    axes: Axes,
    per_layer: int = 8,
    *,
    layers: Sequence[int] | None = None,
    kind: Literal["neurons", "heads"] = "neurons",
    parts: str = HEAD_PARTS,
    sides: Sides = "both",
) -> Units:
    """Per layer, the `per_layer` units whose weights load most on the axis (the largest share among the parts that
    will be edited). `layers` limits the scope; by default every layer the axis covers."""
    scope = sorted(layers) if layers is not None else axes.layers()
    units = Units()
    for layer in scope:
        if kind == "neurons":
            sh = editor.neuron_shares(layer, *axes.neuron_axes(layer))
            used = {"both": [0, 1, 2], "reads": [0, 1], "writes": [2]}[sides]
        else:
            sh = editor.head_shares(layer, *axes.head_axes(layer))
            used = ["qkvo".index(c) for c in parts]
        load = sh[used].max(0).values
        n = min(per_layer, len(load))
        if n <= 0:
            continue
        top = sorted(int(i) for i in load.topk(n).indices)
        (units.neurons if kind == "neurons" else units.heads)[layer] = top
    return units


def random_units_like(editor: Editor, units: Units, seed: int = 0) -> Units:
    """The random-unit control: as many neurons (heads) per layer as `units` has, drawn at random from the units it
    does not touch."""
    import random

    rng = random.Random(seed)
    out = Units()
    for layer, idx in sorted(units.neurons.items()):
        taken = set(idx)
        pool = [i for i in range(editor.arch.d_mlp(layer)) if i not in taken]
        out.neurons[layer] = sorted(rng.sample(pool, min(len(idx), len(pool))))
    for layer, heads in sorted(units.heads.items()):
        taken = set(heads)
        pool = [h for h in range(editor.arch.n_heads) if h not in taken] or list(range(editor.arch.n_heads))
        out.heads[layer] = sorted(rng.sample(pool, min(len(heads), len(pool))))
    return out


def random_axes_like(axes: Axes, seed: int = 0) -> Axes:
    """The random-axis control: a random unit read and write direction at each of the axis's layers."""
    import torch

    gen = torch.Generator().manual_seed(seed)
    d = next(iter(axes.reads.values())).shape[0]
    reads = {layer: unit(torch.randn(d, generator=gen)) for layer in axes.layers()}
    writes = {layer: unit(torch.randn(d, generator=gen)) for layer in axes.layers()}
    return Axes(reads, writes, {"kind": "random", "seed": seed, "like": axes.source})


# ---------------------------------------------------------------------------------------------------------------------
# Axes from activations


def capture(model: Any, ids: Sequence[int], positions: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """`[n_layers, len(positions), d]` residuals at the MLP input (before its norm) and at each layer's output."""
    import torch

    arch = Arch(model)
    mid: list[Any] = [None] * arch.n_layers
    out: list[Any] = [None] * arch.n_layers
    pos = list(positions)
    handles = []
    for i in range(arch.n_layers):

        def pre(module: Any, args: tuple[Any, ...], i: int = i) -> None:
            mid[i] = args[0][0, pos].detach().float().cpu()

        def post(module: Any, args: tuple[Any, ...], output: Any, i: int = i) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            out[i] = hidden[0, pos].detach().float().cpu()

        handles.append(arch.mlp_in_norm(i).register_forward_pre_hook(pre))
        handles.append(arch.layers[i].register_forward_hook(post))
    try:
        device = next(model.parameters()).device
        with torch.no_grad():
            model(torch.tensor([list(ids)], device=device), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return torch.stack(mid), torch.stack(out)


def word_positions(x: Sequence[int], y: Sequence[int]) -> tuple[list[int], list[int]]:
    """In two token lists that differ in one word, the last token where they differ (counted back from their shared
    ending) and the token after it, where an inflected word completes (neuron-reflection's rule)."""
    s = 0
    while s < min(len(x), len(y)) - 1 and x[-1 - s] == y[-1 - s]:
        s += 1
    if s == 0:
        raise ValueError("paired sentences need a shared ending after the concept word")
    return [len(x) - s - 1, len(x) - s], [len(y) - s - 1, len(y) - s]


def _encode(tokenizer: Any, text: str) -> list[int]:
    ids: list[int] = tokenizer(text, add_special_tokens=True).input_ids
    return ids


def _from_diffs(mids: list[torch.Tensor], outs: list[torch.Tensor], source: dict[str, Any]) -> Axes:
    """Each axis is the normalized sum of the pairs' unit differences, so every pair counts equally."""
    import torch

    mid = unit(torch.stack(mids).sum(0))  # [L, d]
    out = unit(torch.stack(outs).sum(0))
    n = mid.shape[0]
    return Axes({layer: mid[layer] for layer in range(n)}, {layer: out[layer] for layer in range(n)}, source)


def sentence_axes(model: Any, tokenizer: Any, pairs: Sequence[tuple[str, str]]) -> Axes:
    """Word axes: paired sentences that differ in the concept word ("my mother lives" / "my father lives"); at
    every layer, the residual difference at the word (source minus target), averaged over the pairs."""
    if not pairs:
        raise ValueError("give at least one sentence pair")
    mids, outs = [], []
    for a, b in pairs:
        x, y = _encode(tokenizer, a), _encode(tokenizer, b)
        px, py = word_positions(x, y)
        (ma, oa), (mb, ob) = capture(model, x, px), capture(model, y, py)
        mids.append(unit(ma.mean(1) - mb.mean(1)))
        outs.append(unit(oa.mean(1) - ob.mean(1)))
    return _from_diffs(mids, outs, {"kind": "sentences", "pairs": [list(p) for p in pairs]})


FRAMES = (
    "I thought about the {w} all day.",
    "She said the word {w} twice.",
    "This story is about a {w} and a friend.",
    "Everyone knows what a {w} is.",
)
"""Neutral sentences a single word is dropped into, for activation axes from word pairs."""


def pair_axes(
    model: Any,
    tokenizer: Any,
    pairs: Sequence[tuple[str, str]],
    method: Literal["activations", "embed"] = "activations",
    frames: Sequence[str] = FRAMES,
) -> Axes:
    """Axes from word pairs (mother/father). `activations`: each pair dropped into neutral frames, the residual
    difference at the word. `embed`: no forward pass; the read axis is the input-embedding difference and the write
    axis the unembedding difference (with the final norm's gain), the same at every layer (a lens with an identity
    Jacobian: what the token is, and what makes the model say it)."""
    if not pairs:
        raise ValueError("give at least one word pair")
    clean = [(a.strip(), b.strip()) for a, b in pairs]
    if any(not a or not b or a == b for a, b in clean):
        raise ValueError("each pair needs two different, non-empty words")
    if method == "activations":
        sentences = [(f.format(w=a), f.format(w=b)) for a, b in clean for f in frames]
        axes = sentence_axes(model, tokenizer, sentences)
        axes.source = {"kind": "pairs", "method": "activations", "pairs": [list(p) for p in clean]}
        return axes
    if method != "embed":
        raise ValueError(f"unknown pair method {method!r}")
    import torch

    arch = Arch(model)
    emb = model.get_input_embeddings().weight.detach().float().cpu()
    unemb = model.get_output_embeddings().weight.detach().float().cpu()
    final_norm = getattr(getattr(model, "model", model), "norm", None)
    gain = _norm_gain(final_norm) if final_norm is not None else torch.ones(arch.d_model)
    reads, writes = [], []
    for a, b in clean:
        ia, ib = (tokenizer(" " + w, add_special_tokens=False).input_ids[0] for w in (a, b))
        reads.append(unit(emb[ia]) - unit(emb[ib]))
        writes.append(unit(gain * unemb[ia]) - unit(gain * unemb[ib]))
    read, write = unit(torch.stack(reads).sum(0)), unit(torch.stack(writes).sum(0))
    layers = range(arch.n_layers)
    return Axes(
        {layer: read.clone() for layer in layers},
        {layer: write.clone() for layer in layers},
        {"kind": "pairs", "method": "embed", "pairs": [list(p) for p in clean]},
    )


def chat_ids(tokenizer: Any, user: str) -> tuple[list[int], int]:
    """The chat prompt for one user turn (generation prompt added, thinking off where the template has it) and the
    index of the first token after the user's text."""
    kwargs: dict[str, Any] = {"add_generation_prompt": True, "tokenize": False}
    messages = [{"role": "user", "content": user}]
    try:
        text = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        text = tokenizer.apply_chat_template(messages, **kwargs)
    end = text.rfind(user) + len(user)
    ids: list[int] = tokenizer(text, add_special_tokens=False).input_ids
    head: list[int] = tokenizer(text[:end], add_special_tokens=False).input_ids
    return ids, min(len(head), len(ids) - 1)


def template_axes(model: Any, tokenizer: Any, pairs: Sequence[tuple[str, str]]) -> Axes:
    """Template axes: paired user turns (a plain request and the same request with an instruction); at every layer
    the residual difference averaged over the chat-template tokens after the user's text, where the reply is
    planned."""
    if not pairs:
        raise ValueError("give at least one prompt pair")
    mids, outs = [], []
    for a, b in pairs:
        (x, sx), (y, sy) = chat_ids(tokenizer, a), chat_ids(tokenizer, b)
        (ma, oa), (mb, ob) = capture(model, x, range(sx, len(x))), capture(model, y, range(sy, len(y)))
        mids.append(unit(ma.mean(1) - mb.mean(1)))
        outs.append(unit(oa.mean(1) - ob.mean(1)))
    return _from_diffs(mids, outs, {"kind": "template", "pairs": [list(p) for p in pairs]})


def lens_available() -> bool:
    """Whether a fitted Jacobian lens module (`steerlab.lens.jacobian`) is installed."""
    import importlib.util

    try:
        return importlib.util.find_spec("steerlab.lens.jacobian") is not None
    except (ImportError, ValueError):
        return False


def lens_pair_axes(model: Any, tokenizer: Any, pairs: Sequence[tuple[str, str]], lens: Any = None) -> Axes:
    """Word-pair axes through the Jacobian lens when `steerlab.lens.jacobian` provides `pair_axes`; otherwise the
    activation axes (the source records which one was used)."""
    if lens_available():
        import importlib

        module = importlib.import_module("steerlab.lens.jacobian")
        builder = getattr(module, "pair_axes", None)
        if builder is not None:
            reads, writes = builder(model, tokenizer, list(pairs), lens)
            source = {"kind": "pairs", "method": "lens", "pairs": [list(p) for p in pairs]}
            return Axes(dict(reads), dict(writes), source)
    return pair_axes(model, tokenizer, pairs, "activations")


def build_axes(model: Any, tokenizer: Any, source: dict[str, Any]) -> Axes:
    """Rebuild axes from their JSON description (what an edit artifact stores)."""
    kind = source.get("kind")
    pairs: list[tuple[str, str]] = [(str(p[0]), str(p[1])) for p in source.get("pairs", [])]
    if kind == "sentences":
        return sentence_axes(model, tokenizer, pairs)
    if kind == "pairs":
        method = source.get("method", "activations")
        if method == "lens":
            return lens_pair_axes(model, tokenizer, pairs, source.get("lens"))
        return pair_axes(model, tokenizer, pairs, method)
    if kind == "template":
        return template_axes(model, tokenizer, pairs)
    if kind == "random":
        like = build_axes(model, tokenizer, source.get("like") or {})
        return random_axes_like(like, int(source.get("seed", 0)))
    raise ValueError(f"unknown axis source kind {kind!r}")
