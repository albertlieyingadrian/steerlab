"""Jacobian lenses: read a residual-stream vector as words, through what the rest of the model does to it.

    from steerlab.lens import jacobian
    lenses = jacobian.load("Qwen/Qwen3-0.6B", ("R", "S"))          # stdlogits/rs-lenses from the Hub (cached)
    reader = jacobian.Reader(model, lenses)
    reader.logits(h, layer=8, lens="R")                              # [vocab]: what h makes the model say
    reader.top(h, 8, "S", k=8, z=True)                               # which input tokens put h there, hubs removed

Four lenses, each reading a vector `h` that sits in the residual stream after decoder layer `L` (its "out"):

| lens  | maps `h` with                                    | reads with  | answers                                  |
|-------|--------------------------------------------------|-------------|------------------------------------------|
| logit | nothing                                          | `W_U`       | what `h` says if nothing downstream acts |
| J     | `J_L = mean_t Σ_{t'≥t} ∂h_final(t')/∂h_L(t)`     | `W_U`       | what `h` makes the model say later       |
| R     | the same Jacobian under RelP's rules             | `W_U`       | the same, more faithful on early layers  |
| S     | `K_L = mean_t Σ_{s≤t} ∂h_L(t)/∂e(s)`, RelP rules | `W_E − W̄_E` | which input tokens put `h` there         |

`h_final` is the last layer's output before the final norm. Reads through `W_U` apply the final RMSNorm (so they are
scale-free), then the unembedding; at the last layer the logit, J and R lenses all give the model's real output.

RelP's rules (arXiv:2508.21258, as in the R-lens): residual norms hold their 1/rms constant; SiLU/GELU in an MLP is
linearised as `φ(g)·dg` with `φ = act(g)/g`; the gate × up product passes half of each factor's change. Attention
weights keep their true derivative. Each rule is written `value.detach() + (lin − lin.detach())`, so the forward
values are the model's own and only the derivative changes, for reverse mode (R) and forward mode (S) alike.

Fitting averages exact Jacobians over a corpus (1,000 DCLM documents × 128 tokens for the published lenses), seeded
at positions `[16, T−1)` (attention sinks and the last position left out), documents weighted equally and split into
even and odd halves for a split-half check. The estimator here uses autograd: one backward pass per block of
`dim_batch` output directions for J/R (copies of the document in the batch dimension, copy `i` seeded with basis
direction `i`), one forward-mode pass per block of input directions for S. Every layer comes out of the same pass.

Layer-contrastive reads (`z=True`) z-score each token's score against the same lens's reads of the layer's MLP
writes (`±W_down[:, n]`, as unit vectors), which removes hub tokens that top every read at a layer.

Credits: the lens definitions, the RelP rule set (adapted below from `neuron_lens/rules.py`), the file format and the
fit settings are from stdlogits/neuron-lens (MIT, github.com/stdlogits/neuron-lens), which publishes fitted R- and
S-lenses for Qwen3 0.6B-14B and Gemma 4 12B at huggingface.co/stdlogits/rs-lenses. The J-lens is Gurnee et al. 2026.

File format (the same as rs-lenses): `{"J": {layer: [d, d]}, ...}` for J/R (`J[L] @ h`), `{"K": {layer: [d, d]}}`
for S (`c @ K[L]` reads covector `c`), plus `n_prompts`, `d_model` and `provenance`. Our fits live in
`~/.cache/steerlab/lenses/<slug>/` (or `$STEERLAB_LENS_DIR`), never in the repo.

Importing this module imports neither torch nor transformers.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import types
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

LENSES = ("logit", "J", "R", "S")
FITTED = ("J", "R", "S")
SKIP_FIRST = 16
DIM_BATCH = 128
HALF_RULE_BETA = 0.5
HF_REPO = "stdlogits/rs-lenses"
# Hub id -> the rs-lenses folder. Post-trained and base Qwen3 share sizes but not lenses.
HF_KEYS: dict[str, str] = {
    **{f"Qwen/Qwen3-{s}": f"qwen3-{s.lower()}" for s in ("0.6B", "1.7B", "4B", "8B", "14B")},
    **{f"Qwen/Qwen3-{s}-Base": f"qwen3-{s.lower()}-base" for s in ("0.6B", "1.7B", "4B", "8B", "14B")},
    "google/gemma-4-12B": "gemma-4-12b",
}
HF_FILES = {"R": "r-lens/lens.pt", "S": "s-lens/out.pt"}
ZERO_CENTRED = ("Qwen3_5RMSNorm", "Gemma2RMSNorm", "Gemma3RMSNorm", "GemmaRMSNorm")  # these scale by 1 + weight
RESIDUAL_NORMS = (
    "input_layernorm",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "post_feedforward_layernorm",
)
_GELU_C = math.sqrt(2 / math.pi)


def slug(model_id: str) -> str:
    """`Qwen/Qwen2.5-0.5B-Instruct` -> `qwen2.5-0.5b-instruct` (also the Units room's model segment)."""
    name = re.sub(r"[^a-z0-9.]+", "-", model_id.rstrip("/").split("/")[-1].lower()).strip("-")
    return name or "model"


def lens_dir() -> Path:
    return Path(os.environ.get("STEERLAB_LENS_DIR") or Path.home() / ".cache" / "steerlab" / "lenses")


def own_path(model_id: str, lens: str) -> Path:
    return lens_dir() / slug(model_id) / f"{lens.lower()}-lens.pt"


# Where each lens comes from


def _hub_cached(model_id: str, lens: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache(HF_REPO, f"{HF_KEYS[model_id]}/{HF_FILES[lens]}"), str)


def availability(model_id: str) -> dict[str, dict[str, Any]]:
    """Each lens: `ready` (on disk), `downloadable` (on the Hub, not yet cached) or `needs a fit`, with its source."""
    out: dict[str, dict[str, Any]] = {
        "logit": {"status": "ready", "source": "the model's own final norm and unembedding"}
    }
    for lens in FITTED:
        own = own_path(model_id, lens)
        if own.exists():
            out[lens] = {"status": "ready", "source": f"steerlab fit ({own.parent.name}/{own.name})"}
        elif lens in HF_FILES and model_id in HF_KEYS:
            where = f"{HF_REPO}/{HF_KEYS[model_id]}/{HF_FILES[lens]}"
            out[lens] = {"status": "ready" if _hub_cached(model_id, lens) else "downloadable", "source": where}
        else:
            out[lens] = {"status": "needs a fit", "source": "fit one with steerlab.lens.jacobian.fit (a GPU)"}
    return out


def load_one(model_id: str, lens: str, *, download: bool = True) -> dict[int, torch.Tensor]:
    """One fitted lens as `{layer: [d, d]}` (float32, CPU): our own fit first, then stdlogits/rs-lenses."""
    import torch

    if lens not in FITTED:
        raise ValueError(f"unknown lens {lens!r}; fitted lenses are {FITTED}")
    own = own_path(model_id, lens)
    if own.exists():
        blob = torch.load(own, map_location="cpu", weights_only=True)
    elif lens in HF_FILES and model_id in HF_KEYS:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(HF_REPO, f"{HF_KEYS[model_id]}/{HF_FILES[lens]}", local_files_only=not download)
        blob = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    else:
        raise ValueError(f"no {lens}-lens for {model_id}: fit one with steerlab.lens.jacobian.fit")
    key = "K" if "K" in blob else "J"
    return {int(layer): t.float() for layer, t in blob[key].items()}


def load(model_id: str, lenses: Iterable[str] = FITTED, *, download: bool = True) -> dict[str, dict[int, Any]]:
    """Every lens of `lenses` that can be had; the others are left out."""
    found: dict[str, dict[int, Any]] = {}
    for lens in lenses:
        if lens == "logit":
            continue
        with contextlib.suppress(ValueError, OSError):
            found[lens] = load_one(model_id, lens, download=download)
    return found


def save(blob: dict[str, Any], path: Path) -> Path:
    """Write a lens file and a `.json` provenance sidecar beside it."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(blob, tmp)
    tmp.replace(path)
    path.with_suffix(".json").write_text(json.dumps(blob["provenance"], indent=1) + "\n")
    return path


# The model's pieces


def decoder_stack(model: Any) -> Any:
    """`model.model` for a causal LM (Qwen, Llama, Mistral, Gemma text), or a multimodal model's language model."""
    inner = getattr(model, "model", model)
    return getattr(inner, "language_model", inner)


def mid_norm(layer: Any) -> Any:
    """The norm the MLP reads through: Gemma's pre-feedforward norm, else the post-attention norm."""
    return getattr(layer, "pre_feedforward_layernorm", None) or layer.post_attention_layernorm


def norm_weight(norm: Any) -> torch.Tensor:
    w: torch.Tensor = norm.weight.detach().float()
    return 1.0 + w if type(norm).__name__ in ZERO_CENTRED else w


def norm_eps(norm: Any) -> float:
    return float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)))


# RelP's rules, adapted from stdlogits/neuron-lens `neuron_lens/rules.py` (MIT), for dense gated decoders


def act_factor(act_fn: Any, g: torch.Tensor) -> torch.Tensor:
    """`φ(g) = act(g)/g` for the identity rule."""
    import torch

    name = type(act_fn).__name__.casefold()
    if name in ("siluactivation", "silu"):
        return torch.sigmoid(g)
    if name in ("gelutanh", "pytorchgelutanh", "gelupytorchtanh"):
        return 0.5 * (1 + torch.tanh(_GELU_C * (g + 0.044715 * g.pow(3))))
    if name == "geluactivation":
        return 0.5 * (1 + torch.erf(g * 0.5**0.5))
    raise ValueError(f"RelP's identity rule has no factor for {type(act_fn).__name__}")


def _norm(self: Any, x: torch.Tensor) -> torch.Tensor:
    import torch

    y: torch.Tensor = self._rules_orig_forward(x)
    xf = x.float()
    scale = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + norm_eps(self)).detach()
    lin = (norm_weight(self).to(xf.device) * xf * scale).to(y.dtype)
    return y.detach() + (lin - lin.detach())


def _scores_norm(self: Any, x: torch.Tensor) -> torch.Tensor:
    """An attention input norm without q/k norms: queries and keys read its true derivative, values the LN rule's."""
    self._scores_in = self._rules_orig_forward(x)
    return _norm(self, x)


def _proj_scores(self: Any, x: torch.Tensor) -> torch.Tensor:
    out: torch.Tensor = self._rules_orig_forward(self._scores_norm._scores_in)
    return out


def _mlp(self: Any, x: torch.Tensor) -> torch.Tensor:
    g, u = self.gate_proj(x), self.up_proj(x)
    act = self.act_fn(g)
    act_lin = g * act_factor(self.act_fn, g).detach()
    da, du = act_lin - act_lin.detach(), u - u.detach()
    prod = (act * u).detach() + HALF_RULE_BETA * (da * u.detach() + act.detach() * du)
    out: torch.Tensor = self.down_proj(prod)
    return out


def check_layer(layer: Any) -> None:
    """Refuse layers the rules here don't cover (mixture-of-experts, DeltaNet, ungated MLPs)."""
    import torch

    if not hasattr(layer, "self_attn") or not hasattr(getattr(layer, "mlp", None), "gate_proj"):
        raise ValueError(f"{type(layer).__name__}: the rules here cover attention + gated dense MLP layers only")
    act_factor(layer.mlp.act_fn, torch.zeros(1))


@contextlib.contextmanager
def relp_rules(model: Any) -> Iterator[None]:
    """Apply RelP's rules to the model's decoder layers for the duration (forward values unchanged)."""
    patched: list[tuple[Any, Any]] = []
    for layer in decoder_stack(model).layers:
        check_layer(layer)
        norms = [getattr(layer, n) for n in RESIDUAL_NORMS if hasattr(layer, n)]
        at = layer.self_attn
        if getattr(at, "q_norm", None) is None:  # queries and keys read the input norm's true derivative
            n1 = layer.input_layernorm
            patched += [(n, _norm) for n in norms if n is not n1] + [(n1, _scores_norm)]
            for proj in (at.q_proj, at.k_proj):
                proj._scores_norm = n1
                patched.append((proj, _proj_scores))
        else:
            patched += [(n, _norm) for n in norms]
        patched.append((layer.mlp, _mlp))
    for module, _ in patched:
        if "forward" in module.__dict__:
            raise ValueError(f"{type(module).__name__} already has an instance-level forward")
    for module, fn in patched:
        module._rules_orig_forward = module.forward
        module.forward = types.MethodType(fn, module)
    try:
        yield
    finally:
        for module, _ in patched:
            del module.forward
            del module._rules_orig_forward
            for name in ("_scores_norm", "_scores_in"):
                module.__dict__.pop(name, None)


# Fitting


@dataclass
class Accumulator:
    """Per-document estimates summed by target (`out`: each layer's output, `mid`: each MLP's input residual) and by
    half (even and odd documents). Rows index the lens's output space, columns its input space."""

    lens: str
    n_layers: int
    d: int
    sums: dict[tuple[str, int], Any] = field(default_factory=dict)
    n_docs: list[int] = field(default_factory=lambda: [0, 0])

    def __post_init__(self) -> None:
        import torch

        for target in ("out", "mid"):
            for half in (0, 1):
                self.sums.setdefault((target, half), torch.zeros(self.n_layers, self.d, self.d))

    def mean(self, target: str = "out", half: int | None = None) -> torch.Tensor:
        import torch

        halves = (0, 1) if half is None else (half,)
        n = sum(self.n_docs[h] for h in halves)
        out: torch.Tensor = sum(self.sums[(target, h)] for h in halves) / max(n, 1)
        if self.lens in ("J", "R") and target == "out":
            out[-1] = torch.eye(self.d)  # the last layer's output is the target itself
        return out

    def split_half(self, target: str = "out") -> list[float]:
        """Per layer, ‖mean_even − mean_odd‖ / ‖mean‖: how differently the two halves of the corpus read."""
        a, b = (self.sums[(target, h)] for h in (0, 1))
        n0, n1 = self.n_docs
        if not (n0 and n1):
            return [float("nan")] * self.n_layers
        whole = (a + b) / (n0 + n1)
        return [float((a[i] / n0 - b[i] / n1).norm() / whole[i].norm().clamp_min(1e-12)) for i in range(self.n_layers)]

    def blob(self, model_id: str, target: str = "out", **provenance: Any) -> dict[str, Any]:
        """The rs-lenses file format."""
        key = "K" if self.lens == "S" else "J"
        mean = self.mean(target)
        return {
            key: {i: mean[i].contiguous() for i in range(self.n_layers)},
            "n_prompts": sum(self.n_docs),
            "d_model": self.d,
            "provenance": {
                "model_id": model_id,
                "lens": self.lens,
                "target": target,
                "estimator": "true derivative" if self.lens == "J" else "relp",
                "skip_first": SKIP_FIRST,
                "split_half": self.split_half(target),
                "code": "steerlab (steerlab/lens/jacobian.py), after stdlogits/neuron-lens",
                **provenance,
            },
        }

    def state(self) -> dict[str, Any]:
        return {"lens": self.lens, "sums": self.sums, "n_docs": self.n_docs}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> Accumulator:
        first = next(iter(state["sums"].values()))
        return cls(state["lens"], first.shape[0], first.shape[1], dict(state["sums"]), list(state["n_docs"]))


def valid_positions(n_tokens: int, skip_first: int = SKIP_FIRST) -> list[int]:
    if n_tokens - 1 <= skip_first:
        raise ValueError(f"{n_tokens} tokens leave no positions after the first {skip_first}")
    return list(range(skip_first, n_tokens - 1))


def _first(output: Any) -> Any:
    return output[0] if isinstance(output, tuple) else output


@contextlib.contextmanager
def _taps(
    model: Any,
    out_fn: Callable[[int, Any], None],
    mid_fn: Callable[[int, Any], None],
    embed_fn: Callable[[Any], Any],
) -> Iterator[None]:
    """Hooks: `embed_fn(out)` replaces the embedding output; `out_fn(i, h)` / `mid_fn(i, x)` see layer i's output
    and its MLP's input residual."""
    stack = decoder_stack(model)
    handles = [stack.embed_tokens.register_forward_hook(lambda m, a, out: embed_fn(out))]
    for i, layer in enumerate(stack.layers):
        handles.append(layer.register_forward_hook(lambda m, a, out, i=i: out_fn(i, _first(out))))
        handles.append(mid_norm(layer).register_forward_pre_hook(lambda m, args, i=i: mid_fn(i, args[0])))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def fit_doc(
    model: Any,
    ids: Sequence[int],
    acc: Accumulator,
    half: int,
    *,
    dim_batch: int = DIM_BATCH,
    skip_first: int = SKIP_FIRST,
    rules: bool | None = None,
) -> None:
    """Add one document's estimate of `acc.lens` (J, R or S) to `acc`. `rules` defaults to RelP for R and S and the
    true derivative for J (S with `rules=False` is the K-lens)."""
    import torch

    use_rules = acc.lens != "J" if rules is None else rules
    valid = torch.tensor(valid_positions(len(ids), skip_first))
    device = next(model.parameters()).device
    base = torch.tensor(list(ids), device=device)[None]
    with relp_rules(model) if use_rules else contextlib.nullcontext():
        for start in range(0, acc.d, dim_batch):
            b = min(dim_batch, acc.d - start)
            (_s_block if acc.lens == "S" else _r_block)(model, base, valid, acc, half, start, b)
    acc.n_docs[half] += 1


def _r_block(model: Any, base: Any, valid: Any, acc: Accumulator, half: int, start: int, b: int) -> None:
    """Reverse mode: seed basis direction start+i of the last layer's output at every valid position of copy i, pull
    back, and average the gradient at every layer's output and MLP input over the valid positions (a row block)."""
    import torch

    outs: list[Any] = [None] * acc.n_layers
    mids: list[Any] = [None] * acc.n_layers

    def keep(store: list[Any], i: int, h: Any) -> None:
        h.retain_grad()
        store[i] = h

    def embed(out: Any) -> Any:
        return out.detach().requires_grad_(True)

    with torch.enable_grad(), _taps(model, lambda i, h: keep(outs, i, h), lambda i, x: keep(mids, i, x), embed):
        decoder_stack(model)(input_ids=base.expand(b, -1), use_cache=False)
        final = outs[-1]
        v = valid.to(final.device)
        rows = torch.arange(b, device=final.device)
        seed = torch.zeros_like(final)
        seed[rows[:, None], v[None, :], (start + rows)[:, None]] = 1
        final.backward(seed)
    for i in range(acc.n_layers):
        if i < acc.n_layers - 1:
            acc.sums[("out", half)][i, start : start + b] += outs[i].grad[:, v].float().mean(1).cpu()
        acc.sums[("mid", half)][i, start : start + b] += mids[i].grad[:, v].float().mean(1).cpu()


def _s_block(model: Any, base: Any, valid: Any, acc: Accumulator, half: int, start: int, b: int) -> None:
    """Forward mode: seed basis direction start+i of the embeddings at every valid position of copy i, push forward,
    and average each layer's tangent over the valid positions (a column block of K_L)."""
    import torch
    import torch.autograd.forward_ad as fwad

    v = valid.to(base.device)

    def embed(out: Any) -> Any:
        tangent = torch.zeros_like(out)
        rows = torch.arange(b, device=out.device)
        tangent[rows[:, None], v[None, :], (start + rows)[:, None]] = 1
        return fwad.make_dual(out, tangent)  # type: ignore[no-untyped-call,unused-ignore]

    def read(target: str, i: int, h: Any) -> None:
        tangent = fwad.unpack_dual(h).tangent  # type: ignore[no-untyped-call,unused-ignore]
        if tangent is not None:
            acc.sums[(target, half)][i, :, start : start + b] += tangent[:, v].float().mean(1).T.cpu()

    taps = _taps(model, lambda i, h: read("out", i, h), lambda i, x: read("mid", i, x), embed)
    with torch.no_grad(), fwad.dual_level(), taps:
        decoder_stack(model)(input_ids=base.expand(b, -1), use_cache=False)


def fit(
    model: Any,
    docs: Iterable[Sequence[int]],
    lens: str,
    *,
    dim_batch: int = DIM_BATCH,
    skip_first: int = SKIP_FIRST,
    acc: Accumulator | None = None,
    on_doc: Callable[[int, Accumulator], None] | None = None,
) -> Accumulator:
    """Fit `lens` over token-id documents (each longer than `skip_first + 1`), documents alternating halves."""
    if lens not in FITTED:
        raise ValueError(f"unknown lens {lens!r}; fitted lenses are {FITTED}")
    stack = decoder_stack(model)
    d = int(stack.embed_tokens.weight.shape[1])
    acc = acc or Accumulator(lens, len(stack.layers), d)
    for i, ids in enumerate(docs):
        fit_doc(model, ids, acc, sum(acc.n_docs) % 2, dim_batch=dim_batch, skip_first=skip_first)
        if on_doc:
            on_doc(i, acc)
    return acc


# Reading


@dataclass
class Reader:
    """Reads vectors through the lenses of one loaded model: `W_U` and the final norm for logit/J/R, the centred
    input embeddings for S, and per-layer statistics for layer-contrastive reads (computed on first use)."""

    model: Any
    lenses: dict[str, dict[int, Any]] = field(default_factory=dict)
    _z: dict[tuple[str, int], Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        stack = decoder_stack(self.model)
        self.n_layers = len(stack.layers)
        self.W_U = self.model.get_output_embeddings().weight.detach().float()
        self.g = norm_weight(stack.norm)
        self.eps = norm_eps(stack.norm)
        E = stack.embed_tokens.weight.detach().float() * float(getattr(stack.embed_tokens, "embed_scale", 1.0))
        self.E = E - E.mean(0, keepdim=True)
        self.layers = stack.layers

    def available(self) -> list[str]:
        return ["logit", *[lens for lens in FITTED if lens in self.lenses]]

    def _check(self, lens: str, layer: int) -> None:
        if lens not in LENSES:
            raise ValueError(f"unknown lens {lens!r}; choose one of {LENSES}")
        if lens != "logit" and lens not in self.lenses:
            raise ValueError(f"the {lens}-lens is not loaded for this model")
        if not 0 <= layer < self.n_layers:
            raise ValueError(f"layer {layer} is outside 0..{self.n_layers - 1}")

    def mapped(self, h: torch.Tensor, layer: int, lens: str) -> torch.Tensor:
        """What the readout sees: `J_L h` for J/R, `K_Lᵀ c` for S (a covector on the embeddings), `h` for logit.
        Works on a vector `[d]` or rows `[n, d]`."""
        self._check(lens, layer)
        h = h.float()
        if lens == "logit":
            return h
        M = self.lenses[lens][layer]
        return h @ M if lens == "S" else h @ M.T  # type: ignore[no-any-return,unused-ignore]

    def logits(self, h: torch.Tensor, layer: int, lens: str) -> torch.Tensor:
        """`[..., vocab]`: logit/J/R apply the final RMSNorm then `W_U`; S scores `c · K_L (W_E[v] − mean)`."""
        import torch

        x = self.mapped(h, layer, lens)
        if lens == "S":
            return x @ self.E.T  # type: ignore[no-any-return,unused-ignore]
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.g
        return x @ self.W_U.T  # type: ignore[no-any-return,unused-ignore]

    def _unit_read(self, x: torch.Tensor, lens: str) -> torch.Tensor:
        y = x if lens == "S" else x * self.g
        return y / y.norm(dim=-1, keepdim=True).clamp_min(1e-12)  # type: ignore[no-any-return,unused-ignore]

    def z_scale(self, layer: int, lens: str) -> torch.Tensor:
        """Per token, the std of its score over the layer's MLP writes ±W_down[:, n], each read through `lens` as a
        unit vector (the mean is 0 by the ± symmetry)."""
        key = (lens, layer)
        if key not in self._z:
            W = self.layers[layer].mlp.down_proj.weight.detach().float().T  # [d_mlp, d]
            X = self._unit_read(self.mapped(W, layer, lens), lens)
            cov = X.T @ X / X.shape[0]
            U = self.E if lens == "S" else self.W_U
            self._z[key] = ((U @ cov) * U).sum(-1).clamp_min(1e-12).sqrt()
        return self._z[key]  # type: ignore[no-any-return,unused-ignore]

    def z_logits(self, h: torch.Tensor, layer: int, lens: str) -> torch.Tensor:
        """Layer-contrastive scores: each token's score on the unit read vector, over its std at this layer."""
        x = self._unit_read(self.mapped(h, layer, lens), lens)
        U = self.E if lens == "S" else self.W_U
        return (x @ U.T) / self.z_scale(layer, lens)  # type: ignore[no-any-return,unused-ignore]

    def top(
        self, h: torch.Tensor, layer: int, lens: str, k: int = 8, *, z: bool = False
    ) -> tuple[list[int], list[float]]:
        import torch

        scores = (self.z_logits if z else self.logits)(h, layer, lens)
        vals, idx = torch.topk(scores, k)
        return [int(i) for i in idx], [float(v) for v in vals]
