"""Qwen-Scope: TopK residual-stream SAEs for Qwen3-1.7B (trained on the BASE model), one `layer{n}.sae.pt` per layer.

File layout (the release's README loader): a torch-saved dict with exactly four tensors,
`W_enc` (n_features, d_model), `W_dec` (d_model, n_features), `b_enc` (n_features,), `b_dec` (d_model,). No k is stored;
it is the "L0_50" of the release name. The reference encoder is `(h @ W_enc.T + b_enc).topk(k)` scattered back into
zeros, with NO ReLU before the top-k, which this module matches exactly. Decoding is `acts @ W_dec.T + b_dec`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from steerlab import defaults, hub
from steerlab.saes.common import LoadedSAE, check_keys, check_shape

if TYPE_CHECKING:
    import torch

DEFAULT_RELEASE = defaults.QWEN_SCOPE_RELEASE
DEFAULT_LAYER = defaults.QWEN_SCOPE_LAYER
KEYS = ("W_enc", "W_dec", "b_enc", "b_dec")


def filename(layer: int) -> str:
    """The per-layer file in the release: `layer14.sae.pt`."""
    return f"layer{layer}.sae.pt"


def k_from_release(release: str) -> int:
    """The TopK k from the release name's `L0_<k>` suffix (the files store no k)."""
    match = re.search(r"L0_(\d+)", release)
    if match is None:
        raise ValueError(f"cannot read k from release name {release!r}: expected an 'L0_<k>' part; pass k= explicitly")
    return int(match.group(1))


class QwenScopeSAE(LoadedSAE):
    """A TopK SAE: exactly k features fire per position."""

    def __init__(
        self,
        *,
        W_enc: torch.Tensor,
        b_enc: torch.Tensor,
        W_dec: torch.Tensor,
        b_dec: torch.Tensor,
        k: int,
        layer: int,
        release: str = DEFAULT_RELEASE,
    ) -> None:
        if W_enc.ndim != 2:
            raise ValueError(f"W_enc must be 2-D (n_features, d_model), got shape {tuple(W_enc.shape)}")
        n_features, d_model = W_enc.shape
        check_shape("W_dec", W_dec, (d_model, n_features))
        check_shape("b_enc", b_enc, (n_features,))
        check_shape("b_dec", b_dec, (d_model,))
        if not 1 <= k <= n_features:
            raise ValueError(f"k={k} must be between 1 and n_features={n_features}")
        self.W_enc = W_enc.float()
        self.b_enc = b_enc.float()
        self.W_dec = W_dec.float()
        self.b_dec = b_dec.float()
        self.k = k
        self.layer = layer
        self.release = release

    @property
    def d_model(self) -> int:
        return int(self.W_enc.shape[1])

    @property
    def n_features(self) -> int:
        return int(self.W_enc.shape[0])

    @property
    def device(self) -> torch.device:
        return self.W_enc.device

    def to(self, device: torch.device | str) -> QwenScopeSAE:
        for name in ("W_enc", "b_enc", "W_dec", "b_dec"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        pre = self._prepare(h) @ self.W_enc.T + self.b_enc
        values, indices = pre.topk(self.k, dim=-1)
        return pre.new_zeros(pre.shape).scatter_(-1, indices, values)

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return self._prepare(acts) @ self.W_dec.T + self.b_dec

    def decoder_vector(self, feature: int) -> torch.Tensor:
        return self.W_dec[:, feature]

    def encoder_probe(self, feature: int) -> tuple[torch.Tensor, float, float]:
        return self.W_enc[feature], float(self.b_enc[feature]), 0.0


def load_qwen_scope(
    path: str | Path, *, layer: int, release: str = DEFAULT_RELEASE, k: int | None = None
) -> QwenScopeSAE:
    """Load one `layer{n}.sae.pt` file. `k` defaults to the release name's L0."""
    import torch

    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"{path}: expected a dict of tensors, got {type(state).__name__}")
    check_keys(state, KEYS, source=str(path))
    return QwenScopeSAE(
        W_enc=state["W_enc"],
        b_enc=state["b_enc"],
        W_dec=state["W_dec"],
        b_dec=state["b_dec"],
        k=k if k is not None else k_from_release(release),
        layer=layer,
        release=release,
    )


def fetch_qwen_scope(
    layer: int = DEFAULT_LAYER, *, release: str = DEFAULT_RELEASE, k: int | None = None
) -> QwenScopeSAE:
    """Download (or reuse from the cache) one layer of the release and load it."""
    return load_qwen_scope(hub.download(release, filename(layer)), layer=layer, release=release, k=k)
