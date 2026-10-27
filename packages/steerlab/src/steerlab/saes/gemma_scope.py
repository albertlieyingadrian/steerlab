"""Gemma Scope 2: JumpReLU residual-stream SAEs for gemma-3-1b-it, one folder per layer with `params.safetensors`.

Folder: `resid_post/layer_{n}_width_16k_l0_medium/` (plus a `config.json` saying `"architecture": "jump_relu"`).
Tensors, in the Gemma Scope convention: `w_enc` (d_model, n_features), `w_dec` (n_features, d_model), `b_enc`
(n_features,), `b_dec` (d_model,), `threshold` (n_features,). Encoding is `pre = h @ w_enc + b_enc` and
`acts = pre * (pre > threshold)`: a feature fires only when its pre-activation clears its own learned threshold.
Decoding is `acts @ w_dec + b_dec`.

The original Gemma Scope releases for Gemma 2 (`google/gemma-scope-2b-pt-res`, `google/gemma-scope-9b-it-res`) store
the same JumpReLU SAE as `layer_{n}/width_16k/average_l0_{k}/params.npz` with capitalised `W_enc` / `W_dec`.
`load_gemma_scope` reads either file; `fetch_gemma_scope` finds a v1 file in `defaults.EXTRA_SAES`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from steerlab import defaults, hub
from steerlab.saes.common import LoadedSAE, check_keys, check_shape

if TYPE_CHECKING:
    import torch

DEFAULT_RELEASE = defaults.GEMMA_SCOPE_RELEASE
DEFAULT_LAYER = defaults.GEMMA_SCOPE_LAYER
DEFAULT_WIDTH = "16k"
DEFAULT_L0 = "medium"
KEYS = ("w_enc", "w_dec", "b_enc", "b_dec", "threshold")
KEYS_V1 = ("W_enc", "W_dec", "b_enc", "b_dec", "threshold")


def params_path(layer: int, *, width: str = DEFAULT_WIDTH, l0: str = DEFAULT_L0, hook: str = "resid_post") -> str:
    """The per-layer file in the release: `resid_post/layer_13_width_16k_l0_medium/params.safetensors`."""
    return f"{hook}/layer_{layer}_width_{width}_l0_{l0}/params.safetensors"


class GemmaScopeSAE(LoadedSAE):
    """A JumpReLU SAE: each feature has its own firing threshold."""

    def __init__(
        self,
        *,
        w_enc: torch.Tensor,
        w_dec: torch.Tensor,
        b_enc: torch.Tensor,
        b_dec: torch.Tensor,
        threshold: torch.Tensor,
        layer: int,
        release: str = DEFAULT_RELEASE,
    ) -> None:
        if w_enc.ndim != 2:
            raise ValueError(f"w_enc must be 2-D (d_model, n_features), got shape {tuple(w_enc.shape)}")
        d_model, n_features = w_enc.shape
        check_shape("w_dec", w_dec, (n_features, d_model))
        check_shape("b_enc", b_enc, (n_features,))
        check_shape("b_dec", b_dec, (d_model,))
        check_shape("threshold", threshold, (n_features,))
        self.w_enc = w_enc.float()
        self.w_dec = w_dec.float()
        self.b_enc = b_enc.float()
        self.b_dec = b_dec.float()
        self.threshold = threshold.float()
        self.layer = layer
        self.release = release

    @property
    def d_model(self) -> int:
        return int(self.w_enc.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.w_enc.shape[1])

    @property
    def device(self) -> torch.device:
        return self.w_enc.device

    def to(self, device: torch.device | str) -> GemmaScopeSAE:
        for name in ("w_enc", "w_dec", "b_enc", "b_dec", "threshold"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        pre = self._prepare(h) @ self.w_enc + self.b_enc
        return pre * (pre > self.threshold).to(pre.dtype)

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return self._prepare(acts) @ self.w_dec + self.b_dec

    def decoder_vector(self, feature: int) -> torch.Tensor:
        return self.w_dec[feature]

    def encoder_probe(self, feature: int) -> tuple[torch.Tensor, float, float]:
        return self.w_enc[:, feature], float(self.b_enc[feature]), float(self.threshold[feature])


def load_gemma_scope(path: str | Path, *, layer: int, release: str = DEFAULT_RELEASE) -> GemmaScopeSAE:
    """Load one `params.safetensors` (Gemma Scope 2) or `params.npz` (Gemma Scope for Gemma 2) file."""
    if str(path).endswith(".npz"):
        import numpy as np
        import torch

        with np.load(str(path)) as npz:
            raw = {key: torch.from_numpy(np.asarray(npz[key])) for key in npz.files}
        check_keys(raw, KEYS_V1, source=str(path))
        state = {key.lower(): value for key, value in raw.items()}
    else:
        from safetensors.torch import load_file

        state = load_file(str(path))
        check_keys(state, KEYS, source=str(path))
    return GemmaScopeSAE(
        w_enc=state["w_enc"],
        w_dec=state["w_dec"],
        b_enc=state["b_enc"],
        b_dec=state["b_dec"],
        threshold=state["threshold"],
        layer=layer,
        release=release,
    )


def fetch_gemma_scope(
    layer: int = DEFAULT_LAYER,
    *,
    release: str = DEFAULT_RELEASE,
    width: str = DEFAULT_WIDTH,
    l0: str = DEFAULT_L0,
) -> GemmaScopeSAE:
    """Download (or reuse from the cache) one layer of the release and load it. Refuses if the repo is gated to you."""
    if release == DEFAULT_RELEASE or "gemma-scope-2-" in release:  # "gemma-scope-2b-pt-res" is v1
        path = hub.download(release, params_path(layer, width=width, l0=l0))
        return load_gemma_scope(path, layer=layer, release=release)
    known = [sae for sae in defaults.EXTRA_SAES.values() if sae.release == release]
    match = next((sae for sae in known if sae.layer == layer), None)
    if match is None:
        layers = ", ".join(str(sae.layer) for sae in known) or "none"
        raise ValueError(f"no known file for {release} layer {layer} (known layers: {layers}); pass a file instead")
    path = hub.download(release, match.file)
    return load_gemma_scope(path, layer=layer, release=release)
