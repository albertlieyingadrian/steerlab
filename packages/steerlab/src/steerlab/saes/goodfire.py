"""Goodfire: ReLU residual-stream SAEs for Llama-3.1-8B-Instruct (layer 19) and Llama-3.3-70B-Instruct (layer 50).

File layout (the release's `.pth`, a torch-saved PyTorch module state dict): exactly four tensors,
`encoder_linear.weight` (n_features, d_model), `encoder_linear.bias` (n_features,), `decoder_linear.weight`
(d_model, n_features) and `decoder_linear.bias` (d_model,). Goodfire's reference module encodes with
`relu(encoder_linear(h))` and decodes with `decoder_linear(acts)`, which this module matches exactly. The SAE reads the
output of decoder layer 19 (the residual stream after that layer), the same place steerlab's hook adds directions.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from steerlab import hub
from steerlab.saes.common import LoadedSAE, check_keys, check_shape

if TYPE_CHECKING:
    import torch

DEFAULT_RELEASE = "Goodfire/Llama-3.1-8B-Instruct-SAE-l19"
DEFAULT_LAYER = 19
KEYS = ("encoder_linear.weight", "encoder_linear.bias", "decoder_linear.weight", "decoder_linear.bias")


def filename(release: str = DEFAULT_RELEASE) -> str:
    """The single file in a release: the repo name plus `.pth` (`Llama-3.1-8B-Instruct-SAE-l19.pth`)."""
    return release.rsplit("/", 1)[-1] + ".pth"


class GoodfireSAE(LoadedSAE):
    """A ReLU SAE: a feature fires wherever its pre-activation is positive."""

    def __init__(
        self,
        *,
        W_enc: torch.Tensor,
        b_enc: torch.Tensor,
        W_dec: torch.Tensor,
        b_dec: torch.Tensor,
        layer: int,
        release: str = DEFAULT_RELEASE,
    ) -> None:
        if W_enc.ndim != 2:
            raise ValueError(f"W_enc must be 2-D (n_features, d_model), got shape {tuple(W_enc.shape)}")
        n_features, d_model = W_enc.shape
        check_shape("W_dec", W_dec, (d_model, n_features))
        check_shape("b_enc", b_enc, (n_features,))
        check_shape("b_dec", b_dec, (d_model,))
        self.W_enc = W_enc.float()
        self.b_enc = b_enc.float()
        self.W_dec = W_dec.float()
        self.b_dec = b_dec.float()
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

    def to(self, device: torch.device | str) -> GoodfireSAE:
        for name in ("W_enc", "b_enc", "W_dec", "b_dec"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        return (self._prepare(h) @ self.W_enc.T + self.b_enc).relu()

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return self._prepare(acts) @ self.W_dec.T + self.b_dec

    def decoder_vector(self, feature: int) -> torch.Tensor:
        return self.W_dec[:, feature]

    def encoder_probe(self, feature: int) -> tuple[torch.Tensor, float, float]:
        return self.W_enc[feature], float(self.b_enc[feature]), 0.0


def load_goodfire(path: str | Path, *, layer: int = DEFAULT_LAYER, release: str = DEFAULT_RELEASE) -> GoodfireSAE:
    """Load one Goodfire `.pth` file (read with `weights_only=True`: tensors only, no pickled code runs)."""
    import torch

    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"{path}: expected a dict of tensors, got {type(state).__name__}")
    check_keys(state, KEYS, source=str(path))
    return GoodfireSAE(
        W_enc=state["encoder_linear.weight"],
        b_enc=state["encoder_linear.bias"],
        W_dec=state["decoder_linear.weight"],
        b_dec=state["decoder_linear.bias"],
        layer=layer,
        release=release,
    )


def fetch_goodfire(layer: int = DEFAULT_LAYER, *, release: str = DEFAULT_RELEASE) -> GoodfireSAE:
    """Download (or reuse from the cache) the release's file and load it."""
    return load_goodfire(hub.download(release, filename(release)), layer=layer, release=release)
