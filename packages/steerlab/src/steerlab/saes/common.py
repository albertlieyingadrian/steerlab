"""What every SAE loader produces: a `LoadedSAE` that encodes, decodes and hands out feature directions.

An SAE (sparse autoencoder) is a dictionary of directions in the residual stream. Think of it as a big box of labelled
arrows: `encode` says how much of each arrow is in a residual vector (most entries zero), `decode` adds the arrows back
up, and `direction(f)` hands you arrow number f as a unit vector, which is what the steering hook adds to the stream.

All math runs in float32 on the SAE's device, whatever dtype the model's activations arrive in.

torch is only needed for type annotations here (the tensor methods used do not need the module), so importing this
module stays torch-free and `steerlab doctor` can run on the base install.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class FeatureDirection:
    """One steerable direction at one layer.

    `unit` is a float32 unit vector of shape (d_model,). `release` names where it came from (an SAE release, or a
    direction source such as sandbagkit's difference-of-means recipe). `label` is a human-readable name when one is
    known. `scale` is the calibration constant that turns strength 1.0 into a raw added norm; None until
    calibrated.
    """

    unit: torch.Tensor
    layer: int
    release: str
    feature: int
    label: str | None = None
    scale: float | None = None

    def __post_init__(self) -> None:
        if self.unit.ndim != 1:
            raise ValueError(f"unit must be a 1-D vector, got shape {tuple(self.unit.shape)}")

    @property
    def d_model(self) -> int:
        return int(self.unit.shape[0])


class LoadedSAE(ABC):
    """One SAE layer in memory. Subclasses hold the weights and implement the three primitives."""

    release: str
    layer: int
    hook: str = "resid_post"

    @property
    @abstractmethod
    def d_model(self) -> int: ...

    @property
    @abstractmethod
    def n_features(self) -> int: ...

    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    @abstractmethod
    def encode(self, h: torch.Tensor) -> torch.Tensor:
        """Residual vectors (..., d_model) -> sparse feature activations (..., n_features), float32."""

    @abstractmethod
    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        """Feature activations (..., n_features) -> reconstructed residual vectors (..., d_model), float32."""

    @abstractmethod
    def decoder_vector(self, feature: int) -> torch.Tensor:
        """The raw (un-normalised) decoder vector of one feature, shape (d_model,)."""

    def encoder_probe(self, feature: int) -> tuple[torch.Tensor, float, float]:
        """`(row, bias, threshold)` for one feature: its activation on a residual vector h is `h @ row + bias` when
        that is above `threshold`, else 0. Clamping a feature (control mode "clamp") reads it every token.

        A TopK SAE has no fixed threshold (its cut is "the k largest this token"), so it reports 0: the ReLU reading.
        """
        name = type(self).__name__
        raise NotImplementedError(f"{name} does not expose its encoder, so its features cannot be clamped")

    def reconstruct(self, h: torch.Tensor) -> torch.Tensor:
        """decode(encode(h)): how well this SAE explains the residual vectors it is given."""
        return self.decode(self.encode(h))

    def direction(self, feature: int, *, label: str | None = None) -> FeatureDirection:
        """Feature `feature` as a unit vector, ready for the steering hook."""
        if not 0 <= feature < self.n_features:
            raise IndexError(f"feature {feature} out of range for an SAE with {self.n_features} features")
        raw = self.decoder_vector(feature).float()
        return FeatureDirection(
            unit=raw / raw.norm(), layer=self.layer, release=self.release, feature=feature, label=label
        )

    def _prepare(self, h: torch.Tensor) -> torch.Tensor:
        """Move an input onto the SAE's device in float32."""
        return h.to(self.device).float()


def check_shape(name: str, tensor: torch.Tensor, expected: tuple[int, ...]) -> None:
    if tuple(tensor.shape) != expected:
        raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected {expected}")


def check_keys(state: dict[str, torch.Tensor], required: tuple[str, ...], *, source: str) -> None:
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(
            f"{source}: missing {', '.join(missing)}; expected the keys {list(required)}, found {sorted(state)}"
        )
