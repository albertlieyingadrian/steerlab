"""What a model must offer to be validated. circuit-tracer's ReplacementModel is wrapped in `tracer.py`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from circuitkit.validate.types import Intervention

# (layer, position, feature index, activation)
ActiveFeature = tuple[int, int, int, float]


class InterventionBackend(Protocol):
    """The calls the checks need. Everything crosses this boundary as NumPy or plain Python."""

    def tokens(self, prompt: str) -> list[str]:
        """The prompt's tokens as strings, in the order positions count them (special tokens included)."""
        ...

    def logits(self, prompt: str, interventions: Sequence[Intervention] = ()) -> np.ndarray:
        """Logits at every position, shape (positions, vocab), with the features set to the given values."""
        ...

    def active_features(self, prompt: str) -> list[ActiveFeature]:
        """Every non-zero transcoder feature on the prompt, at every position."""
        ...

    def decoder_norm(self, layer: int, feature: int) -> float:
        """Norm of the feature's decoder (all output layers together for a cross-layer transcoder)."""
        ...

    def token_id(self, token: str) -> int:
        """Vocabulary index of a single token, e.g. ' Austin'."""
        ...

    def describe(self) -> dict[str, str]:
        """A few strings for the report: model, transcoders, device."""
        ...
