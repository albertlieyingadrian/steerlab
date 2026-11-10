"""deslop: measure and remove chat-model writing habits.

    from deslop import score
    score("It's not a tool, it's a mindset.").score

The scorer, checks and CLI need nothing beyond the standard library. The sampler and the steering and neuron-edit
hooks (`deslop.sampler`, `deslop.steer`) need the `models` extra (torch and transformers).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from deslop.habits import HABIT_LABELS, WEIGHTS, Span, find_spans
from deslop.score import Score, score

try:
    __version__ = version("steerlab-deslop")
except PackageNotFoundError:  # running from a source checkout without installing
    __version__ = "0.0.0"

__all__ = ["HABIT_LABELS", "WEIGHTS", "Score", "Span", "__version__", "find_spans", "score"]
