"""Scalar readouts of the model's next-token prediction, and the KL divergence between two predictions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


def log_softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    m = x.max(axis=-1, keepdims=True)
    out: np.ndarray = x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))
    return out


def kl(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    """Mean over positions of KL(p || q), where each row of logits is one position's prediction."""
    lp = log_softmax(np.atleast_2d(p_logits))
    lq = log_softmax(np.atleast_2d(q_logits))
    return float((np.exp(lp) * (lp - lq)).sum(axis=-1).mean())


@dataclass(frozen=True)
class Metric:
    """A named function of the last position's logits, e.g. logit(' Austin') - logit(' Dallas')."""

    name: str
    fn: Callable[[np.ndarray], float]

    def __call__(self, last_logits: np.ndarray) -> float:
        return float(self.fn(np.asarray(last_logits, dtype=np.float64)))


def logit_diff(target: int, contrast: int, target_str: str = "", contrast_str: str = "") -> Metric:
    t = target_str or str(target)
    c = contrast_str or str(contrast)
    return Metric(f"logit({t!r}) - logit({c!r})", lambda x: float(x[target] - x[contrast]))


def logprob(target: int, target_str: str = "") -> Metric:
    t = target_str or str(target)
    return Metric(f"log p({t!r})", lambda x: float(log_softmax(x)[target]))


def logit(target: int, target_str: str = "") -> Metric:
    t = target_str or str(target)
    return Metric(f"logit({t!r})", lambda x: float(x[target]))
