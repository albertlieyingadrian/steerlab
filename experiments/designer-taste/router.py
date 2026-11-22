"""
The router for The Taste Stack Episode 7: per part, the cheapest judge that is good enough.

Rule (fixed in the plan): on the training fold, estimate each tier's agreement with the designers (zero-shot tiers
directly; the probe by 5-fold CV inside the training fold); choose the cheapest tier within EPS = 0.01 of the best
tier on that part (formal only where a check is defined for the part). Apply the choice to the held-out fold.

Tiers: formal, probe, small (Qwen2.5-VL-7B), mid (Gemma-3-27B), large (Qwen2.5-VL-72B-AWQ; the frontier stand-in after
OpenRouter refused payment). Cost order is measured, not assumed.
Inputs: a long frame with one row per (designer, part, scene, a, b, y, test fold) and one column per tier holding
P(a beats b) on the held-out fold, plus "<tier>_train" columns holding the training-fold estimate inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 0.01
TIERS = ["formal", "probe", "small", "mid", "large"]


def hit(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.where(np.abs(p - 0.5) < 1e-9, 0.5, (p > 0.5) == (y == 1)).astype(float)


def choose(train: pd.DataFrame, cost: dict, available: dict, eps: float = EPS, tiers: list | None = None) -> dict:
    """train: rows of the training fold with columns <tier> (P on training-fold pairs). -> {part: tier}."""
    out = {}
    for part, g in train.groupby("part"):
        acc = {
            t: float(hit(g[t].to_numpy(), g.y.to_numpy()).mean())
            for t in (tiers or TIERS)
            if t in g and available[part].get(t, False)
        }
        best = max(acc.values())
        ok = [t for t in acc if acc[t] >= best - eps]
        out[part] = min(ok, key=lambda t: cost[t])
    return out


def evaluate(test: pd.DataFrame, choice: dict, cost: dict) -> dict:
    ps = test.apply(lambda r: r[choice[r.part]], axis=1).to_numpy()
    c = test.part.map(lambda p: cost[choice[p]]).to_numpy()
    return {"p": ps, "cost": c}
