"""Paths and the statistics the scripts share: Bradley-Terry scores on the Elo scale, bootstrap intervals, and the
leave-one-voter-out agreement used for every scorer. Copied from experiments/image-to-3d-eval (common.py and
5_analyse.py); `ceiling` and `loo` there assumed five voters, here they take any number."""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DATA = Path(os.environ.get("I2V_DIR", Path.home() / ".cache" / "steerlab" / "image-to-video"))
RESULTS.mkdir(exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

ELO = 400 / np.log(10)  # one Bradley-Terry unit in Elo points


def env_key(name: str) -> str:
    """An API key from the environment or the nearest .env file; never printed."""
    if os.environ.get(name):
        return os.environ[name]
    for parent in [HERE, *HERE.parents]:
        env = parent / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                m = re.match(rf"\s*{name}\s*=\s*(.*)", line)
                if m:
                    return m.group(1).strip().strip("\"'")
    raise SystemExit(f"{name} not found")


def _fit(n: int, i: np.ndarray, j: np.ndarray, wi: np.ndarray, wj: np.ndarray) -> np.ndarray:
    """Bradley-Terry strengths by Hunter's MM updates. wi, wj: (fractional) wins of i over j and of j over i."""
    p = np.ones(n)
    # A prior of half a win and half a loss against a strength-1 opponent keeps unbeaten models finite.
    wins = np.bincount(i, wi, n) + np.bincount(j, wj, n) + 0.5
    games = wi + wj
    for _ in range(1000):
        denom = games / (p[i] + p[j])
        new = wins / (np.bincount(i, denom, n) + np.bincount(j, denom, n) + 1.0 / (p + 1))
        new /= np.exp(np.log(new).mean())
        done = np.abs(new - p).max() < 1e-9
        p = new
        if done:
            break
    theta = np.log(p)
    return 1000 + ELO * (theta - theta.mean())


def bradley_terry(battles, boot: int = 1000, seed: int = 0):
    """battles: (model_a, model_b, weight_a, weight_b) with weights such as votes or 1/0/0.5 verdicts.
    Returns {model: Elo-scale score} (mean 1000) and {model: (2.5%, 97.5%)} from resampling the battles."""
    models = sorted({b[0] for b in battles} | {b[1] for b in battles})
    index = {m: k for k, m in enumerate(models)}
    i = np.array([index[b[0]] for b in battles], dtype=int)
    j = np.array([index[b[1]] for b in battles], dtype=int)
    wa = np.array([b[2] for b in battles], float)
    wb = np.array([b[3] for b in battles], float)
    total = wa + wb
    keep = total > 0
    i, j, wa, wb = i[keep], j[keep], wa[keep] / total[keep], wb[keep] / total[keep]
    point = _fit(len(models), i, j, wa, wb)
    rng = np.random.default_rng(seed)
    draws = [point]
    for _ in range(boot):
        s = rng.integers(0, len(i), len(i))
        draws.append(_fit(len(models), i[s], j[s], wa[s], wb[s]))
    lo, hi = np.percentile(np.array(draws[1:] or draws), [2.5, 97.5], axis=0)
    return {m: float(point[k]) for m, k in index.items()}, {m: (float(lo[k]), float(hi[k])) for m, k in index.items()}


def boot_mean(x, boot: int = 2000, seed: int = 0, groups=None) -> tuple[float, float, float]:
    """Mean and 95% bootstrap interval; with groups, resample whole groups (such as objects) at once."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    if groups is None:
        draws = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(boot)]
    else:
        groups = np.asarray(groups)
        members = [np.flatnonzero(groups == k) for k in np.unique(groups)]
        draws = []
        for _ in range(boot):
            pick = rng.integers(0, len(members), len(members))
            draws.append(x[np.concatenate([members[k] for k in pick])].mean())
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def ceiling(a: int, b: int) -> float:
    """Expected agreement of one voter with the majority of the others (a tied remainder counts as half).
    a, b: votes for side A and side B."""

    def agree(mine_a: bool, oa: int, ob: int) -> float:
        return 0.5 if oa == ob else float((oa > ob) == mine_a)

    return (a * agree(True, a - 1, b) + b * agree(False, a, b - 1)) / (a + b)


def loo(a: int, b: int, v: float) -> float:
    """Expected agreement of a verdict v (1 = side A, 0 = side B, in between = split) with the majority of the voters
    left after removing one, averaged over which voter is removed: the target the one-voter ceiling is scored on."""

    def agree(oa: int, ob: int) -> float:
        return 0.5 if oa == ob else (v if oa > ob else 1 - v)

    return (a * agree(a - 1, b) + b * agree(a, b - 1)) / (a + b)
