"""Paths, keys and the statistics every script shares: AUROC, F1 at a threshold, and bootstrap intervals that resample
whole prompts (a prompt's videos from different models share a scene, so they are not independent)."""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DATA = Path(os.environ.get("PHYS_DIR", Path.home() / ".cache" / "steerlab" / "physics-judge"))
RESULTS.mkdir(exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

# The test split of VideoPhy-2 (MIT licence) and the authors' released AutoEval predictions on it.
TEST_CSV = "https://huggingface.co/datasets/videophysics/videophy2_test/resolve/main/videophy2_test.csv"
AUTO_CSV = (
    "https://huggingface.co/datasets/videophysics/videophy2_autoeval_scores/resolve/main/videophy2_autoeval_scores.csv"
)
# Laws with enough rules in the sample to score on their own; every other law is pooled as "Other".
LAWS = ["Gravity", "Elasticity", "Friction", "Conservation of Momentum", "Conservation of Mass", "Hardness"]


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


def auroc(y, s) -> float:
    """Area under the ROC curve, ties counted as half (Mann-Whitney). y: 1 = violation; higher s = more violating."""
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    from scipy.stats import rankdata

    r = rankdata(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def f1_acc(y, pred) -> tuple[float, float]:
    y = np.asarray(y, int)
    pred = np.asarray(pred, int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return f1, float((pred == y).mean())


def best_threshold(y, s) -> float:
    """The cut that maximises balanced accuracy on a calibration set (scores at or above the cut = violation)."""
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    best, cut = -1.0, float(np.median(s))
    for c in np.unique(s):
        p = s >= c
        bal = (p[y == 1].mean() + (~p[y == 0]).mean()) / 2
        if bal > best:
            best, cut = bal, float(c)
    return cut


def group_boot(stat, groups, boot: int = 2000, seed: int = 0):
    """Point value and 95% interval of stat(index_array), resampling whole groups (prompts) with replacement."""
    groups = np.asarray(groups)
    members = [np.flatnonzero(groups == k) for k in np.unique(groups)]
    rng = np.random.default_rng(seed)
    point = stat(np.arange(len(groups)))
    draws = []
    for _ in range(boot):
        pick = rng.integers(0, len(members), len(members))
        v = stat(np.concatenate([members[k] for k in pick]))
        if np.isfinite(v):
            draws.append(v)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def wilson(k: int, n: int) -> tuple[float, float]:
    """95% Wilson interval for k of n."""
    if n == 0:
        return float("nan"), float("nan")
    z = 1.959964
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)
