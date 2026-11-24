"""Per-rule statistics with a group bootstrap: firing rates, balanced precision, recall, over-representation ratio.

Balanced precision BP = a / (a + h), where a and h are the shares of AI and human windows the rule fires on: the
chance a firing window is AI when human and AI windows are equally common (0.5 = no better than a coin).
Over-representation ratio (Antislop): (matches per word in AI + smoothing) / (matches per word in human + smoothing).
Intervals: 95% percentile bootstrap that resamples prompt groups, so a human text and the replies to its prompt move
together.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from common import SEED

N_BOOT = 1000
KEEP_LO, NOISE_LO, MIN_AI_RATE = 0.60, 0.55, 0.005
SMOOTH = 0.5


def verdict(bp_lo: float, a: float) -> str:
    """keep: BP lower bound > 0.60 and fires on >= 0.5% of AI windows; noise: lower bound <= 0.55 (or never fires on
    AI); rare: fires on < 0.5% of AI windows; weak: the rest."""
    if a < MIN_AI_RATE and (not np.isfinite(bp_lo) or bp_lo > NOISE_LO):
        return "rare"
    if not np.isfinite(bp_lo) or bp_lo <= NOISE_LO:
        return "noise"
    return "keep" if bp_lo > KEEP_LO else "weak"


def _group_sums(counts: np.ndarray, words: np.ndarray, gidx: np.ndarray, n_groups: int, mask: np.ndarray):
    fires = (counts[mask] > 0).astype(np.float64)
    g = gidx[mask]
    n = np.bincount(g, minlength=n_groups).astype(np.float64)
    f = np.zeros((n_groups, counts.shape[1]))
    c = np.zeros((n_groups, counts.shape[1]))
    np.add.at(f, g, fires)
    np.add.at(c, g, counts[mask].astype(np.float64))
    w = np.bincount(g, weights=words[mask].astype(np.float64), minlength=n_groups)
    return n, f, c, w


def rule_table(
    counts: np.ndarray,
    words: np.ndarray,
    is_ai: np.ndarray,
    groups: np.ndarray,
    rids: list[str],
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> pd.DataFrame:
    """One row per rule. `counts` is windows x rules; `is_ai` marks AI windows; `groups` the prompt group per window."""
    uniq, gidx = np.unique(groups, return_inverse=True)
    G = len(uniq)
    nh, fh, ch, wh = _group_sums(counts, words, gidx, G, ~is_ai)
    na, fa, ca, wa = _group_sums(counts, words, gidx, G, is_ai)

    def stats(weight: np.ndarray):
        h = (weight @ fh) / max(weight @ nh, 1)
        a = (weight @ fa) / max(weight @ na, 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            bp = a / (a + h)
        ratio = ((weight @ ca) + SMOOTH) / max(weight @ wa, 1) / (((weight @ ch) + SMOOTH) / max(weight @ wh, 1))
        return h, a, bp, ratio

    h, a, bp, ratio = stats(np.ones(G))
    rng = np.random.default_rng(seed)
    W = rng.multinomial(G, np.full(G, 1 / G), size=n_boot).astype(np.float64)
    boot_bp = np.empty((n_boot, len(rids)))
    boot_ratio = np.empty((n_boot, len(rids)))
    for b in range(n_boot):
        _, _, boot_bp[b], boot_ratio[b] = stats(W[b])
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bp_lo, bp_hi = np.nanpercentile(boot_bp, [2.5, 97.5], axis=0)
    r_lo, r_hi = np.percentile(np.log(boot_ratio), [2.5, 97.5], axis=0)
    df = pd.DataFrame(
        {
            "rule": rids,
            "n_h": int(nh.sum()),
            "n_a": int(na.sum()),
            "h": h,
            "a": a,
            "bp": bp,
            "bp_lo": bp_lo,
            "bp_hi": bp_hi,
            "ratio": ratio,
            "ratio_lo": np.exp(r_lo),
            "ratio_hi": np.exp(r_hi),
        }
    )
    df["verdict"] = [verdict(lo, ai) for lo, ai in zip(df.bp_lo, df.a, strict=True)]
    return df


def auroc(scores: np.ndarray, is_ai: np.ndarray) -> float:
    """Probability a random AI window scores above a random human one (ties count half)."""
    from scipy.stats import rankdata

    r = rankdata(scores)
    n1, n0 = is_ai.sum(), (~is_ai).sum()
    return float((r[is_ai].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auroc_ci(scores: np.ndarray, is_ai: np.ndarray, groups: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED):
    uniq, gidx = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(gidx == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))])
        if is_ai[idx].all() or (~is_ai[idx]).all():
            continue
        vals.append(auroc(scores[idx], is_ai[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return auroc(scores, is_ai), float(lo), float(hi)


def rate_ci(flags: np.ndarray, groups: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED):
    """Share of windows flagged, with a group-bootstrap interval."""
    uniq, gidx = np.unique(groups, return_inverse=True)
    n = np.bincount(gidx).astype(float)
    k = np.bincount(gidx, weights=flags.astype(float), minlength=len(uniq))
    rng = np.random.default_rng(seed)
    W = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=n_boot)
    vals = (W @ k) / (W @ n)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(k.sum() / n.sum()), float(lo), float(hi)
