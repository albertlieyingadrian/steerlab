"""
Pairwise distances per feature space, diversity with a domain bootstrap, and the slop number (plan: Measures).

A "space" maps a list of feature records to an n x n distance matrix:
  siglip, uiclip   1 - cosine of the image embeddings
  layout, dom      1 - cosine of the occupancy grid / WL histogram
  type             1 - cosine of the font-family area histogram
  palette          weighted Chamfer distance between k-means palettes (ΔE76)
  spacing          Euclidean distance between z-scored spacing vectors (z from the pooled reference set)

Diversity of a set = mean distance over pairs from different domains. The bootstrap resamples domains with
replacement (the same draw for every year), so year-to-year comparisons share their noise.
"""

from __future__ import annotations

import numpy as np

SPACES = ["siglip", "uiclip", "layout", "dom", "type", "palette", "spacing"]
HAND = ["layout", "dom", "type", "palette", "spacing"]


def _cos_dist(M: np.ndarray) -> np.ndarray:
    M = M.astype(np.float32)
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1.0
    M = M / n
    return np.clip(1.0 - M @ M.T, 0.0, 2.0)


def type_matrix(recs, vocab=None):
    if vocab is None:
        vocab = sorted({k for r in recs for k in r["type_hist"]})
    ix = {k: i for i, k in enumerate(vocab)}
    M = np.zeros((len(recs), len(vocab)), np.float32)
    for i, r in enumerate(recs):
        for k, v in r["type_hist"].items():
            if k in ix:
                M[i, ix[k]] = v
    return M


def palette_matrix(recs) -> np.ndarray:
    C = np.stack([r["pal_c"] for r in recs]).astype(np.float32)  # n,5,3
    W = np.stack([r["pal_w"] for r in recs]).astype(np.float32)  # n,5
    n = len(recs)
    D = np.zeros((n, n), np.float32)
    for i in range(n):
        d = np.linalg.norm(
            C[i][None, :, None, :] - C[:, None, :, :], axis=-1
        )  # n,5(i),5(j)
        a = (W[i][None, :] * d.min(2)).sum(1)  # i's colours to nearest in j
        b = (W * d.min(1)).sum(1)  # j's colours to nearest in i
        D[i] = 0.5 * (a + b)
    return D


def distance_matrix(recs, space: str, spacing_stats=None) -> np.ndarray:
    if space in ("siglip", "uiclip", "layout", "dom"):
        return _cos_dist(np.stack([np.asarray(r[space], np.float32) for r in recs]))
    if space == "type":
        return _cos_dist(type_matrix(recs))
    if space == "palette":
        return palette_matrix(recs)
    if space == "spacing":
        X = np.stack([r["spacing"] for r in recs]).astype(np.float32)
        mu, sd = (
            spacing_stats
            if spacing_stats is not None
            else (np.nanmean(X, 0), np.nanstd(X, 0) + 1e-6)
        )
        Z = np.nan_to_num((X - mu) / sd)
        Z = np.clip(Z, -4, 4)  # one extreme page should not dominate a set's spread
        sq = (Z**2).sum(1)
        return np.sqrt(np.clip(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0, None))
    raise ValueError(space)


def boot_weights(
    domains: list[str], B: int, seed: int = 0
) -> tuple[list[str], np.ndarray]:
    """Domain list and a (B+1) x n_domains count matrix; row 0 is the observed sample (all ones)."""
    uniq = sorted(set(domains))
    rng = np.random.default_rng(seed)
    W = np.ones((B + 1, len(uniq)), np.float32)
    for b in range(1, B + 1):
        W[b] = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
    return uniq, W


def mean_pair_distance(
    D: np.ndarray, page_domains: list[str], uniq: list[str], W: np.ndarray
) -> np.ndarray:
    """Mean distance over pairs of pages from different domains, for each bootstrap row of W. Returns (B+1,)."""
    ix = {d: i for i, d in enumerate(uniq)}
    cols = np.array([ix[d] for d in page_domains])
    w = W[:, cols]  # (B+1) x n pages
    same = np.array(page_domains)[:, None] == np.array(page_domains)[None, :]
    Dm = np.where(same, 0.0, D).astype(np.float32)
    num = ((w @ Dm) * w).sum(1)
    den = ((w @ (1.0 - same.astype(np.float32))) * w).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def ci(x: np.ndarray, lo: float = 2.5, hi: float = 97.5):
    x = x[np.isfinite(x)]
    return (
        (float(np.percentile(x, lo)), float(np.percentile(x, hi)))
        if len(x)
        else (float("nan"), float("nan"))
    )


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score of a random positive > score of a random negative), ties count half."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1
    # average ranks for ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def two_segment_breakpoint(years: np.ndarray, vals: np.ndarray, candidates) -> int:
    """Year at which a continuous two-segment linear fit has the lowest squared error."""
    best, best_err = None, np.inf
    for k in candidates:
        X = np.stack(
            [np.ones_like(years, float), years - k, np.clip(years - k, 0, None)], 1
        )
        coef, *_ = np.linalg.lstsq(X, vals, rcond=None)
        err = ((X @ coef - vals) ** 2).sum()
        if err < best_err:
            best, best_err = k, err
    return int(best)


def slope(years, vals) -> float:
    years, vals = np.asarray(years, float), np.asarray(vals, float)
    ok = np.isfinite(vals)
    return float(np.polyfit(years[ok], vals[ok], 1)[0])
