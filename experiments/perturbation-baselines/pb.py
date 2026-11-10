"""Shared pieces for the perturbation-baselines note: condition names, the baselines, the metrics and the bootstrap.

Adapted from the author's `pertkit` toolkit (baselines, metrics, stats), trimmed to what this note uses and made to
work on plain numpy arrays, so the scoring step needs nothing but numpy.

Every baseline is a sentence about biology, not a model:

| baseline          | the sentence                                                                   |
|-------------------|--------------------------------------------------------------------------------|
| `no_change`       | "the perturbation did nothing": predict the control profile                    |
| `additive`        | "two perturbations add": control + (A - control) + (B - control)               |
| `mean_response`   | "every perturbation does the same thing": control + the average training shift |
| `ridge_embedding` | "similar genes behave similarly": a linear map from a gene vector to its shift  |
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CONTROL = "ctrl"
SEEDS = (1, 2, 3, 4, 5)
TOP_GENES = 1000
COMPONENTS = 20


def canonical_condition(name: str) -> str:
    """Sort the genes in a condition name, so `B+A` and `A+B` are one condition and a single is `GENE+ctrl`."""
    parts = [part for part in str(name).split("+") if part]
    if not parts:
        raise ValueError("empty condition name")
    if parts == [CONTROL]:
        return CONTROL
    genes = sorted(part for part in parts if part != CONTROL)
    return f"{genes[0]}+{CONTROL}" if len(genes) == 1 else "+".join(genes)


def perturbed_genes(condition: str) -> list[str]:
    """The genes a condition perturbs: [] for the control, one for a single, two for a pair."""
    return [part for part in canonical_condition(condition).split("+") if part != CONTROL]


def make_splits(doubles, seeds=SEEDS, test_fraction: float = 0.5) -> dict[int, dict[str, list[str]]]:
    """Every single trains; half the pairs train and half are held out, five times with different seeds."""
    ordered = sorted(doubles)
    splits = {}
    for seed in seeds:
        shuffled = list(np.random.default_rng(seed).permutation(ordered))
        cut = round(len(ordered) * (1 - test_fraction))
        splits[seed] = {"train_doubles": sorted(shuffled[:cut]), "test_doubles": sorted(shuffled[cut:])}
    return splits


# --- Loading what step 2 wrote ----------------------------------------------------------------------------------


def load_profiles() -> tuple[dict[str, np.ndarray], np.ndarray]:
    """{condition: pseudo-bulk profile on the scored genes} and the control profile, as float64."""
    data = np.load(RESULTS / "pseudobulk_top1000.npz")
    profiles = {
        str(c): np.asarray(row, dtype=np.float64) for c, row in zip(data["conditions"], data["values"], strict=True)
    }
    return profiles, profiles[CONTROL]


def load_splits() -> dict[int, dict[str, list[str]]]:
    raw = json.loads((RESULTS / "splits.json").read_text())
    return {int(seed): split for seed, split in raw.items()}


# --- Baselines --------------------------------------------------------------------------------------------------


def no_change(profiles: dict[str, np.ndarray], targets) -> dict[str, np.ndarray]:
    return {canonical_condition(t): profiles[CONTROL] for t in targets}


def additive(profiles: dict[str, np.ndarray], targets) -> dict[str, np.ndarray]:
    """Control plus each single's measured shift. A pair without both singles measured is left out."""
    control = profiles[CONTROL]
    out = {}
    for target in targets:
        singles = [f"{gene}+{CONTROL}" for gene in perturbed_genes(target)]
        if all(single in profiles for single in singles):
            out[canonical_condition(target)] = control + sum(profiles[s] - control for s in singles)
    return out


def mean_response(profiles: dict[str, np.ndarray], targets, train_conditions) -> dict[str, np.ndarray]:
    control = profiles[CONTROL]
    names = [canonical_condition(c) for c in train_conditions]
    average = np.mean([profiles[c] - control for c in names if c in profiles and c != CONTROL], axis=0)
    return {canonical_condition(t): control + average for t in targets}


def ridge_embedding(profiles, targets, train_conditions, embeddings: dict[str, np.ndarray], alpha: float = 0.1):
    """Fit w = (XᵀX + αI)⁻¹XᵀY from each training single's gene vector to its shift; a pair adds two such shifts."""
    control = profiles[CONTROL]
    rows, shifts = [], []
    for condition in map(canonical_condition, train_conditions):
        genes = perturbed_genes(condition)
        if len(genes) == 1 and condition in profiles and genes[0] in embeddings:
            rows.append(embeddings[genes[0]])
            shifts.append(profiles[condition] - control)
    x, y = np.vstack(rows), np.vstack(shifts)
    weights = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ y)
    out = {}
    for target in targets:
        genes = perturbed_genes(target)
        if all(g in embeddings for g in genes):
            out[canonical_condition(target)] = control + sum(embeddings[g] @ weights for g in genes)
    return out


# --- Metrics ----------------------------------------------------------------------------------------------------


def l2(predicted: np.ndarray, observed: np.ndarray) -> float:
    """Euclidean distance between two profiles on the scored genes. Lower is better."""
    return float(np.linalg.norm(np.asarray(predicted, dtype=np.float64) - np.asarray(observed, dtype=np.float64)))


def pearson_delta(predicted: np.ndarray, observed: np.ndarray, control: np.ndarray) -> float:
    """Correlation of the predicted and observed change from control, gene by gene; NaN when a change is constant."""
    p = np.asarray(predicted, dtype=np.float64) - control
    o = np.asarray(observed, dtype=np.float64) - control
    if np.std(p) == 0 or np.std(o) == 0:
        return float("nan")
    return float(np.corrcoef(p, o)[0, 1])


def normalise(score: float, baseline: float, ceiling: float) -> float:
    """Share of the way from the no-change baseline (0) to the measurement-noise ceiling (1), clipped to [0, 1]."""
    return float(np.clip((baseline - score) / (baseline - ceiling), 0.0, 1.0))


def technical_duplicate(cells: np.ndarray, labels: np.ndarray, seed: int = 0) -> dict[str, float]:
    """The ceiling: L2 between one random half of a condition's cells and all of them, per condition.

    Not half against half: two half-means differ by twice as much as a half differs from the pooled mean
    (half1 - pooled = (half1 - half2) / 2), so half against half would double the floor.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for condition in np.unique(labels):
        rows = np.flatnonzero(labels == condition)
        if rows.size < 2:
            continue
        shuffled = rng.permutation(rows)
        half = shuffled.size // 2
        used = shuffled[: 2 * half]
        out[str(condition)] = l2(cells[used[:half]].mean(axis=0), cells[used].mean(axis=0))
    return out


def bootstrap_ci(values, n_resamples: int = 5000, level: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    """(mean, low, high): a percentile bootstrap interval for the mean, resampling held-out pairs."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    estimates = [float(np.mean(values[rng.integers(0, n, size=n)])) for _ in range(n_resamples)]
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(estimates, [tail, 100 - tail])
    return float(values.mean()), float(low), float(high)
