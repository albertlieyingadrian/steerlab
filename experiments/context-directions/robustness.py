"""Exploratory check (added after the pre-registered geometry): is the low stability an artefact of a few huge
dimensions of the residual stream?

    python experiments/context-directions/robustness.py

Recomputes the matched-half stability S per layer two ways: (1) after dividing every dimension by its standard
deviation pooled over all rows of the concept (all contexts, both classes), so no dimension dominates the cosine; and
(2) after dropping the 8 dimensions with the largest mean absolute activation at that layer. Writes
results/robustness.json.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study as s


def stability(acts: dict, paired: bool, rng, n_splits: int = 50) -> dict:
    ctxs = list(acts)
    n_pos, n_neg = acts[ctxs[0]][0].shape[0], acts[ctxs[0]][1].shape[0]
    ceil = {x: [] for x in ctxs}
    cross = {f"{x}|{y}": [] for x, y in combinations(ctxs, 2)}
    for _ in range(n_splits):
        if paired:
            perm = rng.permutation(n_pos)
            ia_p, ib_p = perm[: n_pos // 2], perm[n_pos // 2 :]
            ia_n, ib_n = ia_p, ib_p
        else:
            pp, pn = rng.permutation(n_pos), rng.permutation(n_neg)
            ia_p, ib_p, ia_n, ib_n = pp[: n_pos // 2], pp[n_pos // 2 :], pn[: n_neg // 2], pn[n_neg // 2 :]
        half = {x: (p[ia_p].mean(0) - n[ia_n].mean(0), p[ib_p].mean(0) - n[ib_n].mean(0)) for x, (p, n) in acts.items()}
        for x in ctxs:
            ceil[x].append(s.cos_layers(*half[x]))
        for x, y in combinations(ctxs, 2):
            cross[f"{x}|{y}"].append(
                0.5 * (s.cos_layers(half[x][0], half[y][1]) + s.cos_layers(half[y][0], half[x][1]))
            )
    cm = {x: np.mean(v, 0) for x, v in ceil.items()}
    return {k: (np.mean(v, 0) / np.sqrt(cm[k.split("|")[0]] * cm[k.split("|")[1]])).round(4).tolist()
            for k, v in cross.items()}  # fmt: skip


def main() -> None:
    rng = np.random.default_rng(0)
    out: dict = {}
    for concept in s.CONCEPTS:
        raw = {x: s.load_acts(concept, x) for x in s.CONTEXTS}
        pooled = np.concatenate([np.concatenate(v) for v in raw.values()])  # [rows, layers, d]
        sd = pooled.std(0) + 1e-6
        whitened = {x: (p / sd, n / sd) for x, (p, n) in raw.items()}
        big = np.argsort(-np.abs(pooled).mean(0), axis=-1)[:, :8]  # per layer
        keep = np.ones(pooled.shape[1:], dtype=bool)
        np.put_along_axis(keep, big, False, axis=-1)
        dropped = {x: (np.where(keep, p, 0.0), np.where(keep, n, 0.0)) for x, (p, n) in raw.items()}
        out[concept] = {
            "whitened": stability(whitened, s.PAIRED[concept], rng),
            "drop_top8": stability(dropped, s.PAIRED[concept], rng),
        }
        for kind in ("whitened", "drop_top8"):
            summary = {k: (round(v[s.STEER_LAYER], 3), round(float(np.nanmean(v[8:21])), 3))
                       for k, v in out[concept][kind].items()}  # fmt: skip
            print(f"{concept:8s} {kind:9s} S at L14, mean L8-20: {summary}")
    (s.RESULTS / f"robustness{s.TAG}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
