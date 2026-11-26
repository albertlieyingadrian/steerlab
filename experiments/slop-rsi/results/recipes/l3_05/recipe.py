"""Recipe R0 data, with an unlabeled-calibrated cutoff.

Mining: the 2,000 highest-scored pre-2022 mining pages as human (bucket 0). Replay: 8,000 random rows of data v2b
with their own labels and weights. One epoch at lr 1e-4. Cutoff: max(q99 of v2b-val human scores, q99.5 of the
unlabeled pool scores), capped at +0.04 over the v2b-val q99.
"""

import json
from pathlib import Path

import numpy as np

HUMAN_W = 1.3278
K_HARD = 2000
N_REPLAY = 8000
CAP = 0.04


def build(ctx):
    data = Path(ctx["data"])
    scores = np.load(data / "mining_scores_r0.npy")
    order = np.lexsort((np.arange(len(scores)), -scores))  # highest score first, ties by pool order
    rows = [{"pool": "mining", "idx": int(i), "bucket": 0, "weight": HUMAN_W} for i in order[:K_HARD]]

    meta = []
    with open(data / "replay.jsonl") as f:
        for line in f:
            r = json.loads(line)
            meta.append((int(r["bucket"]), float(r["weight"])))
    rng = np.random.default_rng(0)
    for i in sorted(rng.choice(len(meta), N_REPLAY, replace=False)):
        b, w = meta[i]
        rows.append({"pool": "replay", "idx": int(i), "bucket": b, "weight": w})
    return {"rows": rows, "train": {"lr": 1e-4, "epochs": 1.0}, "note": "R0 data; unlabeled-calibrated cutoff"}


def fit_cutoff(cal):
    s = np.asarray(cal["v2b_val"]["scores"], float)
    human = np.asarray(cal["v2b_val"]["kind"]) == "human"
    hs = s[human]
    if hs.size == 0:
        return 0.4
    base = float(np.quantile(hs, 0.99))
    cut = base
    try:
        u = np.asarray(cal["unlabeled"]["scores"], float)
        u = u[np.isfinite(u)]
        if u.size >= 100:
            uq = float(np.quantile(u, 0.995))
            cut = max(base, min(uq, base + CAP))
    except Exception:
        cut = base
    return float(min(1.0, max(0.0, cut)))