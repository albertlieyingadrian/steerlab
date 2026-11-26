"""Recipe R0: the hard-negative round, at update scale.

Mining: the 2,000 pages of the pre-2022 mining pool that the starting detector scores highest, added as human
(bucket 0, data v2b's human weight), as the hard-negative note's arm (a) round 1 did. Replay: 8,000 random rows of the
detector's own training data (data v2b) with their own labels and weights, so the update does not forget. One epoch at
learning rate 1e-4. Cutoff: the 99th percentile of the updated detector's scores on data v2b's validation humans (the
hard-negative note's rule for each run's own cut).
"""

import json
from pathlib import Path

import numpy as np

HUMAN_W = 1.3278
K_HARD = 2000
N_REPLAY = 8000


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
    return {"rows": rows, "train": {"lr": 1e-4, "epochs": 1.0}, "note": "R0: top-2000 mined pages + 8,000 replay"}


def fit_cutoff(cal):
    s = np.asarray(cal["v2b_val"]["scores"], float)
    human = np.asarray(cal["v2b_val"]["kind"]) == "human"
    return float(np.quantile(s[human], 0.99))
