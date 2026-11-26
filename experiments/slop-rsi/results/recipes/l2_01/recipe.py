"""Recipe R1: R0 with more hard negatives, mirror AI pages, and a slightly higher cutoff quantile."""

import json
from pathlib import Path

import numpy as np

HUMAN_W = 1.3278
AI_W = 1.3092
K_HARD = 2600
N_REPLAY = 7000
N_MIRROR = 1000


def build(ctx):
    data = Path(ctx["data"])
    scores = np.load(data / "mining_scores_r0.npy")
    order = np.lexsort((np.arange(len(scores)), -scores))
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

    n_mirror_total = 0
    with open(data / "mirrors.jsonl") as f:
        for _ in f:
            n_mirror_total += 1
    rng2 = np.random.default_rng(1)
    for i in sorted(rng2.choice(n_mirror_total, min(N_MIRROR, n_mirror_total), replace=False)):
        rows.append({"pool": "mirrors", "idx": int(i), "bucket": 3, "weight": AI_W})

    return {"rows": rows, "train": {"lr": 1e-4, "epochs": 1.0},
            "note": "R1: top-2600 mined humans + 7000 replay + 1000 mirrors as AI"}


def fit_cutoff(cal):
    s = np.asarray(cal["v2b_val"]["scores"], float)
    human = np.asarray(cal["v2b_val"]["kind"]) == "human"
    return float(np.quantile(s[human], 0.993))