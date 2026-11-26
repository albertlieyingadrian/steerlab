"""Recipe R2: cleaner hard negatives (skip the noisiest top ranks), fewer mirrors, more replay, safer cutoff window."""

import json
from pathlib import Path

import numpy as np

HUMAN_W = 1.3278
AI_W = 1.3092
SKIP_TOP = 150
K_HARD = 2500
N_MIRROR = 1000
N_REPLAY = 7500


def build(ctx):
    data = Path(ctx["data"])
    scores = np.load(data / "mining_scores_r0.npy")
    order = np.lexsort((np.arange(len(scores)), -scores))
    sel = order[SKIP_TOP:SKIP_TOP + K_HARD]
    rows = [{"pool": "mining", "idx": int(i), "bucket": 0, "weight": HUMAN_W, "aug": "none"} for i in sel]

    n_mirrors = 0
    with open(data / "mirrors.jsonl") as f:
        for _ in f:
            n_mirrors += 1
    rng = np.random.default_rng(1)
    for i in sorted(rng.choice(n_mirrors, min(N_MIRROR, n_mirrors), replace=False)):
        rows.append({"pool": "mirrors", "idx": int(i), "bucket": 3, "weight": AI_W, "aug": "none"})

    meta = []
    with open(data / "replay.jsonl") as f:
        for line in f:
            r = json.loads(line)
            meta.append((int(r["bucket"]), float(r["weight"])))
    rng = np.random.default_rng(0)
    for i in sorted(rng.choice(len(meta), N_REPLAY, replace=False)):
        b, w = meta[i]
        rows.append({"pool": "replay", "idx": int(i), "bucket": b, "weight": w, "aug": "none"})
    return {"rows": rows, "train": {"lr": 1e-4, "epochs": 1.0},
            "note": "R2: mining ranks 150-2650 human + 1000 mirrors AI + 7500 replay"}


def fit_cutoff(cal):
    s = np.asarray(cal["v2b_val"]["scores"], float)
    human = np.asarray(cal["v2b_val"]["kind"]) == "human"
    c = float(np.quantile(s[human], 0.993))
    return float(min(max(c, 0.38), 0.42))