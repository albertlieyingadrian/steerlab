"""Recipe R5: R3-style training plus lower-cased copies of the hardest mined humans."""

import json
from pathlib import Path

import numpy as np

HUMAN_W = 1.3278
AI_W = 1.3092
K_HARD = 3000
K_LOWER = 400
LOWER_W = 1.0
N_MIRROR = 1500
N_REPLAY = 8000


def build(ctx):
    data = Path(ctx["data"])
    scores = np.load(data / "mining_scores_r0.npy")
    order = np.lexsort((np.arange(len(scores)), -scores))
    rows = []
    for rank, i in enumerate(order[:K_HARD]):
        rows.append({"pool": "mining", "idx": int(i), "bucket": 0,
                     "weight": HUMAN_W, "aug": "none"})
    for i in order[:K_LOWER]:
        rows.append({"pool": "mining", "idx": int(i), "bucket": 0,
                     "weight": LOWER_W, "aug": "lower"})

    rng = np.random.default_rng(0)
    n_mirrors = 0
    with open(data / "mirrors.jsonl") as f:
        for _ in f:
            n_mirrors += 1
    pick = sorted(rng.choice(n_mirrors, min(N_MIRROR, n_mirrors), replace=False))
    for i in pick:
        rows.append({"pool": "mirrors", "idx": int(i), "bucket": 3,
                     "weight": AI_W, "aug": "none"})

    meta = []
    with open(data / "replay.jsonl") as f:
        for line in f:
            r = json.loads(line)
            meta.append((int(r["bucket"]), float(r["weight"])))
    for i in sorted(rng.choice(len(meta), N_REPLAY, replace=False)):
        b, w = meta[i]
        rows.append({"pool": "replay", "idx": int(i), "bucket": b,
                     "weight": w, "aug": "none"})
    return {"rows": rows, "train": {"lr": 1e-4, "epochs": 1.0},
            "note": "R5: top-3000 mined humans + 400 lowercased copies + 1500 mirrors + 8000 replay, cutoff q0.998 clamped [0.52,0.56]"}


def fit_cutoff(cal):
    s = np.asarray(cal["v2b_val"]["scores"], float)
    human = np.asarray(cal["v2b_val"]["kind"]) == "human"
    c = float(np.quantile(s[human], 0.998))
    return float(min(max(c, 0.52), 0.56))