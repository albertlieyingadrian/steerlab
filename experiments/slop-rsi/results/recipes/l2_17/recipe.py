"""Recipe R5: R4 training plus middle-view augmented hard negatives, cutoff in 0.52-0.54."""

import json
from pathlib import Path

import numpy as np

HARD_W = 1.6
AUG_W = 1.3278
AI_W = 1.3092
K_HARD = 3000
K_AUG = 600
N_MIRROR = 1500
N_REPLAY = 8000


def build(ctx):
    data = Path(ctx["data"])
    scores = np.load(data / "mining_scores_r0.npy")
    order = np.lexsort((np.arange(len(scores)), -scores))
    rows = [{"pool": "mining", "idx": int(i), "bucket": 0, "weight": HARD_W, "aug": "none"}
            for i in order[:K_HARD]]
    for i in order[K_HARD:K_HARD + K_AUG]:
        rows.append({"pool": "mining", "idx": int(i), "bucket": 0, "weight": AUG_W, "aug": "middle"})

    rng = np.random.default_rng(0)
    n_mirrors = 0
    with open(data / "mirrors.jsonl") as f:
        for _ in f:
            n_mirrors += 1
    pick = sorted(rng.choice(n_mirrors, min(N_MIRROR, n_mirrors), replace=False))
    for i in pick:
        rows.append({"pool": "mirrors", "idx": int(i), "bucket": 3, "weight": AI_W, "aug": "none"})

    meta = []
    with open(data / "replay.jsonl") as f:
        for line in f:
            r = json.loads(line)
            meta.append((int(r["bucket"]), float(r["weight"])))
    for i in sorted(rng.choice(len(meta), N_REPLAY, replace=False)):
        b, w = meta[i]
        rows.append({"pool": "replay", "idx": int(i), "bucket": b, "weight": w, "aug": "none"})
    return {"rows": rows, "train": {"lr": 1e-4, "epochs": 1.0},
            "note": "R5: top-3000 mined humans (w1.6) + 600 middle-view next-ranked humans + 1500 mirrors + 8000 replay"}


def fit_cutoff(cal):
    s = np.asarray(cal["v2b_val"]["scores"], float)
    human = np.asarray(cal["v2b_val"]["kind"]) == "human"
    c = float(np.quantile(s[human], 0.995))
    return float(min(max(c, 0.52), 0.54))