"""Recipe R5: R4 training, but replay resampled toward edit buckets (1,2); cutoff from pooled human scores."""

import json
from pathlib import Path

import numpy as np

HUMAN_W = 1.3278
AI_W = 1.3092
K_HARD = 3000
N_MIRROR = 1500
N_REPLAY = 8000
BUCKET_FRAC = {0: 0.30, 1: 0.25, 2: 0.25, 3: 0.20}


def build(ctx):
    data = Path(ctx["data"])
    scores = np.load(data / "mining_scores_r0.npy")
    order = np.lexsort((np.arange(len(scores)), -scores))
    rows = [{"pool": "mining", "idx": int(i), "bucket": 0, "weight": HUMAN_W} for i in order[:K_HARD]]

    rng = np.random.default_rng(0)
    n_mirrors = 0
    with open(data / "mirrors.jsonl") as f:
        for _ in f:
            n_mirrors += 1
    pick = sorted(rng.choice(n_mirrors, min(N_MIRROR, n_mirrors), replace=False))
    for i in pick:
        rows.append({"pool": "mirrors", "idx": int(i), "bucket": 3, "weight": AI_W})

    meta = []
    with open(data / "replay.jsonl") as f:
        for line in f:
            r = json.loads(line)
            meta.append((int(r["bucket"]), float(r["weight"])))
    bks = np.array([m[0] for m in meta])
    chosen = []
    for b, frac in BUCKET_FRAC.items():
        idxs = np.where(bks == b)[0]
        n = min(int(round(N_REPLAY * frac)), len(idxs))
        chosen.extend(rng.choice(idxs, n, replace=False).tolist())
    for i in sorted(chosen):
        b, w = meta[i]
        rows.append({"pool": "replay", "idx": int(i), "bucket": b, "weight": w})
    return {"rows": rows, "train": {"lr": 1e-4, "epochs": 1.0},
            "note": "R5: top-3000 mined humans + 1500 mirrors + 8000 replay skewed to edit buckets; cutoff pooled q0.997 clamped [0.54,0.57]"}


def fit_cutoff(cal):
    s = np.asarray(cal["v2b_val"]["scores"], float)
    human = np.asarray(cal["v2b_val"]["kind"]) == "human"
    u = np.asarray(cal["unlabeled"]["scores"], float)
    pooled = np.concatenate([s[human], u])
    c = float(np.quantile(pooled, 0.997))
    return float(min(max(c, 0.54), 0.57))