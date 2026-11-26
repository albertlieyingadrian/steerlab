"""Recipe R5: R4 training, plus upweighting human replay rows from minority (non-dominant) sources."""

import json
from collections import Counter
from pathlib import Path

import numpy as np

HUMAN_W = 1.3278
AI_W = 1.3092
K_HARD = 3000
K_TOP = 1000
TOP_W = 1.8
N_MIRROR = 1500
N_REPLAY = 8000
MINOR_HUMAN_W = 1.8


def build(ctx):
    data = Path(ctx["data"])
    scores = np.load(data / "mining_scores_r0.npy")
    order = np.lexsort((np.arange(len(scores)), -scores))
    rows = []
    for rank, i in enumerate(order[:K_HARD]):
        w = TOP_W if rank < K_TOP else HUMAN_W
        rows.append({"pool": "mining", "idx": int(i), "bucket": 0, "weight": w})

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
            meta.append((int(r["bucket"]), float(r["weight"]), str(r.get("source", ""))))
    hs = Counter(m[2] for m in meta if m[0] == 0)
    dominant = hs.most_common(1)[0][0] if hs else None
    for i in sorted(rng.choice(len(meta), N_REPLAY, replace=False)):
        b, w, s = meta[i]
        if b == 0 and s != dominant:
            w = max(w, MINOR_HUMAN_W)
        rows.append({"pool": "replay", "idx": int(i), "bucket": b, "weight": w})
    return {"rows": rows, "train": {"lr": 1e-4, "epochs": 1.0},
            "note": "R5: R4 + minority-source human replay rows upweighted to 1.8, cutoff q0.997 clamped [0.50,0.54]"}


def fit_cutoff(cal):
    s = np.asarray(cal["v2b_val"]["scores"], float)
    human = np.asarray(cal["v2b_val"]["kind"]) == "human"
    c = float(np.quantile(s[human], 0.997))
    return float(min(max(c, 0.50), 0.54))