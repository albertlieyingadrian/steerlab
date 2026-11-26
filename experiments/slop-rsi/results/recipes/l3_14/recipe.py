"""Recipe: rank-window hard negatives + R0 replay + large capped cutoff raise.

Mining: drop the 150 highest-scored pages (by the starting detector), sample 2,000 pages from ranks 150-2,400,
labelled human (bucket 0). Replay: R0's 8,000 rows with their own labels and weights. 1 epoch, lr 1e-4.
Cutoff: min(q99 of v2b-val human scores + 0.16, 0.56).
"""

import json
import random
from pathlib import Path

import numpy as np

HUMAN_W = 1.3278
N_MINING = 2000
N_REPLAY = 8000
OFFSET = 0.16
CAP = 0.56


def build(ctx):
    data = Path(ctx["data"])
    seed = int(ctx.get("seed", 0))
    rows = []

    scores = np.load(data / "mining_scores_r0.npy")
    order = np.lexsort((np.arange(len(scores)), -scores))
    window = [int(i) for i in order[150:2400]]
    pick = random.Random(seed).sample(window, min(N_MINING, len(window)))
    for i in pick:
        rows.append({"pool": "mining", "idx": int(i), "bucket": 0, "weight": HUMAN_W, "aug": "none"})

    meta = []
    try:
        with open(data / "replay.jsonl") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    meta.append((int(r["bucket"]), float(r["weight"])))
                except Exception:
                    meta.append((0, HUMAN_W))
    except Exception:
        meta = []
    if len(meta) >= N_REPLAY:
        rng = np.random.default_rng(0)
        for i in sorted(rng.choice(len(meta), N_REPLAY, replace=False)):
            b, w = meta[int(i)]
            b = min(max(int(b), 0), 3)
            w = min(max(float(w), 0.0), 5.0)
            rows.append({"pool": "replay", "idx": int(i), "bucket": b, "weight": w, "aug": "none"})
    return {"rows": rows, "train": {"lr": 1e-4, "epochs": 1.0},
            "note": "rank-window mining 150-2400 (2000 sampled) + 8000 replay, cutoff q99+0.16 cap 0.56"}


def fit_cutoff(cal):
    try:
        s = np.asarray(cal["v2b_val"]["scores"], dtype=float)
        k = np.asarray(cal["v2b_val"]["kind"])
        h = s[k == "human"]
        if h.size < 50:
            return 0.45
        q99 = float(np.quantile(h, 0.99))
        c = min(q99 + OFFSET, CAP)
        c = max(c, q99)
        return float(min(max(c, 0.0), 1.0))
    except Exception:
        return 0.45