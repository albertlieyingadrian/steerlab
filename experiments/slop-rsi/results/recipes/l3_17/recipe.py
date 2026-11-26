"""Recipe: rank-window hard negatives + R0 replay + large capped cutoff raise.

Mining: 2,000 pages sampled from ranks 150-2,400 of the starting detector's score ranking (top 150 dropped),
as human (bucket 0). Replay: R0's 8,000 random rows with their own labels/weights. 1 epoch, lr 1e-4.
Cutoff: min(q99 of v2b-val human scores + 0.24, 0.62), never below q99.
"""

import json
import random
from pathlib import Path

import numpy as np

HUMAN_W = 1.3278
N_MINING = 2000
N_REPLAY = 8000
OFFSET = 0.24
CAP = 0.62


def _replay_rows(data):
    rows = []
    meta = []
    with open(data / "replay.jsonl") as f:
        for line in f:
            try:
                r = json.loads(line)
                meta.append((int(r["bucket"]), float(r["weight"])))
            except Exception:
                meta.append((0, HUMAN_W))
    rng = np.random.default_rng(0)
    for i in sorted(rng.choice(len(meta), N_REPLAY, replace=False)):
        b, w = meta[int(i)]
        w = min(max(w, 0.0), 5.0)
        rows.append({"pool": "replay", "idx": int(i), "bucket": int(b), "weight": float(w), "aug": "none"})
    return rows


def build(ctx):
    data = Path(ctx["data"])
    sc = np.load(data / "mining_scores_r0.npy")
    order = np.lexsort((np.arange(len(sc)), -sc))
    window = [int(i) for i in order[150:2400]]
    pick = random.Random(ctx.get("seed", 0)).sample(window, N_MINING)
    rows = [{"pool": "mining", "idx": i, "bucket": 0, "weight": HUMAN_W, "aug": "none"} for i in pick]
    try:
        rows += _replay_rows(data)
    except Exception:
        pass
    return {"rows": rows, "train": {"lr": 1e-4, "epochs": 1.0},
            "note": "rank-window mining 150-2400 (2000) + R0 replay; cutoff q99+0.24 cap 0.62"}


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