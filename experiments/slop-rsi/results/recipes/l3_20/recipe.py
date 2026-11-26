"""Rank-window hard negatives + moderate capped cutoff raise.

Mining: pages ranked by the starting detector's score; drop the top 150 (likely contaminated / truly AI-like),
sample 2,000 from ranks 150-2,400 as human (bucket 0). Replay: R0's 8,000 rows unchanged. 1 epoch, lr 1e-4.
Cutoff: min(q99 of v2b-val human scores + 0.17, 0.53).
"""

import json
import random
from pathlib import Path

import numpy as np

HUMAN_W = 1.3278
K_MINE = 2000
N_REPLAY = 8000
OFFSET = 0.17
CAP = 0.53


def build(ctx):
    data = Path(ctx["data"])
    seed = int(ctx.get("seed", 0)) if isinstance(ctx, dict) else 0
    scores = np.load(data / "mining_scores_r0.npy")
    order = np.lexsort((np.arange(len(scores)), -scores))
    window = [int(i) for i in order[150:2400]]
    pick = random.Random(seed).sample(window, K_MINE)
    rows = [{"pool": "mining", "idx": int(i), "bucket": 0, "weight": HUMAN_W, "aug": "none"} for i in pick]

    try:
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
            rows.append({"pool": "replay", "idx": int(i), "bucket": b, "weight": w, "aug": "none"})
    except Exception:
        pass
    return {"rows": rows, "train": {"lr": 1e-4, "epochs": 1.0},
            "note": "rank-window mining (150-2400) + 8000 replay, cutoff q99+0.17 cap 0.53"}


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