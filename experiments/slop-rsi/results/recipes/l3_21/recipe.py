"""Recipe: rank-window hard negatives + R0 replay, with a large capped cutoff raise.

Mining: pages sorted by the starting detector's score (descending); drop the top 150 (likely mislabeled or truly
AI-like), sample 2,000 from ranks 150-2,400 as human (bucket 0). Replay: R0's 8,000 rows. 1 epoch, lr 1e-4.
Cutoff: q99 of v2b-val human scores + 0.14, capped at 0.55.
"""

import json
from pathlib import Path

import numpy as np

HUMAN_W = 1.3278
K_MINE = 2000
SKIP_TOP = 150
WINDOW_END = 2400
N_REPLAY = 8000
OFFSET = 0.14
CAP = 0.55


def build(ctx):
    data = Path(ctx["data"])
    scores = np.load(data / "mining_scores_r0.npy")
    order = np.lexsort((np.arange(len(scores)), -scores))
    window = order[SKIP_TOP:WINDOW_END]
    rng_m = np.random.default_rng(1)
    k = min(K_MINE, len(window))
    pick = rng_m.choice(len(window), k, replace=False)
    rows = [{"pool": "mining", "idx": int(window[j]), "bucket": 0, "weight": HUMAN_W,
             "aug": "none"} for j in sorted(pick)]

    meta = []
    with open(data / "replay.jsonl") as f:
        for line in f:
            try:
                r = json.loads(line)
                meta.append((int(r["bucket"]), float(r["weight"])))
            except Exception:
                meta.append((0, 1.0))  # keep index alignment
    rng = np.random.default_rng(0)
    n = min(N_REPLAY, len(meta))
    for i in sorted(rng.choice(len(meta), n, replace=False)):
        b, w = meta[int(i)]
        b = min(3, max(0, b))
        w = min(5.0, max(0.0, w))
        rows.append({"pool": "replay", "idx": int(i), "bucket": b, "weight": w, "aug": "none"})
    return {"rows": rows, "train": {"lr": 1e-4, "epochs": 1.0},
            "note": "rank-window mining 150-2400 + 8k replay, cutoff q99+0.14 cap 0.55"}


def fit_cutoff(cal):
    try:
        s = np.asarray(cal["v2b_val"]["scores"], float)
        kind = np.asarray(cal["v2b_val"]["kind"])
        h = s[kind == "human"]
        h = h[np.isfinite(h)]
        if h.size == 0:
            return 0.5
        c = min(float(np.quantile(h, 0.99)) + OFFSET, CAP)
        return float(min(1.0, max(0.0, c)))
    except Exception:
        return 0.5