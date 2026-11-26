"""Recipe: rank-window hard negatives + R0 replay, with a large capped cutoff raise.

Mining: pages ranked by the starting detector's score (desc); drop the top 150 (likely mislabeled/AI-like), sample
~2,000 from ranks 150-2,400 as human (bucket 0). Replay: R0's 8,000 rows with their own labels and weights.
One epoch at lr 1e-4. Cutoff: min(q99 of v2b-val human scores + 0.24, 0.55).
"""

import json
from pathlib import Path

import numpy as np

HUMAN_W = 1.3278
N_MINING = 2000
SKIP_TOP = 150
WINDOW_END = 2400
N_REPLAY = 8000
OFFSET = 0.24
CAP = 0.55


def build(ctx):
    data = Path(ctx["data"])
    scores = np.load(data / "mining_scores_r0.npy")
    order = np.lexsort((np.arange(len(scores)), -scores))
    window = order[SKIP_TOP:WINDOW_END]
    rng_m = np.random.default_rng(1)
    n_m = min(N_MINING, len(window))
    pick = rng_m.choice(len(window), n_m, replace=False)
    rows = [{"pool": "mining", "idx": int(window[j]), "bucket": 0, "weight": HUMAN_W,
             "aug": "none"} for j in sorted(pick)]

    meta = []
    with open(data / "replay.jsonl") as f:
        for line in f:
            try:
                r = json.loads(line)
                b = int(r["bucket"])
                w = float(r["weight"])
                if not (0 <= b <= 3):
                    b, w = 0, HUMAN_W
                meta.append((b, min(max(w, 0.0), 5.0)))
            except Exception:
                meta.append((0, HUMAN_W))
    rng = np.random.default_rng(0)
    n_r = min(N_REPLAY, len(meta))
    for i in sorted(rng.choice(len(meta), n_r, replace=False)):
        b, w = meta[int(i)]
        rows.append({"pool": "replay", "idx": int(i), "bucket": b, "weight": w, "aug": "none"})
    return {"rows": rows, "train": {"lr": 1e-4, "epochs": 1.0},
            "note": "rank-window mining (150-2400) + 8000 replay, cutoff q99+0.24 cap 0.55"}


def fit_cutoff(cal):
    try:
        v = cal["v2b_val"]
        s = np.asarray(v["scores"], float)
        kind = np.asarray(v["kind"])
        human = kind == "human"
        if s.size == 0 or human.sum() == 0:
            return 0.5
        q = float(np.quantile(s[human], 0.99))
        c = min(q + OFFSET, CAP)
        if not np.isfinite(c):
            return 0.5
        return float(np.clip(c, 0.0, 1.0))
    except Exception:
        return 0.5