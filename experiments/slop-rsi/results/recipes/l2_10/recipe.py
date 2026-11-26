"""Recipe R4: R3 with a broader hard-negative set (top 2000 + 1000 sampled from ranks 2000-20000)."""

import json
from pathlib import Path

import numpy as np

HUMAN_W = 1.3278
AI_W = 1.3092
K_TOP = 2000
K_BROAD = 1000
N_MIRROR = 1500
N_REPLAY = 8000


def build(ctx):
    data = Path(ctx["data"])
    scores = np.load(data / "mining_scores_r0.npy")
    order = np.lexsort((np.arange(len(scores)), -scores))
    rng = np.random.default_rng(0)
    top = [int(i) for i in order[:K_TOP]]
    broad = rng.choice(order[K_TOP:20000], K_BROAD, replace=False)
    rows = [{"pool": "mining", "idx": i, "bucket": 0, "weight": HUMAN_W} for i in top]
    rows += [{"pool": "mining", "idx": int(i), "bucket": 0, "weight": HUMAN_W} for i in sorted(broad)]

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
    for i in sorted(rng.choice(len(meta), N_REPLAY, replace=False)):
        b, w = meta[i]
        rows.append({"pool": "replay", "idx": int(i), "bucket": b, "weight": w})
    return {"rows": rows, "train": {"lr": 1e-4, "epochs": 1.0},
            "note": "R4: top-2000 + 1000 broad mined humans + 1500 mirrors + 8000 replay, cutoff ~0.54"}


def fit_cutoff(cal):
    s = np.asarray(cal["v2b_val"]["scores"], float)
    human = np.asarray(cal["v2b_val"]["kind"]) == "human"
    c = float(np.quantile(s[human], 0.995))
    return float(min(max(c, 0.52), 0.56))