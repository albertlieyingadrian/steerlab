"""Recipe R5: R4 training, with source-balanced replay sampling within each bucket."""

import json
from collections import Counter
from pathlib import Path

import numpy as np

HUMAN_W = 1.3278
HARD_W = 1.6
AI_W = 1.3092
K_HARD = 3000
N_MIRROR = 1500
N_REPLAY = 8000


def build(ctx):
    data = Path(ctx["data"])
    scores = np.load(data / "mining_scores_r0.npy")
    order = np.lexsort((np.arange(len(scores)), -scores))
    rows = [{"pool": "mining", "idx": int(i), "bucket": 0, "weight": HARD_W} for i in order[:K_HARD]]

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
    n = len(meta)
    buckets = np.array([m[0] for m in meta])
    chosen = []
    for b in range(4):
        idxs = np.where(buckets == b)[0]
        if len(idxs) == 0:
            continue
        quota = int(round(N_REPLAY * len(idxs) / n))
        quota = min(quota, len(idxs))
        cnt = Counter(meta[i][2] for i in idxs)
        p = np.array([1.0 / np.sqrt(cnt[meta[i][2]]) for i in idxs])
        p = p / p.sum()
        sel = rng.choice(idxs, quota, replace=False, p=p)
        chosen.extend(int(i) for i in sel)
    for i in sorted(chosen):
        b, w, _ = meta[i]
        rows.append({"pool": "replay", "idx": int(i), "bucket": b, "weight": w})
    return {"rows": rows, "train": {"lr": 1e-4, "epochs": 1.0},
            "note": "R5: top-3000 mined humans (w1.6) + 1500 mirrors + ~8000 source-balanced replay, cutoff ~0.52-0.56"}


def fit_cutoff(cal):
    s = np.asarray(cal["v2b_val"]["scores"], float)
    human = np.asarray(cal["v2b_val"]["kind"]) == "human"
    c = float(np.quantile(s[human], 0.995))
    return float(min(max(c, 0.52), 0.56))