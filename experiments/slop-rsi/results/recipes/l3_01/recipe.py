"""Recipe R1: more hard negatives + mirror AI rewrites (bucket 3) + replay.

Mining: top-3000 scored pre-2022 pages as human. Mirrors: 2000 AI rewrites as fully AI, so that polished prose is
separated by style and not by topic. Replay: 6,000 random rows for retention. Cutoff: 99th percentile of v2b-val humans.
"""

import json
from pathlib import Path

import numpy as np

HUMAN_W = 1.3278
AI_W = 1.3092
K_HARD = 3000
N_MIRROR = 2000
N_REPLAY = 6000


def build(ctx):
    data = Path(ctx["data"])
    scores = np.load(data / "mining_scores_r0.npy")
    order = np.lexsort((np.arange(len(scores)), -scores))
    rows = [{"pool": "mining", "idx": int(i), "bucket": 0, "weight": HUMAN_W, "aug": "none"} for i in order[:K_HARD]]

    n_mirrors = 0
    with open(data / "mirrors.jsonl") as f:
        for _ in f:
            n_mirrors += 1
    rng = np.random.default_rng(1)
    for i in sorted(rng.choice(n_mirrors, min(N_MIRROR, n_mirrors), replace=False)):
        rows.append({"pool": "mirrors", "idx": int(i), "bucket": 3, "weight": AI_W, "aug": "none"})

    meta = []
    with open(data / "replay.jsonl") as f:
        for line in f:
            r = json.loads(line)
            meta.append((int(r["bucket"]), float(r["weight"])))
    rng = np.random.default_rng(0)
    for i in sorted(rng.choice(len(meta), N_REPLAY, replace=False)):
        b, w = meta[i]
        rows.append({"pool": "replay", "idx": int(i), "bucket": b, "weight": w, "aug": "none"})
    return {"rows": rows, "train": {"lr": 1e-4, "epochs": 1.0},
            "note": "R1: top-3000 mined humans + 2000 mirrors AI + 6000 replay"}


def fit_cutoff(cal):
    s = np.asarray(cal["v2b_val"]["scores"], float)
    human = np.asarray(cal["v2b_val"]["kind"]) == "human"
    return float(np.quantile(s[human], 0.99))