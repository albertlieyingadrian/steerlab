"""Recipe R1: enhanced hard-negative round with increased diversity.

Mining: the 3,000 pages of the pre-2022 mining pool that the starting detector scores highest, added as human
(bucket 0, data v2b's human weight), to capture more diverse human examples. Mirrors: 500 AI rewrites of mining pages,
to add more variety in AI styles. Replay: 7,500 random rows of the detector's own training data (data v2b) with their
own labels and weights, to maintain knowledge of the original distribution. 1.5 epochs at learning rate 2e-4.
Cutoff: the 99th percentile of the updated detector's scores on data v2b's validation humans.
"""

import json
from pathlib import Path

import numpy as np

HUMAN_W = 1.3278
K_HARD = 3000
N_REPLAY = 7500
N_MIRRORS = 500


def build(ctx):
    data = Path(ctx["data"])
    scores = np.load(data / "mining_scores_r0.npy")
    order = np.lexsort((np.arange(len(scores)), -scores))  # highest score first, ties by pool order
    rows = [{"pool": "mining", "idx": int(i), "bucket": 0, "weight": HUMAN_W} for i in order[:K_HARD]]

    # Add mirror samples
    with open(data / "mirrors.jsonl") as f:
        mirrors = []
        for line in f:
            r = json.loads(line)
            mirrors.append(r)
    
    # Sample mirrors randomly
    rng = np.random.default_rng(0)
    mirror_indices = sorted(rng.choice(len(mirrors), N_MIRRORS, replace=False))
    for i in mirror_indices:
        r = mirrors[i]
        rows.append({"pool": "mirrors", "idx": int(i), "bucket": 3, "weight": 1.3092})

    # Add replay samples
    meta = []
    with open(data / "replay.jsonl") as f:
        for line in f:
            r = json.loads(line)
            meta.append((int(r["bucket"]), float(r["weight"])))
    rng = np.random.default_rng(0)
    for i in sorted(rng.choice(len(meta), N_REPLAY, replace=False)):
        b, w = meta[i]
        rows.append({"pool": "replay", "idx": int(i), "bucket": b, "weight": w})
        
    return {"rows": rows, "train": {"lr": 2e-4, "epochs": 1.5}, "note": "R1: 3000 mining + 500 mirrors + 7500 replay"}


def fit_cutoff(cal):
    s = np.asarray(cal["v2b_val"]["scores"], float)
    human = np.asarray(cal["v2b_val"]["kind"]) == "human"
    return float(np.quantile(s[human], 0.99))