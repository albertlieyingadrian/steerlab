"""Recipe R5: R4 with replay oversampling edit buckets (1,2) to lift recall, slightly higher cutoff."""

import json
from pathlib import Path

import numpy as np

HUMAN_W = 1.3278
AI_W = 1.3092
K_HARD = 2800
K_TOP = 1000
TOP_W = 1.8
N_MIRROR = 1000
N_REPLAY_EDIT = 4500
N_REPLAY_OTHER = 4500


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
            meta.append((int(r["bucket"]), float(r["weight"])))
    buckets = np.array([m[0] for m in meta])
    edit_idx = np.where((buckets == 1) | (buckets == 2))[0]
    other_idx = np.where((buckets == 0) | (buckets == 3))[0]
    sel = list(rng.choice(edit_idx, min(N_REPLAY_EDIT, len(edit_idx)), replace=False))
    sel += list(rng.choice(other_idx, min(N_REPLAY_OTHER, len(other_idx)), replace=False))
    for i in sorted(int(x) for x in sel):
        b, w = meta[i]
        rows.append({"pool": "replay", "idx": i, "bucket": b, "weight": w})
    return {"rows": rows, "train": {"lr": 1e-4, "epochs": 1.0},
            "note": "R5: 2800 hard mined humans + 1000 mirrors + 9000 replay (half edit buckets), cutoff q0.997 clamped [0.52,0.56]"}


def fit_cutoff(cal):
    s = np.asarray(cal["v2b_val"]["scores"], float)
    human = np.asarray(cal["v2b_val"]["kind"]) == "human"
    c = float(np.quantile(s[human], 0.997))
    return float(min(max(c, 0.52), 0.56))