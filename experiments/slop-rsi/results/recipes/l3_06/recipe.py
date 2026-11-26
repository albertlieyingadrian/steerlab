"""Recipe R0 variant: rank-window hard negatives.

Mining: pages ranked 150..2,400 by the starting detector's score (dropping the top 150, likely AI-like or
mislabeled pages), 2,000 of them sampled, added as human (bucket 0, data v2b's human weight). Replay: 8,000
random rows of data v2b with their own labels and weights. One epoch at lr 1e-4. Cutoff: q99 of the updated
detector's scores on v2b-val humans.
"""

import json
from pathlib import Path

import numpy as np

HUMAN_W = 1.3278
K_HARD = 2000
SKIP_TOP = 150
WINDOW_END = 2400
N_REPLAY = 8000


def build(ctx):
    data = Path(ctx["data"])
    scores = np.load(data / "mining_scores_r0.npy")
    order = np.lexsort((np.arange(len(scores)), -scores))  # highest score first, ties by pool order
    try:
        window = order[SKIP_TOP:WINDOW_END]
        rs = np.random.default_rng(1)
        pick = np.sort(rs.choice(len(window), min(K_HARD, len(window)), replace=False))
        chosen = window[pick]
    except Exception:
        chosen = order[:K_HARD]
    rows = [{"pool": "mining", "idx": int(i), "bucket": 0, "weight": HUMAN_W} for i in chosen]

    meta = []
    with open(data / "replay.jsonl") as f:
        for line in f:
            r = json.loads(line)
            meta.append((int(r["bucket"]), float(r["weight"])))
    rng = np.random.default_rng(0)
    for i in sorted(rng.choice(len(meta), N_REPLAY, replace=False)):
        b, w = meta[i]
        rows.append({"pool": "replay", "idx": int(i), "bucket": b, "weight": w})
    return {"rows": rows, "train": {"lr": 1e-4, "epochs": 1.0},
            "note": "mining ranks 150-2400 (2000 sampled) + 8,000 replay"}


def fit_cutoff(cal):
    try:
        s = np.asarray(cal["v2b_val"]["scores"], float)
        human = np.asarray(cal["v2b_val"]["kind"]) == "human"
        if human.sum() == 0:
            return 0.4
        return float(np.clip(np.quantile(s[human], 0.99), 0.0, 1.0))
    except Exception:
        return 0.4