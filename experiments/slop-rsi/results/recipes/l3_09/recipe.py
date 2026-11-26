"""Recipe R0 data + large capped cutoff raise.

Mining: the 2,000 pages of the pre-2022 mining pool that the starting detector scores highest, added as human
(bucket 0, data v2b's human weight). Replay: 8,000 random rows of the detector's own training data (data v2b) with
their own labels and weights. One epoch at learning rate 1e-4.
Cutoff: min(q99 of updated detector's scores on v2b-val humans + 0.10, 0.52).
"""

import json
from pathlib import Path

import numpy as np

HUMAN_W = 1.3278
K_HARD = 2000
N_REPLAY = 8000


def build(ctx):
    data = Path(ctx["data"])
    scores = np.load(data / "mining_scores_r0.npy")
    order = np.lexsort((np.arange(len(scores)), -scores))  # highest score first, ties by pool order
    rows = [{"pool": "mining", "idx": int(i), "bucket": 0, "weight": HUMAN_W, "aug": "none"}
            for i in order[:K_HARD]]

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
            "note": "R0 data + cutoff min(q99+0.10, 0.52)"}


def fit_cutoff(cal):
    try:
        v = cal["v2b_val"]
        s = np.asarray(v["scores"], float)
        human = np.asarray(v["kind"]) == "human"
        h = s[human]
        h = h[np.isfinite(h)]
        if h.size == 0:
            return 0.45
        c = min(float(np.quantile(h, 0.99)) + 0.10, 0.52)
        return float(np.clip(c, 0.0, 1.0))
    except Exception:
        return 0.45