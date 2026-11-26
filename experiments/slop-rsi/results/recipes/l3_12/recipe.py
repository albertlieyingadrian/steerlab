"""Recipe: rank-window mining negatives + R0 replay + large capped, recall-guarded cutoff raise.

Mining: pages ranked by the starting detector's score; skip the top 150, sample 2,000 from ranks 150-2,400
as human (bucket 0). Replay: 8,000 random rows with their own labels/weights. 1 epoch, lr 1e-4.
Cutoff: min(q99 of v2b-val human + 0.10, 0.52), lowered if v2b-val macro AI-kind recall would drop more than
1.0 point below its value at q99.
"""

import json
from pathlib import Path

import numpy as np

HUMAN_W = 1.3278
K_MINING = 2000
SKIP = 150
WINDOW_END = 2400
N_REPLAY = 8000


def build(ctx):
    data = Path(ctx["data"])
    rows = []
    try:
        scores = np.load(data / "mining_scores_r0.npy")
        order = np.lexsort((np.arange(len(scores)), -scores))
        window = order[SKIP:WINDOW_END]
        rng_m = np.random.default_rng(0)
        k = min(K_MINING, len(window))
        pick = rng_m.choice(len(window), k, replace=False)
        for j in sorted(pick):
            rows.append({"pool": "mining", "idx": int(window[j]), "bucket": 0,
                         "weight": HUMAN_W, "aug": "none"})
    except Exception:
        rows = []

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
        b, w = meta[i]
        b = min(max(int(b), 0), 3)
        w = min(max(float(w), 0.0), 5.0)
        rows.append({"pool": "replay", "idx": int(i), "bucket": b, "weight": w, "aug": "none"})
    return {"rows": rows, "train": {"lr": 1e-4, "epochs": 1.0},
            "note": "rank window 150-2400 mining + R0 replay + guarded cutoff"}


def _macro_recall(s, kind, c):
    ks = sorted(set(kind.tolist()))
    rec = []
    for k in ks:
        m = kind == k
        if m.sum() > 0:
            rec.append(float(np.mean(s[m] >= c)))
    return float(np.mean(rec)) if rec else 1.0


def fit_cutoff(cal):
    try:
        v = cal["v2b_val"]
        s = np.asarray(v["scores"], float)
        kind = np.asarray(v["kind"])
        human = kind == "human"
        h = s[human]
        h = h[np.isfinite(h)]
        if h.size == 0:
            return 0.45
        q99 = float(np.quantile(h, 0.99))
        target = min(q99 + 0.10, 0.52)
        target = max(target, q99)
        ai = (~human) & np.isfinite(s)
        if ai.sum() == 0:
            return float(np.clip(target, 0.0, 1.0))
        sa, ka = s[ai], kind[ai]
        base = _macro_recall(sa, ka, q99)
        c = target
        floor_c = min(q99 + 0.03, target)
        while c > floor_c + 1e-9 and _macro_recall(sa, ka, c) < base - 0.010:
            c -= 0.01
        c = max(c, floor_c)
        return float(np.clip(c, 0.0, 1.0))
    except Exception:
        return 0.45