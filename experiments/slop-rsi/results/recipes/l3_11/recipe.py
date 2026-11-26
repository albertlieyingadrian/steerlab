"""Recipe: rank-window mining negatives + R0 replay + larger capped cutoff raise informed by dev feedback."""

import json
from pathlib import Path

import numpy as np

HUMAN_W = 1.3278
N_REPLAY = 8000
DROP_TOP = 150
WIN_END = 2400
N_MINE = 2000


def _replay_rows(data):
    meta = []
    with open(data / "replay.jsonl") as f:
        for line in f:
            try:
                r = json.loads(line)
                meta.append((int(r.get("bucket", 0)), float(r.get("weight", 1.0))))
            except Exception:
                meta.append((0, 1.0))
    rng = np.random.default_rng(0)
    out = []
    n = min(N_REPLAY, len(meta))
    for i in sorted(rng.choice(len(meta), n, replace=False)):
        b, w = meta[int(i)]
        b = min(3, max(0, b))
        w = min(5.0, max(0.0, w))
        out.append({"pool": "replay", "idx": int(i), "bucket": b, "weight": w})
    return out


def build(ctx):
    data = Path(ctx["data"])
    seed = int(ctx.get("seed", 0))
    scores = np.load(data / "mining_scores_r0.npy")
    order = np.lexsort((np.arange(len(scores)), -scores))
    window = order[DROP_TOP:WIN_END]
    rng = np.random.default_rng(seed + 12345)
    n = min(N_MINE, len(window))
    pick = rng.choice(len(window), n, replace=False)
    chosen = window[np.sort(pick)]
    rows = [{"pool": "mining", "idx": int(i), "bucket": 0, "weight": HUMAN_W} for i in chosen]
    rows += _replay_rows(data)
    return {"rows": rows, "train": {"lr": 1e-4, "epochs": 1.0},
            "note": "rank-window mining (150-2400, 2000 sampled) + 8000 replay, cutoff q99+0.18 cap 0.55 with recall guard"}


def fit_cutoff(cal):
    try:
        v = cal["v2b_val"]
        s = np.asarray(v["scores"], float)
        kind = np.asarray(v["kind"])
        human = kind == "human"
        h = s[human]
        if h.size == 0:
            return 0.45
        q99 = float(np.quantile(h, 0.99))
        lo = q99 + 0.08
        c = min(q99 + 0.18, 0.55)
        c = max(c, lo)
        c = min(c, 0.58)
        ai = s[~human]
        if ai.size > 0:
            base_c = min(q99 + 0.10, 0.52)
            base_rec = float(np.mean(ai >= base_c))
            while c > lo and base_rec - float(np.mean(ai >= c)) > 0.015:
                c -= 0.01
            c = max(c, lo)
        if not np.isfinite(c):
            return 0.45
        return float(min(1.0, max(0.0, c)))
    except Exception:
        return 0.45