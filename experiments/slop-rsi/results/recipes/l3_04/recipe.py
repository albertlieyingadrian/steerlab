"""Recipe: R0 data + a few AI mirror rewrites; recall-aware cutoff that is raised as far as macro recall allows."""

import json
from pathlib import Path

import numpy as np

HUMAN_W = 1.3278
AI_W = 1.3092
K_HARD = 2000
N_REPLAY = 8000
N_MIRROR = 800


def build(ctx):
    data = Path(ctx["data"])
    scores = np.load(data / "mining_scores_r0.npy")
    order = np.lexsort((np.arange(len(scores)), -scores))
    rows = [{"pool": "mining", "idx": int(i), "bucket": 0, "weight": HUMAN_W} for i in order[:K_HARD]]

    meta = []
    with open(data / "replay.jsonl") as f:
        for line in f:
            r = json.loads(line)
            meta.append((int(r["bucket"]), float(r["weight"])))
    rng = np.random.default_rng(0)
    for i in sorted(rng.choice(len(meta), N_REPLAY, replace=False)):
        b, w = meta[i]
        rows.append({"pool": "replay", "idx": int(i), "bucket": b, "weight": w})

    n_mirrors = 0
    with open(data / "mirrors.jsonl") as f:
        for _ in f:
            n_mirrors += 1
    rng2 = np.random.default_rng(1)
    for i in sorted(rng2.choice(n_mirrors, min(N_MIRROR, n_mirrors), replace=False)):
        rows.append({"pool": "mirrors", "idx": int(i), "bucket": 3, "weight": AI_W})
    return {"rows": rows, "train": {"lr": 1e-4, "epochs": 1.0}, "note": "R0 + 800 mirrors"}


def _macro_recall(scores, kinds, ai_kinds, c):
    rec = []
    for k in ai_kinds:
        m = kinds == k
        rec.append(float(np.mean(scores[m] >= c)))
    return float(np.mean(rec))


def fit_cutoff(cal):
    s = np.asarray(cal["v2b_val"]["scores"], float)
    kinds = np.asarray(cal["v2b_val"]["kind"])
    human = kinds == "human"
    c0 = float(np.quantile(s[human], 0.99))
    ai_kinds = [k for k in np.unique(kinds) if k != "human"]
    if not ai_kinds:
        return c0
    base = _macro_recall(s, kinds, ai_kinds, c0)
    best = c0
    for c in np.arange(c0, min(c0 + 0.12, 0.95) + 1e-9, 0.005):
        if _macro_recall(s, kinds, ai_kinds, float(c)) >= base - 0.006:
            best = float(c)
    return float(min(max(best, 0.0), 1.0))