"""Recipe R0 data, with a modestly raised, recall-guarded cutoff.

Mining: the 2,000 highest-scored pre-2022 mining pages as human (bucket 0). Replay: 8,000 random rows of data v2b
with their own labels and weights. One epoch at lr 1e-4. Cutoff: v2b-val human q99 raised by up to +0.03, keeping
v2b-val macro AI recall within 0.8 points of its value at the q99 cutoff.
"""

import json
from pathlib import Path

import numpy as np

HUMAN_W = 1.3278
K_HARD = 2000
N_REPLAY = 8000
MAX_RAISE = 0.03
MAX_RECALL_DROP = 0.008


def build(ctx):
    data = Path(ctx["data"])
    scores = np.load(data / "mining_scores_r0.npy")
    order = np.lexsort((np.arange(len(scores)), -scores))  # highest score first, ties by pool order
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
    return {"rows": rows, "train": {"lr": 1e-4, "epochs": 1.0}, "note": "R0 data; raised guarded cutoff"}


def _macro_recall(s, kinds, ai_kinds, cut):
    rec = []
    for k in ai_kinds:
        m = kinds == k
        if m.sum() > 0:
            rec.append(float(np.mean(s[m] >= cut)))
    return float(np.mean(rec)) if rec else 0.0


def fit_cutoff(cal):
    try:
        s = np.asarray(cal["v2b_val"]["scores"], float)
        kinds = np.asarray(cal["v2b_val"]["kind"])
        human = kinds == "human"
        hs = s[human]
        if hs.size == 0:
            return 0.4
        base = float(np.quantile(hs, 0.99))
        cut = base
        try:
            ai_kinds = [k for k in np.unique(kinds) if k != "human"]
            r_base = _macro_recall(s, kinds, ai_kinds, base)
            for step in np.linspace(MAX_RAISE, 0.0, 16):
                c = base + float(step)
                if r_base - _macro_recall(s, kinds, ai_kinds, c) <= MAX_RECALL_DROP:
                    cut = c
                    break
        except Exception:
            cut = base + 0.015
        return float(min(1.0, max(0.0, cut)))
    except Exception:
        return 0.4