"""Recipe R1: more diverse hard negatives + mirror AI rows to protect recall, slightly higher cutoff."""

import json
from pathlib import Path

import numpy as np

HUMAN_W = 1.3278
AI_W = 1.3092
K_HARD = 3500
PER_HOST = 3
N_MIRROR = 1500
N_REPLAY = 7000


def build(ctx):
    data = Path(ctx["data"])
    scores = np.load(data / "mining_scores_r0.npy")
    order = np.lexsort((np.arange(len(scores)), -scores))
    top = order[:20000]
    need = set(int(i) for i in top)
    hosts = {}
    with open(data / "mining.jsonl") as f:
        for i, line in enumerate(f):
            if i in need:
                hosts[i] = json.loads(line).get("host", "")
    rows = []
    cnt = {}
    for i in top:
        i = int(i)
        h = hosts.get(i, "")
        if cnt.get(h, 0) >= PER_HOST:
            continue
        cnt[h] = cnt.get(h, 0) + 1
        rows.append({"pool": "mining", "idx": i, "bucket": 0, "weight": HUMAN_W})
        if len(rows) >= K_HARD:
            break

    rng = np.random.default_rng(1)
    n_mirrors = 0
    with open(data / "mirrors.jsonl") as f:
        for _ in f:
            n_mirrors += 1
    for i in sorted(rng.choice(n_mirrors, min(N_MIRROR, n_mirrors), replace=False)):
        rows.append({"pool": "mirrors", "idx": int(i), "bucket": 3, "weight": AI_W})

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
            "note": "R1: host-diverse hard negatives 3500 + 1500 mirrors + 7000 replay"}


def fit_cutoff(cal):
    s = np.asarray(cal["v2b_val"]["scores"], float)
    human = np.asarray(cal["v2b_val"]["kind"]) == "human"
    return float(np.quantile(s[human], 0.993))