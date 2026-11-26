"""Recipe R2: host-diverse hard negatives (skipping the very top), more mirrors, mild cutoff clip."""

import json
from pathlib import Path

import numpy as np

HUMAN_W = 1.3278
HARD_W = 1.5
AI_W = 1.3092
K_HARD = 3400
SKIP_TOP = 50
HOST_CAP = 3
N_REPLAY = 6000
N_MIRROR = 1500


def build(ctx):
    data = Path(ctx["data"])
    scores = np.load(data / "mining_scores_r0.npy")
    order = np.lexsort((np.arange(len(scores)), -scores))
    cand = [int(i) for i in order[SKIP_TOP:SKIP_TOP + 12000]]
    cand_set = set(cand)

    hosts = {}
    with open(data / "mining.jsonl") as f:
        for i, line in enumerate(f):
            if i in cand_set:
                try:
                    hosts[i] = json.loads(line).get("host", "")
                except Exception:
                    hosts[i] = ""

    rows = []
    cnt = {}
    for i in cand:
        h = hosts.get(i, "")
        if cnt.get(h, 0) >= HOST_CAP:
            continue
        cnt[h] = cnt.get(h, 0) + 1
        rows.append({"pool": "mining", "idx": i, "bucket": 0, "weight": HARD_W})
        if len(rows) >= K_HARD:
            break

    meta = []
    with open(data / "replay.jsonl") as f:
        for line in f:
            r = json.loads(line)
            meta.append((int(r["bucket"]), float(r["weight"])))
    rng = np.random.default_rng(0)
    for i in sorted(rng.choice(len(meta), N_REPLAY, replace=False)):
        b, w = meta[i]
        rows.append({"pool": "replay", "idx": int(i), "bucket": b, "weight": w})

    n_mirror_total = 0
    with open(data / "mirrors.jsonl") as f:
        for _ in f:
            n_mirror_total += 1
    rng2 = np.random.default_rng(1)
    for i in sorted(rng2.choice(n_mirror_total, min(N_MIRROR, n_mirror_total), replace=False)):
        rows.append({"pool": "mirrors", "idx": int(i), "bucket": 3, "weight": AI_W})

    return {"rows": rows, "train": {"lr": 1e-4, "epochs": 1.0},
            "note": "R2: host-diverse hard negatives ranks 50+, w1.5, 6000 replay, 1500 mirrors"}


def fit_cutoff(cal):
    s = np.asarray(cal["v2b_val"]["scores"], float)
    human = np.asarray(cal["v2b_val"]["kind"]) == "human"
    c = float(np.quantile(s[human], 0.994))
    return float(min(max(c, 0.38), 0.47))