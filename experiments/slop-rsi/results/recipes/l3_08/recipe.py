"""Recipe R0 variant: rank-window hard negatives + moderately raised cutoff.

Mining: pages ranked 150..2,400 by the starting detector's score (dropping the top 150, likely AI-like or
mislabeled pages), 2,000 of them sampled, added as human (bucket 0, data v2b's human weight). Replay: 8,000
random rows of data v2b with their own labels and weights. One epoch at lr 1e-4. Cutoff: q99 of the updated
detector's scores on v2b-val humans, plus 0.10 (capped at 0.52), using the recall slack seen on dev.
"""

import json
from pathlib import Path

import numpy as np

HUMAN_W = 1.3278
K_HARD = 2000
SKIP_TOP = 150
WINDOW_END = 2400
N_REPLAY = 8000
CUT_BUMP = 0.10
CUT_MAX = 0.52


def _read_replay_meta(path):
    """Per-line (bucket, weight), or None for unparseable lines; index = line number."""
    meta = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                meta.append(None)
                continue
            try:
                r = json.loads(line)
                b = int(r["bucket"])
                w = float(r["weight"])
                if b < 0 or b > 3:
                    meta.append(None)
                else:
                    meta.append((b, min(max(w, 0.0), 5.0)))
            except Exception:
                meta.append(None)
    return meta


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
    rows = [{"pool": "mining", "idx": int(i), "bucket": 0, "weight": HUMAN_W, "aug": "none"}
            for i in chosen]

    try:
        meta = _read_replay_meta(data / "replay.jsonl")
        valid = np.array([i for i, m in enumerate(meta) if m is not None], dtype=int)
        if len(valid) > 0:
            rng = np.random.default_rng(0)
            n = min(N_REPLAY, len(valid))
            for i in sorted(rng.choice(valid, n, replace=False)):
                b, w = meta[int(i)]
                rows.append({"pool": "replay", "idx": int(i), "bucket": b, "weight": w, "aug": "none"})
    except Exception:
        pass
    return {"rows": rows, "train": {"lr": 1e-4, "epochs": 1.0},
            "note": "mining ranks 150-2400 (2000 sampled) + 8,000 replay; cutoff q99+0.10"}


def fit_cutoff(cal):
    try:
        s = np.asarray(cal["v2b_val"]["scores"], float)
        human = np.asarray(cal["v2b_val"]["kind"]) == "human"
        if human.sum() == 0:
            return 0.45
        q = float(np.quantile(s[human], 0.99))
        return float(np.clip(min(q + CUT_BUMP, CUT_MAX), 0.0, 1.0))
    except Exception:
        return 0.45