"""Recipe R3: Tighter hard-negative selection with increased replay and refined cutoff.

Mining: 2,000 top-scoring pages + 1,000 carefully selected hard negatives from mining pool
that were misclassified by starting detector (score 0.45-0.55, likely human)
Replay: 11,000 rows for better data coverage  
Cutoff: Refined based on v2b_val human scores distribution for better balance
"""

import json
from pathlib import Path

import numpy as np

HUMAN_W = 1.3278
K_HARD = 2000
K_HARD_NEG = 1000
N_REPLAY = 11000


def build(ctx):
    data = Path(ctx["data"])
    scores = np.load(data / "mining_scores_r0.npy")
    
    # Get top-k highest scoring mining pages (these are the "easy" hard negatives)
    order = np.lexsort((np.arange(len(scores)), -scores))  # highest score first, ties by pool order
    rows = [{"pool": "mining", "idx": int(i), "bucket": 0, "weight": HUMAN_W} for i in order[:K_HARD]]
    
    # Add hard negatives: pages with moderate scores but likely human (misclassified by starting detector)
    # Tighter range [0.45, 0.55] to focus on clearer borderline cases
    mask = (scores >= 0.45) & (scores <= 0.55)
    hard_neg_indices = order[mask]
    # Take top 1000 from these (they're likely to be borderline cases)
    hard_neg_indices = hard_neg_indices[:K_HARD_NEG]
    for i in hard_neg_indices:
        rows.append({"pool": "mining", "idx": int(i), "bucket": 0, "weight": HUMAN_W})
    
    # Replay data
    meta = []
    with open(data / "replay.jsonl") as f:
        for line in f:
            r = json.loads(line)
            meta.append((int(r["bucket"]), float(r["weight"])))
    rng = np.random.default_rng(0)
    for i in sorted(rng.choice(len(meta), N_REPLAY, replace=False)):
        b, w = meta[i]
        rows.append({"pool": "replay", "idx": int(i), "bucket": b, "weight": w})
        
    return {"rows": rows, "train": {"lr": 1e-4, "epochs": 1.0}, "note": "R3: tighter hard-negatives + replay"}


def fit_cutoff(cal):
    s = np.asarray(cal["v2b_val"]["scores"], float)
    human = np.asarray(cal["v2b_val"]["kind"]) == "human"
    
    # Use 98.5th percentile instead of 99th to allow more human texts through
    # This should reduce the pooled FPR at matched recall while maintaining macro recall
    return float(np.quantile(s[human], 0.985))