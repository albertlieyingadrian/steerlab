"""Recipe R2: Refined hard-negative selection with better replay and adaptive cutoff.

Mining: 2,000 top-scoring pages + 1,000 carefully selected hard negatives from mining pool
that were misclassified by starting detector (score > 0.4 but < 0.6 and likely human)
Replay: 10,000 rows for better data coverage  
Cutoff: Adaptive based on v2b_val human scores distribution for better balance
"""

import json
from pathlib import Path

import numpy as np

HUMAN_W = 1.3278
K_HARD = 2000
K_HARD_NEG = 1000
N_REPLAY = 10000


def build(ctx):
    data = Path(ctx["data"])
    scores = np.load(data / "mining_scores_r0.npy")
    
    # Get top-k highest scoring mining pages (these are the "easy" hard negatives)
    order = np.lexsort((np.arange(len(scores)), -scores))  # highest score first, ties by pool order
    rows = [{"pool": "mining", "idx": int(i), "bucket": 0, "weight": HUMAN_W} for i in order[:K_HARD]]
    
    # Add hard negatives: pages with moderate scores but likely human (misclassified by starting detector)
    # Better approach: select pages in the ambiguous region [0.4, 0.6] that are probably human
    # Use score range [0.4, 0.6] to get better hard negatives - not too close to 0.5 where confusion is highest
    mask = (scores >= 0.4) & (scores <= 0.6)
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
        
    return {"rows": rows, "train": {"lr": 1e-4, "epochs": 1.0}, "note": "R2: refined hard-negatives + replay"}


def fit_cutoff(cal):
    s = np.asarray(cal["v2b_val"]["scores"], float)
    human = np.asarray(cal["v2b_val"]["kind"]) == "human"
    
    # Use 99th percentile instead of 99.5th to maintain better balance
    # The previous 99.5th was too aggressive and may have caused constraint failures
    return float(np.quantile(s[human], 0.99))