"""Recipe R2: improved hard-negative selection with better cutoff and replay sampling.

Mining: top 2000 pages + 1000 random pages from top 3000 scoring mining pages, all as human (bucket 0)
Replay: 8000 random rows with uniform sampling (no weighting by score)
Cutoff: 99th percentile with proper minimum to avoid overfitting
"""

import json
from pathlib import Path

import numpy as np

HUMAN_W = 1.3278
K_HARD = 2000
K_HARD_EXTRA = 1000
N_REPLAY = 8000


def build(ctx):
    data = Path(ctx["data"])
    scores = np.load(data / "mining_scores_r0.npy")
    
    # Top 2000 as hard negatives
    order = np.lexsort((np.arange(len(scores)), -scores))
    rows = [{"pool": "mining", "idx": int(i), "bucket": 0, "weight": HUMAN_W} for i in order[:K_HARD]]
    
    # Add 1000 more from top 3000 scoring mining pages (random sample)
    top_3000 = order[:3000]
    rng = np.random.default_rng(0)
    extra_indices = rng.choice(top_3000, K_HARD_EXTRA, replace=False)
    for i in extra_indices:
        rows.append({"pool": "mining", "idx": int(i), "bucket": 0, "weight": HUMAN_W})
    
    # Replay with uniform sampling (no weighting by score)
    meta = []
    with open(data / "replay.jsonl") as f:
        for line in f:
            r = json.loads(line)
            meta.append((int(r["bucket"]), float(r["weight"]), r.get("target", 0.5)))

    # Uniform sampling to avoid biasing toward higher-scoring replay samples
    chosen_indices = rng.choice(len(meta), N_REPLAY, replace=False)
    
    for i in sorted(chosen_indices):
        b, w, _ = meta[i]
        rows.append({"pool": "replay", "idx": int(i), "bucket": b, "weight": w})
        
    return {"rows": rows, "train": {"lr": 1e-4, "epochs": 1.0}, "note": "R2: improved cutoff and replay sampling"}


def fit_cutoff(cal):
    # Get human scores from validation set
    s = np.asarray(cal["v2b_val"]["scores"], float)
    human = np.asarray(cal["v2b_val"]["kind"]) == "human"
    
    # Get AI scores from validation set for reference
    ai = np.asarray(cal["v2b_val"]["kind"]) != "human"
    ai_scores = s[ai]
    
    # Calculate the 99th percentile of human scores
    human_scores = s[human]
    cutoff = np.percentile(human_scores, 99.0)
    
    # Ensure we don't go below a reasonable threshold
    # This prevents the cutoff from being too lenient
    if len(ai_scores) > 0:
        # Make sure we don't flag too many AI texts as human
        # Set a minimum based on the lowest AI score that's reasonably distinguishable
        min_ai_score = np.min(ai_scores)
        # Use a stricter minimum to avoid overfitting to the validation set
        cutoff = max(cutoff, min_ai_score + 0.02)
    
    return float(cutoff)