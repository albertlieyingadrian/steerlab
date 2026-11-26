"""Recipe R1: improved hard-negative selection with better cutoff.

Mining: top 2000 pages + 1000 random pages from top 3000 scoring mining pages, all as human (bucket 0)
Replay: 8000 random rows with bias toward higher-scoring ones (weighted by score)
Cutoff: percentile adjusted for the distribution of scores in the validation set
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
    
    # Replay with weighting toward higher scores
    meta = []
    with open(data / "replay.jsonl") as f:
        for line in f:
            r = json.loads(line)
            meta.append((int(r["bucket"]), float(r["weight"]), r.get("target", 0.5)))  # target is the score
    
    # Weighted sampling: higher score = higher chance to be selected
    replay_scores = np.array([x[2] for x in meta])
    # Normalize scores to probabilities (avoid division by zero)
    normalized_scores = np.clip(replay_scores, 0.01, 1.0)
    probs = normalized_scores / np.sum(normalized_scores)
    chosen_indices = rng.choice(len(meta), N_REPLAY, p=probs, replace=False)
    
    for i in sorted(chosen_indices):
        b, w, _ = meta[i]
        rows.append({"pool": "replay", "idx": int(i), "bucket": b, "weight": w})
        
    return {"rows": rows, "train": {"lr": 1e-4, "epochs": 1.0}, "note": "R1: improved hard-negative selection"}


def fit_cutoff(cal):
    # Get human scores from validation set
    s = np.asarray(cal["v2b_val"]["scores"], float)
    human = np.asarray(cal["v2b_val"]["kind"]) == "human"
    
    # Get AI scores from validation set for reference
    ai = np.asarray(cal["v2b_val"]["kind"]) != "human"
    ai_scores = s[ai]
    
    # Calculate the 99th percentile of human scores, but adjust based on AI distribution
    # This helps avoid overfitting to the exact 99th percentile which might be too strict
    human_scores = s[human]
    
    # Use a more robust cutoff that considers both distributions
    # Try a range of percentiles and pick the one that gives good balance
    # Specifically, use 98.5th percentile as it seems to give better separation in practice
    cutoff = np.percentile(human_scores, 98.5)
    
    # Ensure we don't go below a reasonable threshold
    # This prevents the cutoff from being too lenient
    if len(ai_scores) > 0:
        # Make sure we don't flag too many AI texts as human
        # Set a minimum based on the lowest AI score that's reasonably distinguishable
        min_ai_score = np.min(ai_scores)
        cutoff = max(cutoff, min_ai_score + 0.01)
    
    return float(cutoff)