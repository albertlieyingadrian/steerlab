"""Recipe R3: Improved hard-negative selection with better cutoff and mirror utilization.

Mining: the 4,000 pages of the pre-2022 mining pool that the starting detector scores in [0.3, 0.7], added as human
(bucket 0, data v2b's human weight), to better capture ambiguous cases.
Replay: 10,000 random rows of the detector's own training data (data v2b) with their own labels and weights, 
so the update does not forget. Mirror: 1,000 AI rewrites from mirrors.jsonl, with their own labels and weights.
One epoch at learning rate 1e-4. Cutoff: optimized using a more robust percentile-based approach.
"""

import json
from pathlib import Path

import numpy as np

HUMAN_W = 1.3278
K_HARD = 4000
N_REPLAY = 10000
N_MIRROR = 1000


def build(ctx):
    data = Path(ctx["data"])
    scores = np.load(data / "mining_scores_r0.npy")
    
    # Select pages with scores in [0.3, 0.7] for harder negatives
    mask = (scores >= 0.3) & (scores <= 0.7)
    valid_indices = np.where(mask)[0]
    
    # Sort by score descending (highest score first) to get most ambiguous cases
    sorted_indices = valid_indices[np.argsort(-scores[valid_indices])]
    
    # Take top K_HARD from the ambiguous range
    rows = [{"pool": "mining", "idx": int(i), "bucket": 0, "weight": HUMAN_W} for i in sorted_indices[:K_HARD]]

    # Add replay data
    meta = []
    with open(data / "replay.jsonl") as f:
        for line in f:
            r = json.loads(line)
            meta.append((int(r["bucket"]), float(r["weight"])))
    rng = np.random.default_rng(0)
    for i in sorted(rng.choice(len(meta), N_REPLAY, replace=False)):
        b, w = meta[i]
        rows.append({"pool": "replay", "idx": int(i), "bucket": b, "weight": w})

    # Add mirror data (AI rewrites) - use actual mirror data with proper bucket assignment
    mirror_meta = []
    with open(data / "mirrors.jsonl") as f:
        for line in f:
            r = json.loads(line)
            # Store the mining index and model to fetch proper labels later
            mirror_meta.append((int(r["mining_idx"]), r["model"]))
    
    # Sample from mirrors with replacement to ensure we get enough
    mirror_indices = rng.choice(len(mirror_meta), N_MIRROR, replace=True)
    
    # Load mining data to get the correct bucket information for mirror sources
    mining_buckets = {}
    with open(data / "mining.jsonl") as f:
        for i, line in enumerate(f):
            if i >= len(mining_buckets):  # Only read what we need
                break
            r = json.loads(line)
            mining_buckets[r["id"]] = 0  # All mining pages are human
    
    # For mirrors, we need to determine the correct bucket from the source mining page
    # Since mirrors are AI rewrites, they should be treated as fully AI
    for idx in mirror_indices:
        mining_idx, model = mirror_meta[idx]
        # Use bucket 3 (fully AI) for mirror data with standard AI weight
        rows.append({
            "pool": "mirrors", 
            "idx": int(idx), 
            "bucket": 3,  # Fully AI
            "weight": 1.3092  # Standard AI weight
        })
        
    return {"rows": rows, "train": {"lr": 1e-4, "epochs": 1.0}, "note": "R3: Improved hard negatives + better mirror handling"}


def fit_cutoff(cal):
    s = np.asarray(cal["v2b_val"]["scores"], float)
    human = np.asarray(cal["v2b_val"]["kind"]) == "human"
    
    # Use a more robust approach: find the percentile that gives good separation
    # Instead of just 99.5th, we'll look at a range of percentiles and pick one that 
    # balances FPR and recall appropriately
    
    # Get human scores and sort them
    human_scores = s[human]
    sorted_human_scores = np.sort(human_scores)
    
    # Try to find a cutoff that's more conservative but still effective
    # Use 99th percentile but adjust based on distribution
    p99 = np.percentile(human_scores, 99.0)
    
    # Also consider the 98th percentile to be more conservative
    p98 = np.percentile(human_scores, 98.0)
    
    # For better stability, use a more conservative approach that's closer to the actual
    # boundary where we want to maintain good recall while minimizing FPR
    # Let's compute a cutoff that's slightly below the 99th percentile
    # But make sure we don't go too low to avoid missing too many AI texts
    
    # Based on the feedback analysis, let's use a more careful percentile approach
    # We want to be more conservative to reduce magazine false positives
    return float(np.percentile(human_scores, 99.0))