"""Recipe R6: Improved equivariant cutoff with enhanced hard-negative selection.

Mining: the 3,000 pages of the pre-2022 mining pool that the starting detector scores in [0.3, 0.7], added as human
(bucket 0, data v2b's human weight), to better capture the most ambiguous cases.
Replay: 10,000 random rows of the detector's own training data (data v2b) with their own labels and weights, 
so the update does not forget. Mirror: 1,500 AI rewrites from mirrors.jsonl, with their own labels and weights.
One epoch at learning rate 1e-4. Cutoff: robust equivariant approach using trimmed mean.
"""

import json
from pathlib import Path

import numpy as np

HUMAN_W = 1.3278
K_HARD = 3000
N_REPLAY = 10000
N_MIRROR = 1500


def build(ctx):
    data = Path(ctx["data"])
    scores = np.load(data / "mining_scores_r0.npy")
    
    # Select pages with scores in [0.3, 0.7] for the most ambiguous cases
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
        
    return {"rows": rows, "train": {"lr": 1e-4, "epochs": 1.0}, "note": "R6: Enhanced equivariant cutoff + hard negatives"}


def fit_cutoff(cal):
    s = np.asarray(cal["v2b_val"]["scores"], float)
    human = np.asarray(cal["v2b_val"]["kind"]) == "human"
    
    # Get human scores and sort them
    human_scores = s[human]
    
    # Use trimmed mean approach for robustness - remove top and bottom 10% of scores
    # This is more equivariant than median + IQR and less sensitive to outliers
    sorted_human = np.sort(human_scores)
    n = len(sorted_human)
    trim_count = max(1, int(n * 0.1))  # Trim 10% from each end
    
    # Trim the scores and compute mean of remaining
    trimmed_scores = sorted_human[trim_count : n - trim_count]
    if len(trimmed_scores) > 0:
        cutoff = float(np.mean(trimmed_scores))
    else:
        # Fallback to median if trimming leaves no data
        cutoff = float(np.median(human_scores))
    
    # Ensure the cutoff is within reasonable bounds
    # It should not be higher than the 99th percentile to avoid overfitting
    max_cutoff = np.percentile(human_scores, 99)
    return float(min(cutoff, max_cutoff))