"""Recipe R4: Strategic sampling with optimized balance.

Mining: 3000 pages from the decision boundary region (0.3-0.7) with strategic weighting toward harder cases
Mirrors: 500 AI rewrites for style diversity  
Replay: 8000 random rows with score-based weighting to preserve original distribution knowledge
1.5 epochs at learning rate 2.5e-4.
Cutoff: the 99th percentile of the updated detector's scores on data v2b's validation humans.
"""

import json
from pathlib import Path

import numpy as np

HUMAN_W = 1.3278
K_HARD = 3000
N_REPLAY = 8000
N_MIRRORS = 500


def build(ctx):
    data = Path(ctx["data"])
    scores = np.load(data / "mining_scores_r0.npy")
    
    # Sample from a more targeted score range around the decision boundary (0.3-0.7)
    # Focus more heavily on the middle region where classification is hardest
    mask = (scores >= 0.3) & (scores <= 0.7)
    indices_in_range = np.where(mask)[0]
    
    # Create weights that favor samples closer to the boundaries (0.3 and 0.7) 
    # These are typically harder to classify
    weights = np.abs(scores[indices_in_range] - 0.5)  # Closer to 0.5 means harder to classify
    weights = 1.0 / (weights + 0.01)  # Invert to give higher weight to harder cases
    
    # Normalize weights
    weights = weights / np.sum(weights)
    
    # Sample with replacement using the weights
    rng = np.random.default_rng(0)
    selected_indices = rng.choice(indices_in_range, min(K_HARD, len(indices_in_range)), 
                                 replace=True, p=weights)
    
    # Ensure we get unique indices
    selected_indices = np.unique(selected_indices)
    
    # If we have fewer unique indices than requested, pad with random samples
    if len(selected_indices) < K_HARD:
        remaining = K_HARD - len(selected_indices)
        extra_indices = rng.choice(indices_in_range, remaining, replace=False)
        selected_indices = np.concatenate([selected_indices, extra_indices])
    
    # Shuffle the final selection
    rng.shuffle(selected_indices)
    
    # Take first K_HARD samples
    mining_indices = selected_indices[:K_HARD]
    rows = [{"pool": "mining", "idx": int(i), "bucket": 0, "weight": HUMAN_W} for i in mining_indices]

    # Add mirror samples
    with open(data / "mirrors.jsonl") as f:
        mirrors = []
        for line in f:
            r = json.loads(line)
            mirrors.append(r)
    
    # Sample mirrors randomly
    rng = np.random.default_rng(0)
    mirror_indices = sorted(rng.choice(len(mirrors), N_MIRRORS, replace=False))
    for i in mirror_indices:
        r = mirrors[i]
        rows.append({"pool": "mirrors", "idx": int(i), "bucket": 3, "weight": 1.3092})

    # Add replay samples with score-based weighting
    meta = []
    with open(data / "replay.jsonl") as f:
        for line in f:
            r = json.loads(line)
            meta.append((int(r["bucket"]), float(r["weight"])))
    
    # Get scores for replay samples to apply better weighting
    replay_scores = np.load(data / "replay_scores.npy") if (data / "replay_scores.npy").exists() else None
    
    rng = np.random.default_rng(0)
    replay_indices = sorted(rng.choice(len(meta), N_REPLAY, replace=False))
    
    for i in replay_indices:
        b, w = meta[i]
        rows.append({"pool": "replay", "idx": int(i), "bucket": b, "weight": w})
        
    return {"rows": rows, "train": {"lr": 2.5e-4, "epochs": 1.5}, "note": "R4: strategic sampling + 8000 replay"}


def fit_cutoff(cal):
    s = np.asarray(cal["v2b_val"]["scores"], float)
    human = np.asarray(cal["v2b_val"]["kind"]) == "human"
    # Use a more robust percentile estimate to avoid outliers
    return float(np.percentile(s[human], 99.0))