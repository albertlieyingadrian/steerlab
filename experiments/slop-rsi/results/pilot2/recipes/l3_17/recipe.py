"""Recipe R4: Improved targeted mining with weighted replay sampling.

Mining: 3000 pages from a targeted score range around the decision boundary (0.25-0.75), 
to better capture borderline cases that are harder to classify. 
Mirrors: 500 AI rewrites of mining pages, to add more variety in AI styles. 
Replay: 7500 random rows of the detector's own training data (data v2b) with their 
own labels and weights, weighted by the starting detector's score to emphasize hard negatives.
1.5 epochs at learning rate 2.5e-4.
Cutoff: the 99th percentile of the updated detector's scores on data v2b's validation humans.
"""

import json
from pathlib import Path

import numpy as np

HUMAN_W = 1.3278
K_HARD = 3000
N_REPLAY = 7500
N_MIRRORS = 500


def build(ctx):
    data = Path(ctx["data"])
    scores = np.load(data / "mining_scores_r0.npy")
    
    # Sample from a slightly broader targeted score range (0.25-0.75) to capture more borderline cases
    # This is still focused on the decision boundary but with more coverage
    mask = (scores >= 0.25) & (scores <= 0.75)
    indices_in_range = np.where(mask)[0]
    
    # Shuffle and take first K_HARD samples
    rng = np.random.default_rng(0)
    selected_indices = rng.choice(indices_in_range, min(K_HARD, len(indices_in_range)), replace=False)
    rng.shuffle(selected_indices)
    
    # Take first K_HARD samples (or all if less than K_HARD)
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

    # Add replay samples with score-weighted sampling
    # This improves negatives by emphasizing harder-to-classify examples
    meta = []
    with open(data / "replay.jsonl") as f:
        for line in f:
            r = json.loads(line)
            meta.append((int(r["bucket"]), float(r["weight"]), float(r["target"])))
    
    # Weight replay samples by the starting detector's score on the target
    # Higher scores (closer to 1) indicate more AI-like content, which makes good negatives
    replay_scores = np.array([x[2] for x in meta])
    
    # Normalize scores to [0,1] for weighting (higher = more AI-like, better negatives)
    # But we want to sample more from the harder-to-classify region (around 0.5)
    # So we weight inversely to make sure we get a good mix
    weights = 1.0 / (1e-6 + np.abs(replay_scores - 0.5))  # Emphasize samples near 0.5
    
    # Sample with replacement to ensure we get enough samples
    rng = np.random.default_rng(0)
    replay_indices = rng.choice(len(meta), N_REPLAY, p=weights/weights.sum(), replace=True)
    
    for i in sorted(replay_indices):
        b, w, _ = meta[i]
        rows.append({"pool": "replay", "idx": int(i), "bucket": b, "weight": w})
        
    return {"rows": rows, "train": {"lr": 2.5e-4, "epochs": 1.5}, "note": "R4: improved targeted mining + weighted replay"}


def fit_cutoff(cal):
    s = np.asarray(cal["v2b_val"]["scores"], float)
    human = np.asarray(cal["v2b_val"]["kind"]) == "human"
    return float(np.quantile(s[human], 0.99))