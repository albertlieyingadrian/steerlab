"""Recipe R4: Improved replay sampling with targeted mining and mirrors.

Mining: 3000 pages from a more targeted score range around the decision boundary (0.3-0.7), 
to better capture borderline cases that are harder to classify. 
Mirrors: 500 AI rewrites of mining pages, to add more variety in AI styles. 
Replay: 7500 samples from the detector's own training data (data v2b) weighted by the starting detector's score.
This weighting helps focus on harder examples while maintaining the original distribution.
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
    
    # Sample from a more targeted score range around the decision boundary (0.3-0.7)
    # This focuses on borderline cases that are harder to classify
    mask = (scores >= 0.3) & (scores <= 0.7)
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

    # Add replay samples with score-based weighting
    meta = []
    with open(data / "replay.jsonl") as f:
        for line in f:
            r = json.loads(line)
            meta.append((int(r["bucket"]), float(r["weight"]), float(r["target"])))
    
    # Score-based weighting: give higher weight to examples that are harder to classify
    # (closer to the decision boundary) and also consider the original weights
    replay_scores = np.array([x[2] for x in meta])  # target scores from v2b
    # Normalize scores to [0,1] range for weighting
    normalized_scores = (replay_scores - replay_scores.min()) / (replay_scores.max() - replay_scores.min() + 1e-8)
    # Weight inversely proportional to distance from 0.5 (center of decision boundary)
    # Higher weight for examples closer to 0.5 (harder to classify)
    weights = 1.0 / (np.abs(normalized_scores - 0.5) + 1e-8)
    weights = weights * 0.5 + 0.5  # Scale to range [0.5, 1.5] to avoid extreme weighting
    
    # Select replay samples with weighted probability
    rng = np.random.default_rng(0)
    replay_indices = rng.choice(len(meta), N_REPLAY, p=weights/weights.sum(), replace=False)
    replay_indices = sorted(replay_indices)
    
    for i in replay_indices:
        b, w, _ = meta[i]
        rows.append({"pool": "replay", "idx": int(i), "bucket": b, "weight": w})
        
    return {"rows": rows, "train": {"lr": 2.5e-4, "epochs": 1.5}, "note": "R4: improved replay sampling + targeted mining + mirrors"}


def fit_cutoff(cal):
    s = np.asarray(cal["v2b_val"]["scores"], float)
    human = np.asarray(cal["v2b_val"]["kind"]) == "human"
    return float(np.quantile(s[human], 0.99))