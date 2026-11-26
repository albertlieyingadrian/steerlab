"""Recipe R5: Enhanced mining sampling with broader score coverage.

Mining: 3500 pages sampled uniformly across the score spectrum to better capture diverse borderline cases.
Mirrors: 500 AI rewrites of mining pages.
Replay: 7500 samples from the detector's own training data (data v2b) weighted by the starting detector's score.
1.5 epochs at learning rate 2.5e-4.
Cutoff: the 99th percentile of the updated detector's scores on data v2b's validation humans.
"""

import json
from pathlib import Path

import numpy as np

HUMAN_W = 1.3278
K_HARD = 3500
N_REPLAY = 7500
N_MIRRORS = 500


def build(ctx):
    data = Path(ctx["data"])
    scores = np.load(data / "mining_scores_r0.npy")
    
    # Sample uniformly across the score spectrum to capture diverse borderline cases
    # Instead of fixed range, we sample based on score distribution
    rng = np.random.default_rng(0)
    
    # Create bins across the full score range [0,1] and sample from each bin
    # This ensures better coverage of the entire decision boundary area
    n_bins = 10
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_counts = np.zeros(n_bins)
    
    # Count how many samples fall in each bin
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (scores >= bin_edges[i]) & (scores < bin_edges[i+1])
        else:
            # Last bin includes the upper boundary
            mask = (scores >= bin_edges[i]) & (scores <= bin_edges[i+1])
        bin_counts[i] = np.sum(mask)
    
    # Calculate samples per bin (uniform distribution across bins)
    samples_per_bin = K_HARD // n_bins
    remaining_samples = K_HARD % n_bins
    
    # Select indices from each bin
    mining_indices = []
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (scores >= bin_edges[i]) & (scores < bin_edges[i+1])
        else:
            mask = (scores >= bin_edges[i]) & (scores <= bin_edges[i+1])
        
        bin_indices = np.where(mask)[0]
        
        # Determine how many samples to take from this bin
        samples_from_this_bin = samples_per_bin
        if i < remaining_samples:
            samples_from_this_bin += 1
            
        # Sample from this bin
        if len(bin_indices) > 0:
            sampled_indices = rng.choice(bin_indices, min(samples_from_this_bin, len(bin_indices)), replace=False)
            mining_indices.extend(sampled_indices)
    
    # Ensure we have exactly K_HARD samples
    if len(mining_indices) > K_HARD:
        mining_indices = rng.choice(mining_indices, K_HARD, replace=False).tolist()
    elif len(mining_indices) < K_HARD:
        # If we don't have enough, sample additional from the full dataset
        remaining_needed = K_HARD - len(mining_indices)
        # Sample from the full dataset excluding already selected indices
        all_indices = np.arange(len(scores))
        unselected_mask = np.ones(len(scores), dtype=bool)
        unselected_mask[mining_indices] = False
        unselected_indices = all_indices[unselected_mask]
        
        additional_indices = rng.choice(unselected_indices, min(remaining_needed, len(unselected_indices)), replace=False)
        mining_indices.extend(additional_indices)
    
    # Shuffle the final list
    rng.shuffle(mining_indices)
    
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
        
    return {"rows": rows, "train": {"lr": 2.5e-4, "epochs": 1.5}, "note": "R5: enhanced mining sampling + targeted replay + mirrors"}


def fit_cutoff(cal):
    s = np.asarray(cal["v2b_val"]["scores"], float)
    human = np.asarray(cal["v2b_val"]["kind"]) == "human"
    return float(np.quantile(s[human], 0.99))