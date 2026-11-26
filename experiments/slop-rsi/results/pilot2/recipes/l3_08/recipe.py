"""Recipe R3: Enhanced mining diversity with score-weighted replay sampling.

Mining: 3500 pages from a broader score range in the pre-2022 mining pool, added as human
(bucket 0, data v2b's human weight), to capture more diverse human examples. Mirrors: 500 AI rewrites of mining pages,
to add more variety in AI styles. Replay: 7500 rows sampled from data v2b's training data with weights inversely proportional
to the starting detector's confidence (i.e., higher weight for borderline cases), to better focus on difficult examples.
1.5 epochs at learning rate 2.5e-4. Cutoff: the 99th percentile of the updated detector's scores on data v2b's validation humans.
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
    
    # Sample from a broader range of mining scores rather than just top ones
    # Use a stratified approach: pick from top 30% (high confidence human), 
    # middle 40% (borderline cases), bottom 30% (low confidence human)
    order = np.argsort(scores)  # lowest score first
    n_top = int(len(scores) * 0.3)
    n_middle = int(len(scores) * 0.4)
    n_bottom = len(scores) - n_top - n_middle
    
    # Select indices from each range
    top_indices = order[-n_top:]  # highest scores (likely human)
    middle_indices = order[n_top:n_top+n_middle]  # middle scores (borderline)
    bottom_indices = order[:n_bottom]  # lowest scores (likely human)
    
    # Combine and shuffle
    selected_indices = np.concatenate([top_indices, middle_indices, bottom_indices])
    rng = np.random.default_rng(0)
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
            meta.append((int(r["bucket"]), float(r["weight"]), float(r["target"])))
    
    # Score-based weights: inverse of confidence (lower score = higher weight)
    # Weights are adjusted so that the total weight is preserved
    targets = np.array([m[2] for m in meta])
    # Normalize targets to [0, 1] for easier handling
    normalized_targets = (targets - targets.min()) / (targets.max() - targets.min() + 1e-8)
    # Inverse of confidence as weight
    weights = 1.0 / (normalized_targets + 1e-8)
    # Normalize weights to have similar magnitude to original weights
    weights = weights * np.mean([m[1] for m in meta]) / np.mean(weights)
    
    # Sample with replacement based on weights
    rng = np.random.default_rng(0)
    replay_indices = rng.choice(len(meta), N_REPLAY, p=weights/weights.sum(), replace=True)
    
    for i in replay_indices:
        b, w, _ = meta[i]
        rows.append({"pool": "replay", "idx": int(i), "bucket": b, "weight": w})
        
    return {"rows": rows, "train": {"lr": 2.5e-4, "epochs": 1.5}, "note": "R3: enhanced mining + score-weighted replay"}


def fit_cutoff(cal):
    s = np.asarray(cal["v2b_val"]["scores"], float)
    human = np.asarray(cal["v2b_val"]["kind"]) == "human"
    return float(np.quantile(s[human], 0.99))