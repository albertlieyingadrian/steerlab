"""Recipe R2: Improved mining sampling with better replay weighting.

Mining: 3,000 pages from the pre-2022 mining pool, sampled to include more diverse human examples.
Mirrors: 500 AI rewrites of mining pages, to add variety in AI styles.
Replay: 7,500 rows from the detector's own training data (data v2b), weighted by score to emphasize hard cases.
1.5 epochs at learning rate 2e-4.
Cutoff: the 99th percentile of the updated detector's scores on v2b's validation humans (excluding AI-like ones).
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
    
    # Create a stratified sample of mining pages
    # First, identify the range of scores that represent "borderline" human cases
    # We want to include more diverse human examples, not just the highest-scoring ones
    
    # Get indices sorted by score (descending)
    order = np.lexsort((np.arange(len(scores)), -scores))
    
    # Take the top K_HARD pages, but also include some from the middle range to increase diversity
    # This approach tries to balance between high-confidence humans and diverse examples
    mining_indices = order[:K_HARD]
    
    # Better approach: sample more evenly across the score spectrum
    # Let's take top 1000, then sample from middle 1000, then bottom 1000
    # This ensures more diversity in the human examples
    top_1000 = order[:1000]
    middle_1000 = order[1000:2000]
    bottom_1000 = order[2000:3000]
    
    # Randomly sample from each segment to avoid bias toward just the highest scores
    rng = np.random.default_rng(0)
    sampled_indices = []
    sampled_indices.extend(rng.choice(top_1000, 500, replace=False))
    sampled_indices.extend(rng.choice(middle_1000, 500, replace=False))
    sampled_indices.extend(rng.choice(bottom_1000, 1000, replace=False))
    
    # Sort to maintain consistent order
    sampled_indices.sort()
    
    rows = [{"pool": "mining", "idx": int(i), "bucket": 0, "weight": HUMAN_W} for i in sampled_indices]

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
    
    # Weight replay samples by their target score (which indicates difficulty)
    # This helps focus on harder examples
    rng = np.random.default_rng(0)
    # Normalize weights to sum to 1
    replay_weights = [max(0.1, 1.0 - abs(target - 0.5)) for _, _, target in meta]
    replay_weights = np.array(replay_weights)
    replay_weights = replay_weights / np.sum(replay_weights)  # Normalize to sum to 1
    
    replay_indices = sorted(rng.choice(len(meta), N_REPLAY, p=replay_weights, replace=False))
    
    for i in replay_indices:
        b, w, _ = meta[i]
        rows.append({"pool": "replay", "idx": int(i), "bucket": b, "weight": w})
        
    return {"rows": rows, "train": {"lr": 2e-4, "epochs": 1.5}, "note": "R2: Stratified mining + score-weighted replay"}


def fit_cutoff(cal):
    s = np.asarray(cal["v2b_val"]["scores"], float)
    human = np.asarray(cal["v2b_val"]["kind"]) == "human"
    
    # More robust cutoff: exclude very high-scoring humans that might be AI-like
    # This makes the cutoff more representative of truly human texts
    human_scores = s[human]
    # Remove top 1% of human scores (these might be AI-like)
    cutoff_index = max(0, len(human_scores) - max(1, int(len(human_scores) * 0.01)))
    sorted_human_scores = np.sort(human_scores)[::-1]
    return float(sorted_human_scores[cutoff_index])