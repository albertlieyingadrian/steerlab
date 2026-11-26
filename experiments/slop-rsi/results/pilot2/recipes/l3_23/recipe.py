"""Recipe R5: Adjusted cutoff, enhanced mirror samples, and balanced replay/mining.

Adjusted cutoff to 97th percentile for better FPR control.
Added 100 high-confidence AI samples from mirrors (lowest scores).
Increased replay samples to 8500 to maintain distribution.
Maintained mining focus on 0.3-0.7 score range with expanded sample count.
1.5 epochs at learning rate 2.5e-4.
"""

import json
from pathlib import Path

import numpy as np

HUMAN_W = 1.3278
K_HARD = 3500  # Slightly increased to allow for more varied borderline cases
N_REPLAY = 8500  # Increased from 8000
N_MIRRORS = 700  # Increased from 600


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
    
    # Prioritize high-confidence AI samples (lowest scores) from mirrors
    mirror_scores = np.array([r.get("score", 0.5) for r in mirrors])
    # Get indices of lowest-scoring mirrors (high-confidence AI)
    ai_indices = np.argsort(mirror_scores)[:100]
    
    # Add 100 high-confidence AI samples
    for i in ai_indices:
        r = mirrors[i]
        rows.append({"pool": "mirrors", "idx": int(i), "bucket": 3, "weight": 1.3092})

    # Add remaining mirror samples
    remaining_mirror_indices = sorted(rng.choice(len(mirrors), N_MIRRORS - 100, replace=False))
    for i in remaining_mirror_indices:
        r = mirrors[i]
        rows.append({"pool": "mirrors", "idx": int(i), "bucket": 3, "weight": 1.3092})

    # Add replay samples
    meta = []
    with open(data / "replay.jsonl") as f:
        for line in f:
            r = json.loads(line)
            meta.append((int(r["bucket"]), float(r["weight"])))
    rng = np.random.default_rng(0)
    for i in sorted(rng.choice(len(meta), N_REPLAY, replace=False)):
        b, w = meta[i]
        rows.append({"pool": "replay", "idx": int(i), "bucket": b, "weight": w})
        
    return {"rows": rows, "train": {"lr": 2.5e-4, "epochs": 1.5}, "note": "R5: 97th cutoff + 100 high-conf AI mirrors + 8500 replay"}


def fit_cutoff(cal):
    s = np.asarray(cal["v2b_val"]["scores"], float)
    human = np.asarray(cal["v2b_val"]["kind"]) == "human"
    # Use 97th percentile instead of 95th to reduce false positives even more
    return float(np.quantile(s[human], 0.97))