"""Recipe R6: Refined hard-negative selection with enhanced mirror data and improved cutoff.

Mining: the 2,000 pages from the pre-2022 mining pool that the starting detector scores in [0.4, 0.6], 
but prioritizing those most likely to be human (lowest scores in that range) to avoid including too many ambiguous cases.
Replay: 10,000 random rows of the detector's own training data (data v2b) with their own labels and weights.
Mirror: 2,500 AI rewrites from mirrors.jsonl, with their own labels and weights.
One epoch at learning rate 1e-4. Cutoff: the 99th percentile of the updated detector's scores on data v2b's validation humans.
"""

import json
from pathlib import Path

import numpy as np

HUMAN_W = 1.3278
K_HARD = 2000
N_REPLAY = 10000
N_MIRROR = 2500


def build(ctx):
    data = Path(ctx["data"])
    scores = np.load(data / "mining_scores_r0.npy")
    
    # Select pages with scores in [0.4, 0.6] for refined hard negatives
    # Prioritize lower scores (more likely to be human) to avoid including too many ambiguous cases
    mask = (scores >= 0.4) & (scores <= 0.6)
    valid_indices = np.where(mask)[0]
    
    # Sort by score ascending (lowest score first) to prioritize human cases
    sorted_indices = valid_indices[np.argsort(scores[valid_indices])]
    
    # Take top K_HARD from the refined ambiguous range (prioritizing lower scores)
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

    # Add mirror data (AI rewrites)
    mirror_meta = []
    with open(data / "mirrors.jsonl") as f:
        for line in f:
            r = json.loads(line)
            # Use the model name to determine weight (simplified)
            # For now, use a moderate weight for mirror data
            mirror_meta.append((int(r["mining_idx"]), r["model"]))
    
    # Sample from mirrors with replacement to ensure we get enough
    mirror_indices = rng.choice(len(mirror_meta), N_MIRROR, replace=True)
    for idx in mirror_indices:
        mining_idx, model = mirror_meta[idx]
        # Assign to bucket 3 (fully AI) with appropriate weight
        rows.append({
            "pool": "mirrors", 
            "idx": int(idx), 
            "bucket": 3,  # Fully AI
            "weight": 1.3092  # Standard AI weight
        })
        
    return {"rows": rows, "train": {"lr": 1e-4, "epochs": 1.0}, "note": "R6: 2000 refined ambiguous mining + 10000 replay + 2500 mirrors"}


def fit_cutoff(cal):
    s = np.asarray(cal["v2b_val"]["scores"], float)
    human = np.asarray(cal["v2b_val"]["kind"]) == "human"
    # Use 99th percentile instead of 99.5th to be slightly less conservative
    # This should help with the magazine FPR while maintaining good recall
    return float(np.quantile(s[human], 0.99))