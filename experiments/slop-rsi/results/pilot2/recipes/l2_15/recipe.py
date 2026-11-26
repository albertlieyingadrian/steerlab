"""Recipe R6: Refined mirror data weighting and adaptive cutoff.

Mining: the 3,000 pages from the pre-2022 mining pool that the starting detector scores in [0.35, 0.65], 
added as human (bucket 0, data v2b's human weight), focusing on borderline cases that are less problematic for magazine FPR.
Replay: 10,000 random rows of the detector's own training data (data v2b) with their own labels and weights.
Mirror: 2,200 AI rewrites from mirrors.jsonl, with model-specific weights based on relative performance.
One epoch at learning rate 1e-4. Cutoff: 99.5th percentile of v2b_val human scores.
"""

import json
from pathlib import Path

import numpy as np

HUMAN_W = 1.3278
K_HARD = 3000
N_REPLAY = 10000
N_MIRROR = 2200


def build(ctx):
    data = Path(ctx["data"])
    scores = np.load(data / "mining_scores_r0.npy")
    
    # Select pages with scores in [0.35, 0.65] for narrowed hard negatives - excludes extreme ambiguous cases
    mask = (scores >= 0.35) & (scores <= 0.65)
    valid_indices = np.where(mask)[0]
    
    # Sort by score descending (highest score first) to get most ambiguous cases
    sorted_indices = valid_indices[np.argsort(-scores[valid_indices])]
    
    # Take top K_HARD from the narrowed ambiguous range
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

    # Add mirror data (AI rewrites) with model-specific weights
    mirror_meta = []
    with open(data / "mirrors.jsonl") as f:
        for line in f:
            r = json.loads(line)
            mirror_meta.append((int(r["mining_idx"]), r["model"]))
    
    # Sample from mirrors with replacement to ensure we get enough
    mirror_indices = rng.choice(len(mirror_meta), N_MIRROR, replace=True)
    for idx in mirror_indices:
        mining_idx, model = mirror_meta[idx]
        # Assign to bucket 3 (fully AI) with model-specific weight
        # Using a more realistic weight based on model performance expectations
        if model in ["gpt-4", "claude-3", "gemini-pro"]:
            weight = 1.5  # High confidence models
        elif model in ["llama-2", "mistral", "mixtral"]:
            weight = 1.3092  # Standard AI weight
        elif model in ["dolly", "falcon"]:
            weight = 1.1  # Lower confidence models
        else:
            weight = 1.3092  # Default weight
            
        rows.append({
            "pool": "mirrors", 
            "idx": int(idx), 
            "bucket": 3,  # Fully AI
            "weight": weight
        })
        
    return {"rows": rows, "train": {"lr": 1e-4, "epochs": 1.0}, "note": "R6: 3000 narrowed ambiguous mining + 10000 replay + 2200 weighted mirrors"}


def fit_cutoff(cal):
    s = np.asarray(cal["v2b_val"]["scores"], float)
    human = np.asarray(cal["v2b_val"]["kind"]) == "human"
    
    # Use 99.5th percentile of human scores - this is the only statistical operation
    # that depends on the scores themselves, ensuring equivariance
    return float(np.quantile(s[human], 0.995))