"""Recipe R0+: improved hard-negative round with better mining selection and increased replay.

Mining: stratified sampling of mining pages based on score distribution to include more borderline cases
Replay: increased to 10,000 rows to better preserve learned knowledge
Mirror: small addition of mirror data for diversity
One epoch at learning rate 1e-4. Cutoff: the 99th percentile of the updated detector's scores on data v2b's validation humans.
"""

import json
from pathlib import Path

import numpy as np

HUMAN_W = 1.3278
K_MINING_STRATIFIED = 2000
N_REPLAY = 10000
N_MIRROR = 500


def build(ctx):
    data = Path(ctx["data"])
    scores = np.load(data / "mining_scores_r0.npy")
    
    # Stratified sampling: include both high and medium-high scoring pages
    # Split into quantiles and sample from each
    quantiles = np.quantile(scores, [0.2, 0.4, 0.6, 0.8])
    indices = np.arange(len(scores))
    
    # Sample more evenly across score ranges
    rows = []
    
    # High scoring (top 20%) - keep as is
    high_score_mask = scores >= quantiles[3]
    high_indices = indices[high_score_mask]
    selected_high = np.random.choice(high_indices, min(len(high_indices), 800), replace=False)
    for i in selected_high:
        rows.append({"pool": "mining", "idx": int(i), "bucket": 0, "weight": HUMAN_W})
    
    # Medium-high scoring (20-40% range) - add more samples
    med_high_mask = (scores >= quantiles[2]) & (scores < quantiles[3])
    med_high_indices = indices[med_high_mask]
    selected_med_high = np.random.choice(med_high_indices, min(len(med_high_indices), 600), replace=False)
    for i in selected_med_high:
        rows.append({"pool": "mining", "idx": int(i), "bucket": 0, "weight": HUMAN_W})
        
    # Medium scoring (40-60% range) - add more samples for boundary cases
    med_mask = (scores >= quantiles[1]) & (scores < quantiles[2])
    med_indices = indices[med_mask]
    selected_med = np.random.choice(med_indices, min(len(med_indices), 400), replace=False)
    for i in selected_med:
        rows.append({"pool": "mining", "idx": int(i), "bucket": 0, "weight": HUMAN_W})
    
    # Add some low scoring (60-80% range) to capture harder negatives
    low_med_mask = (scores >= quantiles[0]) & (scores < quantiles[1])
    low_med_indices = indices[low_med_mask]
    selected_low_med = np.random.choice(low_med_indices, min(len(low_med_indices), 200), replace=False)
    for i in selected_low_med:
        rows.append({"pool": "mining", "idx": int(i), "bucket": 0, "weight": HUMAN_W})

    # Replay data - increased to 10,000
    meta = []
    with open(data / "replay.jsonl") as f:
        for line in f:
            r = json.loads(line)
            meta.append((int(r["bucket"]), float(r["weight"])))
    rng = np.random.default_rng(0)
    for i in sorted(rng.choice(len(meta), N_REPLAY, replace=False)):
        b, w = meta[i]
        rows.append({"pool": "replay", "idx": int(i), "bucket": b, "weight": w})
    
    # Mirror data - small addition for diversity
    if N_MIRROR > 0:
        mirror_data = []
        with open(data / "mirrors.jsonl") as f:
            for line in f:
                r = json.loads(line)
                mirror_data.append(r)
        selected_mirrors = rng.choice(len(mirror_data), min(N_MIRROR, len(mirror_data)), replace=False)
        for i in selected_mirrors:
            m = mirror_data[i]
            # Use the mining index to get the original bucket
            mining_idx = m["mining_idx"]
            # Get the original bucket from mining data
            orig_bucket = 0  # All mirrors are assumed to be human-like for this purpose
            rows.append({"pool": "mirrors", "idx": int(i), "bucket": orig_bucket, "weight": 1.0})

    return {"rows": rows, "train": {"lr": 1e-4, "epochs": 1.0}, "note": "R0+: stratified mining + 10k replay + mirror diversity"}


def fit_cutoff(cal):
    s = np.asarray(cal["v2b_val"]["scores"], float)
    human = np.asarray(cal["v2b_val"]["kind"]) == "human"
    return float(np.quantile(s[human], 0.99))