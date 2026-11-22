"""
TASTE (huggingface.co/datasets/purvanshi/TASTE, MIT) as pairs, scenes and splits for The Taste Stack Episode 7.

A scene is one brief with its four images (one per text-to-image model); every designer of a track ranks those four
images on every criterion ("part") of that track. Each ranking becomes its 6 pairs. Splits are by scene, within
track, from a hash of the scene's smallest asset id: 20 dev scenes per track, the rest in confirm folds A and B.

    python taste.py        # download the tables (not the images), print counts and the R1 check
"""

from __future__ import annotations

import hashlib
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = Path.home() / ".cache" / "steerlab" / "taste-ep7"
RAW = CACHE / "raw"
HF_REPO = "purvanshi/TASTE"
HF_REV = "731a7f588d433214c6d864d2e9f47978d91aed6b"
TABLES = ("prompts", "assets", "rankings", "evaluators")
EXCLUDE_PROMPTS = {213, 217}  # 8 images per ranking (plan, "Exclusion fixed now")
N_DEV = 20
PARTS = [
    ("aesthetics", "preference"),
    ("aesthetics", "typography"),
    ("aesthetics", "color_harmony"),
    ("aesthetics", "mood_and_color_tone"),
    ("aesthetics", "visual_hierarchy"),
    ("descriptions", "preference"),
    ("descriptions", "typography"),
    ("descriptions", "color_accuracy"),
    ("descriptions", "spatial_accuracy"),
]
PART_LABEL = {
    ("aesthetics", "preference"): "A · overall",
    ("aesthetics", "typography"): "A · typography",
    ("aesthetics", "color_harmony"): "A · colour harmony",
    ("aesthetics", "mood_and_color_tone"): "A · mood & tone",
    ("aesthetics", "visual_hierarchy"): "A · hierarchy",
    ("descriptions", "preference"): "B · overall",
    ("descriptions", "typography"): "B · text rendering",
    ("descriptions", "color_accuracy"): "B · colour accuracy",
    ("descriptions", "spatial_accuracy"): "B · layout accuracy",
}
# The dataset card's one-line definitions, shown to every judge.
CRITERION = {
    "preference": "Overall preference: which design you would rather deliver for this brief.",
    "typography": "Typography: the quality and legibility of the typesetting.",
    "color_harmony": "Colour harmony: whether the colours work together.",
    "mood_and_color_tone": "Mood and colour tone: whether the atmosphere and palette match the brief.",
    "visual_hierarchy": "Visual hierarchy: whether the eye is guided correctly through the design.",
    "color_accuracy": "Colour accuracy: faithfulness to the colours the brief specifies.",
    "spatial_accuracy": "Spatial accuracy: whether the layout and placement of elements match the brief.",
}
TABLE3_PAIR_TAU = {  # TASTE paper, Table 3, mean pairwise Kendall tau
    ("aesthetics", "preference"): 0.159,
    ("aesthetics", "mood_and_color_tone"): 0.147,
    ("aesthetics", "visual_hierarchy"): 0.128,
    ("aesthetics", "color_harmony"): 0.103,
    ("aesthetics", "typography"): 0.119,
    ("descriptions", "preference"): 0.163,
    ("descriptions", "color_accuracy"): 0.144,
    ("descriptions", "spatial_accuracy"): 0.182,
    ("descriptions", "typography"): 0.224,
}


def download() -> None:
    from huggingface_hub import hf_hub_download

    RAW.mkdir(parents=True, exist_ok=True)
    for t in TABLES:
        if not (RAW / f"{t}.parquet").exists():
            hf_hub_download(HF_REPO, f"{t}.parquet", repo_type="dataset", revision=HF_REV, local_dir=RAW)


def _h(x) -> int:
    return int(hashlib.sha256(f"taste-ep7:{x}".encode()).hexdigest()[:12], 16)


def designer_ids(r: pd.DataFrame) -> dict:
    """Fixed public relabelling: aesthetics designers D1-D5, descriptions D6-D10, in evaluator-id order."""
    out, k = {}, 1
    for track in ("aesthetics", "descriptions"):
        for e in sorted(r[r.track == track].evaluator_id.unique()):
            out[e] = f"D{k}"
            k += 1
    return out


def load() -> dict:
    """Return rankings, scenes (with split), pairs and assets as DataFrames."""
    download()
    p = pd.read_parquet(RAW / "prompts.parquet")
    a = pd.read_parquet(RAW / "assets.parquet")
    r = pd.read_parquet(RAW / "rankings.parquet")
    r = r[~r.prompt_id.isin(EXCLUDE_PROMPTS)].copy()
    for c in ("prompt_id", "asset_id", "rank"):
        r[c] = r[c].astype(int)
    scene_of = r.groupby("prompt_id").asset_id.min().rename("scene")
    r = r.merge(scene_of, left_on="prompt_id", right_index=True)
    r["designer"] = r.evaluator_id.map(designer_ids(r))
    brief = p.set_index("prompt_id").prompt_text
    sc = r.groupby(["track", "scene"]).agg(prompt_id=("prompt_id", "min")).reset_index()
    sc["brief"] = sc.prompt_id.map(brief)
    sc["assets"] = sc.scene.map(r.groupby("scene").asset_id.apply(lambda s: tuple(sorted(set(s)))))
    sc["h"] = sc.scene.map(_h)
    sc = sc.sort_values(["track", "h"]).reset_index(drop=True)
    sc["split"] = ""
    for track, g in sc.groupby("track"):
        idx = g.index.tolist()
        sc.loc[idx[:N_DEV], "split"] = "dev"
        rest = idx[N_DEV:]
        sc.loc[rest[0::2], "split"] = "A"
        sc.loc[rest[1::2], "split"] = "B"
    r = r.merge(sc[["scene", "split"]], on="scene")
    # pairs: one row per (designer, part, scene, image pair), a < b by asset id; y = 1 if a ranked above b
    rows = []
    for (track, dim, scene, des), g in r.groupby(["track", "dimension", "scene", "designer"]):
        rk = dict(zip(g.asset_id, g["rank"]))
        assert len(rk) == 4 and sorted(rk.values()) == [1, 2, 3, 4], (track, dim, scene, des)
        for i, j in itertools.combinations(sorted(rk), 2):
            rows.append((track, dim, scene, des, i, j, int(rk[i] < rk[j])))
    pairs = pd.DataFrame(rows, columns=["track", "dim", "scene", "designer", "a", "b", "y"])
    pairs = pairs.merge(sc[["scene", "split"]], on="scene")
    a["model_ix"] = a.model.map({m: k for k, m in enumerate(sorted(a.model.unique()))})
    return {"rankings": r, "scenes": sc, "pairs": pairs, "assets": a}


def pair_tau(r: pd.DataFrame) -> dict:
    from scipy.stats import kendalltau

    out = {}
    for (track, dim), g in r.groupby(["track", "dimension"]):
        taus = []
        for _, g2 in g.groupby("scene"):
            piv = g2.pivot_table(index="asset_id", columns="evaluator_id", values="rank")
            for e1, e2 in itertools.combinations(piv.columns, 2):
                taus.append(kendalltau(piv[e1], piv[e2]).statistic)
        out[(track, dim)] = float(np.mean(taus))
    return out


if __name__ == "__main__":
    d = load()
    sc, pairs = d["scenes"], d["pairs"]
    print(sc.groupby(["track", "split"]).size())
    print("pairs", len(pairs), pairs.groupby("split").size().to_dict())
    tau = pair_tau(d["rankings"])
    worst = 0.0
    for k, v in tau.items():
        diff = v - TABLE3_PAIR_TAU[k]
        worst = max(worst, abs(diff))
        print(f"R1 {k}: {v:.3f} vs paper {TABLE3_PAIR_TAU[k]:.3f} ({diff:+.3f})")
    print(f"R1 {'PASS' if worst <= 0.01 else 'FAIL'} (max |diff| {worst:.3f})")
