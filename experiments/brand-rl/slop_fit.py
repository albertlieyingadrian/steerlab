"""Freeze Episode 3's two slop probes as plain weights, so the RL reward and the evaluation can run them anywhere.

    .venv/bin/python experiments/brand-rl/slop_fit.py      # laptop; reads Episode 2 and 3 caches read-only

- markup probe (Episode 3's A3_markup): StandardScaler + logistic regression on the badge-masked block-tree
  histogram, C = 0.01 (Episode 3's dev choice). In the reward of the ENV and TEAM arms.
- screenshot probe (Episode 3's A1_siglip): the same on SigLIP embeddings of the masked first screen. Never in any
  reward: an unseen ruler.
Both are refit on exactly Episode 3's dev rows and checked against Episode 3's held-out AUROC before they are saved
(results/slop_probes.json: mean, scale, coef, intercept per probe; no page data).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EP3 = HERE.parent / "slop-probes"
OUT = HERE / "results" / "slop_probes.json"


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    sys.path.insert(0, str(EP3))
    from ep3 import load_own, load_pages

    ep3_probes = json.loads((EP3 / "results" / "probes.json").read_text())
    pages = load_pages()
    masked = load_own("masked.pkl")
    pages = [r for r in pages if r["pid"] in masked and "hand_mask" in masked[r["pid"]]]
    dev = [r for r in pages if r["y_main"] is not None and r["split"] == "dev"]
    held = [r for r in pages if r["y_main"] is not None and r["split"] == "held"]
    y_dev = np.array([r["y_main"] for r in dev])
    y_held = np.array([r["y_main"] for r in held])
    feats = {
        "markup": ("A3_markup", lambda r: masked[r["pid"]]["hand_mask"]["dom"]),
        "screenshot": ("A1_siglip", lambda r: masked[r["pid"]]["sig_mask"]),
    }
    out = {}
    for name, (arm, get) in feats.items():
        C = ep3_probes[arm]["C"]
        Xd = np.stack([np.asarray(get(r), np.float32) for r in dev])
        Xh = np.stack([np.asarray(get(r), np.float32) for r in held])
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000)).fit(Xd, y_dev)
        auc = roc_auc_score(y_held, pipe.decision_function(Xh))
        want = ep3_probes[arm]["held"]["auroc"]
        print(f"{name}: C={C} held-out AUROC {auc:.4f} (Episode 3: {want:.4f})")
        assert abs(auc - want) < 0.005, "refit does not reproduce Episode 3"
        sc, lr = pipe[0], pipe[1]
        out[name] = {
            "episode3_arm": arm,
            "C": C,
            "held_auroc": round(float(auc), 4),
            "n_dev": len(dev),
            "mean": sc.mean_.round(6).tolist(),
            "scale": sc.scale_.round(6).tolist(),
            "coef": lr.coef_[0].round(6).tolist(),
            "intercept": float(lr.intercept_[0]),
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out))
    print("wrote", OUT)


def load_probe(name: str, path: Path = OUT):
    """Returns f(X) -> P(AI-builder) for rows of features (the logistic regression's probability)."""
    p = json.loads(path.read_text())[name]
    mean, scale, coef = (np.asarray(p[k], np.float64) for k in ("mean", "scale", "coef"))

    def f(X):
        z = ((np.asarray(X, np.float64) - mean) / scale) @ coef + p["intercept"]
        return 1.0 / (1.0 + np.exp(-z))

    return f


if __name__ == "__main__":
    main()
