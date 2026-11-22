"""R2 on dev scenes: the head learns a synthetic brightness designer (>= 0.90) and sits at chance on shuffled rankings."""

import json
import pickle

import numpy as np

from formal import srgb_to_lab
from heads import CONFIG, agreement, features, per_designer_rows, run_arms
from taste import CACHE, HERE, PARTS, load


def main():
    d = load()
    pairs, assets = d["pairs"], d["assets"]
    feat = pickle.loads((CACHE / "features.pkl").read_bytes())
    model_ix = dict(zip(assets.asset_id.astype(int), assets.model_ix))
    stats_X = features(feat, "base", ("stats",))
    bright = {int(a): float(srgb_to_lab(f["thumb"].reshape(-1, 3))[:, 0].mean()) for a, f in feat["img"].items()}
    cfg = json.loads(CONFIG.read_text())
    dev_ids = sorted({int(x) for x in pairs[pairs.split == "dev"].a} | {int(x) for x in pairs[pairs.split == "dev"].b})
    X = features(feat, cfg["emb"], tuple(cfg["blocks"]), dev_ids)
    rng = np.random.default_rng(2)
    b, sp, so = [], [], []
    for track, dim in PARTS:
        dv = pairs[(pairs.track == track) & (pairs.dim == dim) & (pairs.split == "dev")]
        scenes = sorted(dv.scene.unique())
        for k in range(4):
            f = scenes[k::4]
            te = dv[dv.scene.isin(f)].reset_index(drop=True)
            tr = dv[~dv.scene.isin(f)].reset_index(drop=True)
            P = run_arms(tr, te, X, model_ix, cfg, stats_X, bright, rng, arms={"bright", "shuffled", "own"})
            b.append(agreement(P["bright"], P["bright_y"]))
            sp += [agreement(p, te.y.to_numpy()) for p in P["shuffled_pooled"]]
            so += [agreement(p, te.y.to_numpy()) for p in per_designer_rows(P, te, "shuffled_own")]
    out = {
        "bright_agreement": float(np.mean(b)),
        "shuffled_pooled_mean": float(np.mean(sp)),
        "shuffled_own_mean": float(np.mean(so)),
        "shuffled_pooled_range": [float(np.min(sp)), float(np.max(sp))],
    }
    out["pass"] = (
        out["bright_agreement"] >= 0.90
        and 0.45 <= out["shuffled_pooled_mean"] <= 0.55
        and 0.45 <= out["shuffled_own_mean"] <= 0.55
    )
    (HERE / "results" / "rulers_heads_dev.json").write_text(json.dumps(out, indent=1))
    print(out)


if __name__ == "__main__":
    main()
