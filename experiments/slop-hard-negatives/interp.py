"""Step 5: what changes inside, per round. Laptop only (numpy, scikit-learn), from data/feats/<run>.npz.

    python interp.py     # results/interp.json

The state is the detector head's input at the token it reads (final layer, last token, after the final norm); the
head is a 4-way linear map W, and score = expected bucket / 3. For each concept set (dev pages, never trained on):
- corp: pages a judge called polished corporate/marketing copy (1) vs pages it did not (0);
- slop: dev pages (0) vs AI mirrors of the same pages (1), split by pair.
Half A fits the direction (difference of class means) and the probe; half B is scored.
- Decodability (presence): StandardScaler + L2 logistic regression, C by 5-fold CV on A (slop-probes' recipe); AUROC
  on B; a length probe (log words) beside it.
- Reliance (use): mean-ablate the direction in B (remove each row's deviation from B's mean along it), recompute the
  score from the state with W. corp reliance = drop in mean score on B's corporate pages; slop reliance = drop in
  recall at the fixed cut on B's AI pages (and the drop in their mean score). Beside each: 20 random directions drawn
  from B's feature covariance, with the same ablation.
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hn import DATA, FLAG_CUT, RESULTS, auroc

RUNS = ["r0", "r0s1", "a1", "a1s1", "a2", "a3", "a4", "b1", "b2", "b3", "c1", "c2", "c3"]
C_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
N_RANDOM = 20


def score_from(h: np.ndarray, W: np.ndarray) -> np.ndarray:
    z = h @ W.T
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    return (p * np.arange(W.shape[0])).sum(1) / (W.shape[0] - 1)


def ablate(h: np.ndarray, d: np.ndarray, mu: np.ndarray) -> np.ndarray:
    d = d / np.linalg.norm(d)
    return h - np.outer((h - mu) @ d, d)


def probe_auroc(Xa, ya, Xb, yb) -> float:
    gs = GridSearchCV(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        {"logisticregression__C": C_GRID},
        cv=StratifiedKFold(5, shuffle=True, random_state=0),
        scoring="roc_auc",
    )
    gs.fit(Xa, ya)
    return auroc(yb, gs.predict_proba(Xb)[:, 1])


def halves(man: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    sets, ys, ids = np.array(man["set"]), np.array(man["y"]), np.array(man["id"])
    rng = np.random.default_rng(0)
    corp = np.where(sets == "corp")[0]
    a = np.concatenate([rng.permutation(corp[ys[corp] == c])[: (ys[corp] == c).sum() // 2] for c in (0, 1)])
    out = {"corp": (np.sort(a), np.setdiff1d(corp, a))}
    slop = np.where(sets == "slop")[0]
    pair = np.array([i.split("|")[0] for i in ids[slop]])  # a mirror's key starts with its page id
    keys = rng.permutation(sorted(set(pair)))
    ka = set(keys[: len(keys) // 2])
    a = slop[np.array([p in ka for p in pair])]
    out["slop"] = (np.sort(a), np.setdiff1d(slop, a))
    return out


def one_run(f, man: dict, hv: dict) -> dict:
    H, W, S = f["h"].astype(np.float64), f["W"].astype(np.float64), f["score"]
    y, words = np.array(man["y"]), np.log(np.array(man["words"]) + 1)[:, None]
    res = {"head_check": float(np.abs(score_from(H, W) - S).max())}
    w_ai = W[-1] - W[0]
    for c, (A, B) in hv.items():
        d = H[A][y[A] == 1].mean(0) - H[A][y[A] == 0].mean(0)
        mu = H[B].mean(0)
        base = score_from(H[B], W)
        abl = score_from(ablate(H[B], d, mu), W)
        pos = y[B] == 1
        rng = np.random.default_rng(1)
        rand_dirs = rng.multivariate_normal(np.zeros(H.shape[1]), np.cov(H[B].T), size=N_RANDOM, method="eigh")
        r_scores = [score_from(ablate(H[B], r, mu), W) for r in rand_dirs]
        if c == "corp":
            rel = float(base[pos].mean() - abl[pos].mean())
            rand = [float(base[pos].mean() - rs[pos].mean()) for rs in r_scores]
            extra = {
                "fpr_corp_base": float((base[pos] >= FLAG_CUT).mean()),
                "fpr_corp_ablated": float((abl[pos] >= FLAG_CUT).mean()),
                "fpr_noncorp_base": float((base[~pos] >= FLAG_CUT).mean()),
                "mean_score_corp": float(base[pos].mean()),
                "mean_score_noncorp": float(base[~pos].mean()),
            }
        else:
            rel = float((base[pos] >= FLAG_CUT).mean() - (abl[pos] >= FLAG_CUT).mean())
            rand = [float((base[pos] >= FLAG_CUT).mean() - (rs[pos] >= FLAG_CUT).mean()) for rs in r_scores]
            extra = {
                "recall_base": float((base[pos] >= FLAG_CUT).mean()),
                "recall_ablated": float((abl[pos] >= FLAG_CUT).mean()),
                "score_drop_ai": float(base[pos].mean() - abl[pos].mean()),
                "random_score_drop_ai_max": float(max(base[pos].mean() - rs[pos].mean() for rs in r_scores)),
            }
        # Added after reading round 1 (scale-free readout): the score and the recall at a fixed cut saturate once
        # training pushes every human score to ~0, so also read the head's margin, generated logit minus human
        # logit, in units of its standard deviation over half B. The registered numbers above stay as they are.
        m, m_abl = H[B] @ w_ai, ablate(H[B], d, mu) @ w_ai
        sd = float(m.std())
        r_m = [float((m[pos] - ablate(H[B], r, mu)[pos] @ w_ai).mean() / sd) for r in rand_dirs]
        margin = {
            "gap_sd": float((m[pos].mean() - m[~pos].mean()) / sd),  # class 1 minus class 0, in sd units
            "drop_sd": float((m[pos] - m_abl[pos]).mean() / sd),  # how far ablating d lowers class 1's margin
            "gap_after_sd": float((m_abl[pos].mean() - m_abl[~pos].mean()) / sd),
            "random_drop_sd": {"min": min(r_m), "max": max(r_m), "mean": float(np.mean(r_m))},
        }
        proj = H[B] @ (d / np.linalg.norm(d))
        res[c] = {
            "margin": margin,
            "reliance": rel,
            "random": {"min": min(rand), "max": max(rand), "mean": float(np.mean(rand))},
            "probe_auroc": probe_auroc(H[A], y[A], H[B], y[B]),
            "length_auroc": probe_auroc(words[A], y[A], words[B], y[B]),
            "direction_auroc": auroc(y[B], proj),
            "cos_head": float(d @ w_ai / np.linalg.norm(d) / np.linalg.norm(w_ai)),
            "spread_along_dir": float(proj.std()),
            "n_b": len(B),
            **extra,
        }
    res["corp_reliance"], res["slop_reliance"] = res["corp"]["reliance"], res["slop"]["reliance"]
    return res


def main() -> None:
    man = json.loads((RESULTS / "interpset.json").read_text())
    hv = halves(man)
    out = {"n": {c: {"A": len(a), "B": len(b)} for c, (a, b) in hv.items()}, "runs": {}}
    for r in RUNS:
        p = DATA / "feats" / f"{r}.npz"
        if not p.exists():
            continue
        x = out["runs"][r] = one_run(np.load(p), man, hv)
        print(
            f"{r:5} head {x['head_check']:.1e}  corp rel {x['corp_reliance']:+.4f} "
            f"[rand {x['corp']['random']['min']:+.4f},{x['corp']['random']['max']:+.4f}] "
            f"probe {x['corp']['probe_auroc']:.3f}  slop rel {x['slop_reliance']:+.3f} "
            f"[rand {x['slop']['random']['min']:+.3f},{x['slop']['random']['max']:+.3f}] "
            f"probe {x['slop']['probe_auroc']:.3f}  score corp/noncorp "
            f"{x['corp']['mean_score_corp']:.3f}/{x['corp']['mean_score_noncorp']:.3f}"
        )
    (RESULTS / "interp.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
