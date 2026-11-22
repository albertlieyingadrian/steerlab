"""
Reward heads (tier 1, "probes") for The Taste Stack Episode 7: Bradley-Terry heads on frozen per-image features.

A head scores an image with a linear function w.x; P(a beats b) = sigmoid(w.(x_a - x_b)). It is an L2 logistic
regression on feature differences with no intercept, trained on both orders of every pair.

Arms (per part; trained on one confirm fold, scored on the other):
  pooled         all five designers of the track
  pooled_c_own   the same at the own head's regularisation strength (post hoc)
  pooled_fifth   a random fifth of the pooled pairs (the per-designer amount of data)
  own            this designer's pairs only
  shrunk         shared + designer offset: features [dx, lam*dx*1{designer=d}], one fit for the cohort
  other          the next designer's own head (cyclic within the cohort), scored on this designer
  gen_prior      four generator one-hots only, per designer (gen_prior_pooled: cohort)
  stats          six image statistics only, per designer
  resid_*        pooled / own / shrunk on features with each generator's mean removed (fit on training images)
  shuffled_*     pooled / own on rankings shuffled within (scene, designer), 10 shuffles
  bright         R2 positive control: a synthetic designer who prefers the brighter image (mean L*)

    python heads.py --stage dev     # choose features, C and lam on dev scenes (inner CV by scene); ledger rows
    python heads.py --stage confirm # fit on fold A -> score B and B -> A with the frozen choices
"""

from __future__ import annotations

import argparse
import json
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from taste import CACHE, HERE, PARTS, load

CONFIG = HERE / "results" / "head_config.json"
LEDGER = HERE / "results.tsv"


def features(feat: dict, emb: str, blocks: tuple[str, ...], basis_ids: list | None = None) -> dict:
    """asset -> raw feature vector (standardised later on training images).
    Blocks: "emb" (full embedding), "pca<k>" (first k principal components of the embedding, basis fitted on
    basis_ids only: the dev scenes' images), "sim" (brief-image similarity), "inter", "stats" (six image statistics)."""
    basis = {}
    for b in blocks:
        if b.startswith("pca"):
            k = int(b[3:])
            ids = sorted(basis_ids)
            M = np.stack([feat["img"][a][f"emb_{emb}"] for a in ids]).astype(np.float64)
            mu = M.mean(0)
            _, _, Vt = np.linalg.svd(M - mu, full_matrices=False)
            basis[b] = (mu, Vt[:k])
    out = {}
    for a, f in feat["img"].items():
        parts = []
        for b in blocks:
            if b == "emb":
                parts.append(f[f"emb_{emb}"])
            elif b.startswith("pca"):
                mu, V = basis[b]
                parts.append((f[f"emb_{emb}"] - mu) @ V.T)
            elif b == "sim":
                parts.append(np.array(f.get(f"sim_{emb}", [0.0, 0.0]), dtype=np.float32) * 10)
            elif b == "inter":
                parts.append(f[f"emb_{emb}"] * f.get(f"t_{emb}", np.zeros_like(f[f"emb_{emb}"])) * 30)
            elif b == "stats":
                parts.append(np.array(f["stats"], dtype=np.float32))
        out[int(a)] = np.concatenate(parts).astype(np.float64)
    return out


class Space:
    """Standardise features on the training images of one fit; optional per-generator mean removal."""

    def __init__(self, X: dict, train_assets, model_ix: dict | None = None):
        ids = sorted(set(train_assets))
        M = np.stack([X[a] for a in ids])
        self.mu, self.sd = M.mean(0), M.std(0) + 1e-6
        self.gmu = None
        if model_ix is not None:
            Z = (M - self.mu) / self.sd
            self.gmu = {g: Z[[model_ix[a] == g for a in ids]].mean(0) for g in {model_ix[a] for a in ids}}
        self.X, self.model_ix = X, model_ix

    def __call__(self, a: int) -> np.ndarray:
        z = (self.X[a] - self.mu) / self.sd
        if self.gmu is not None:
            z = z - self.gmu[self.model_ix[a]]
        return z


def diffs(space, df: pd.DataFrame) -> np.ndarray:
    return np.stack([space(a) - space(b) for a, b in zip(df.a, df.b)]) if len(df) else np.zeros((0, 1))


def fit(D: np.ndarray, y: np.ndarray, C: float):
    Xf = np.concatenate([D, -D])
    yf = np.concatenate([y, 1 - y])
    if len(set(yf)) < 2:
        return None
    m = LogisticRegression(C=C, fit_intercept=False, max_iter=3000)
    m.fit(Xf, yf)
    return m


def predict(m, D: np.ndarray) -> np.ndarray:
    if m is None:
        return np.full(len(D), 0.5)
    return m.predict_proba(D)[:, 1]


def onehot_space(model_ix: dict):
    return lambda a: np.eye(4)[model_ix[a]]


def shuffle_rankings(df: pd.DataFrame, rng) -> pd.DataFrame:
    """Permute the four images' ranks within each (part, scene, designer); pairs stay consistent."""
    out = df.copy()
    ys = []
    for _, g in out.groupby(["dim", "scene", "designer"], sort=False):
        ims = sorted(set(g.a) | set(g.b))
        perm = dict(zip(ims, rng.permutation(len(ims))))
        ys.append(pd.Series([int(perm[a] < perm[b]) for a, b in zip(g.a, g.b)], index=g.index))
    out["y"] = pd.concat(ys).reindex(out.index)
    return out


def run_arms(
    tr: pd.DataFrame,
    te: pd.DataFrame,
    X: dict,
    model_ix: dict,
    cfg: dict,
    stats_X: dict,
    bright: dict,
    rng,
    arms: set | None = None,
) -> dict:
    """Fit every arm on tr (one part, one track), return {arm: p for te rows}."""
    designers = sorted(tr.designer.unique())
    nxt = {d: designers[(i + 1) % len(designers)] for i, d in enumerate(designers)}
    want = (lambda k: True) if arms is None else (lambda k: k in arms)
    tr_assets = set(tr.a) | set(tr.b)
    P = {}
    sp = Space(X, tr_assets)
    Dtr, Dte = diffs(sp, tr), diffs(sp, te)
    ytr = tr.y.to_numpy()
    if want("pooled"):
        P["pooled"] = predict(fit(Dtr, ytr, cfg["C_pooled"]), Dte)
    if want("pooled_c_own"):  # post hoc: the pooled head at the own head's strength (no random draws)
        P["pooled_c_own"] = predict(fit(Dtr, ytr, cfg["C_own"]), Dte)
    if want("pooled_fifth"):
        idx = rng.choice(len(tr), size=len(tr) // len(designers), replace=False)
        P["pooled_fifth"] = predict(fit(Dtr[idx], ytr[idx], cfg["C_own"]), Dte)
    own_models = {}
    for d in designers:
        m = tr.designer.to_numpy() == d
        own_models[d] = fit(Dtr[m], ytr[m], cfg["C_own"])
    ted = te.designer.to_numpy()
    if want("own"):
        P["own"] = np.concatenate([predict(own_models[d], Dte[ted == d]) for d in designers])
    if want("other"):
        P["other"] = np.concatenate([predict(own_models[nxt[d]], Dte[ted == d]) for d in designers])
    order = np.concatenate([np.where(ted == d)[0] for d in designers])  # row order of the per-designer arms

    def shrunk(Dtr_, Dte_, lam):
        k = len(designers)
        di = {d: i for i, d in enumerate(designers)}

        def aug(D, ds):
            A = np.zeros((len(D), D.shape[1] * (k + 1)))
            A[:, : D.shape[1]] = D
            for r, d in enumerate(ds):
                j = di[d] + 1
                A[r, j * D.shape[1] : (j + 1) * D.shape[1]] = lam * D[r]
            return A

        m = fit(aug(Dtr_, tr.designer.to_numpy()), ytr, cfg["C_pooled"])
        return predict(m, aug(Dte_, ted))

    if want("shrunk"):
        P["shrunk"] = shrunk(Dtr, Dte, cfg["lam"])
    # generator prior and stats
    oh = onehot_space(model_ix)
    Gtr, Gte = diffs(oh, tr), diffs(oh, te)
    if want("gen_prior"):
        gp = {}
        for d in designers:
            m = tr.designer.to_numpy() == d
            gp[d] = predict(fit(Gtr[m], ytr[m], 1.0), Gte[ted == d])
        P["gen_prior"] = np.concatenate([gp[d] for d in designers])
    if want("gen_prior_pooled"):
        P["gen_prior_pooled"] = predict(fit(Gtr, ytr, 1.0), Gte)
    if want("stats"):
        ss = Space(stats_X, tr_assets)
        Str, Ste = diffs(ss, tr), diffs(ss, te)
        st = {}
        for d in designers:
            m = tr.designer.to_numpy() == d
            st[d] = predict(fit(Str[m], ytr[m], 1.0), Ste[ted == d])
        P["stats"] = np.concatenate([st[d] for d in designers])
    if want("resid"):
        rs = Space(X, tr_assets, model_ix)
        Rtr, Rte = diffs(rs, tr), diffs(rs, te)
        P["resid_pooled"] = predict(fit(Rtr, ytr, cfg["C_pooled"]), Rte)
        P["resid_own"] = np.concatenate(
            [
                predict(
                    fit(Rtr[tr.designer.to_numpy() == d], ytr[tr.designer.to_numpy() == d], cfg["C_own"]), Rte[ted == d]
                )
                for d in designers
            ]
        )
        P["resid_shrunk"] = shrunk(Rtr, Rte, cfg["lam"])
    if want("shuffled"):
        sp_, so_ = [], []
        for _ in range(cfg.get("n_shuffle", 10)):
            ts = shuffle_rankings(tr, rng)
            ys = ts.y.to_numpy()
            sp_.append(predict(fit(Dtr, ys, cfg["C_pooled"]), Dte))
            so_.append(
                np.concatenate(
                    [
                        predict(
                            fit(Dtr[tr.designer.to_numpy() == d], ys[tr.designer.to_numpy() == d], cfg["C_own"]),
                            Dte[ted == d],
                        )
                        for d in designers
                    ]
                )
            )
        P["shuffled_pooled"] = np.stack(sp_)
        P["shuffled_own"] = np.stack(so_)
    if want("bright"):
        L = {a: bright[a] for a in X}
        yb_tr = np.array([int(L[a] > L[b]) for a, b in zip(tr.a, tr.b)])
        P["bright"] = predict(fit(Dtr, yb_tr, cfg["C_pooled"]), Dte)
        P["bright_y"] = np.array([int(L[a] > L[b]) for a, b in zip(te.a, te.b)])
    P["_order"] = order
    return P


def agreement(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.where(np.abs(p - 0.5) < 1e-9, 0.5, (p > 0.5) == (y == 1))))


def log_row(stage: str, what: str, metric: float, kept: str, note: str) -> None:
    if not LEDGER.exists():
        LEDGER.write_text("time_utc\tstage\twhat\tdev_metric\tcost_usd\tkept\tnote\n")
    with LEDGER.open("a") as f:
        f.write(
            f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\t{stage}\t{what}\t{metric:.4f}\t0\t{kept}\t{note}\n"
        )


def confirm_split(part: pd.DataFrame, train_fold: str, test_fold: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train and test rows for one direction of the cross-fit (A -> B or B -> A). Stops the run if a scene or an
    image sits on both sides: every confirm-fold number rests on the two folds sharing neither."""
    assert {train_fold, test_fold} == {"A", "B"}, (train_fold, test_fold)
    tr = part[part.split == train_fold].reset_index(drop=True)
    te = part[part.split == test_fold].reset_index(drop=True)
    assert not set(tr.scene) & set(te.scene), "a scene is in both confirm folds"
    assert not (set(tr.a) | set(tr.b)) & (set(te.a) | set(te.b)), "an image is in both confirm folds"
    return tr, te


def per_designer_rows(P: dict, te: pd.DataFrame, arm: str) -> np.ndarray:
    """Return arm predictions aligned with te's row order."""
    p = P[arm]
    if arm in ("own", "other", "gen_prior", "stats", "resid_own") or arm.startswith("shuffled_own"):
        out = np.empty_like(p)
        out[..., P["_order"]] = p
        return out
    return p


def dev_cv(pairs, X, model_ix, cfg, stats_X, bright, arms, nfold=4):
    """Inner CV by scene on dev scenes; mean agreement per arm over parts."""
    rng = np.random.default_rng(0)
    scores = {a: [] for a in arms}
    for track, dim in PARTS:
        dv = pairs[(pairs.track == track) & (pairs.dim == dim) & (pairs.split == "dev")]
        scenes = sorted(dv.scene.unique())
        folds = [scenes[i::nfold] for i in range(nfold)]
        for f in folds:
            te = dv[dv.scene.isin(f)].reset_index(drop=True)
            tr = dv[~dv.scene.isin(f)].reset_index(drop=True)
            base = {"pooled", "own", "shrunk", "pooled_fifth", "gen_prior", "stats", "other", "bright"} & set(arms)
            P = run_arms(tr, te, X, model_ix, cfg, stats_X, bright, rng, arms=base | {"own"})
            for a in arms:
                if a == "bright":
                    scores[a].append(agreement(P["bright"], P["bright_y"]))
                else:
                    scores[a].append(agreement(per_designer_rows(P, te, a), te.y.to_numpy()))
    return {a: float(np.mean(v)) for a, v in scores.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["dev", "confirm"], required=True)
    args = ap.parse_args()
    d = load()
    pairs, assets = d["pairs"], d["assets"]
    feat = pickle.loads((CACHE / "features.pkl").read_bytes())
    model_ix = dict(zip(assets.asset_id.astype(int), assets.model_ix))
    stats_X = features(feat, "base", ("stats",))
    from formal import srgb_to_lab

    bright = {int(a): float(srgb_to_lab(f["thumb"].reshape(-1, 3))[:, 0].mean()) for a, f in feat["img"].items()}
    dev_ids = sorted({int(x) for x in pairs[pairs.split == "dev"].a} | {int(x) for x in pairs[pairs.split == "dev"].b})
    if args.stage == "dev":
        # R2 is a gate: a feature set is eligible only if the head learns the brightness designer (>= 0.90, dev CV).
        # Raw embeddings memorise (about 60 training images per part) and fail it; see results.tsv.
        best = None
        for emb in ("so400m", "base"):
            for blocks in (("emb",), ("pca4", "stats"), ("pca8", "stats"), ("pca16", "stats"), ("stats",)):
                X = features(feat, emb, blocks, dev_ids)
                for C in (0.01, 0.1, 1.0):
                    cfg = {"C_pooled": C, "C_own": C, "lam": 1.0}
                    s = dev_cv(pairs, X, model_ix, cfg, stats_X, bright, ["pooled", "bright"])
                    key = {"emb": emb, "blocks": list(blocks), "C_pooled": C}
                    eligible = s["bright"] >= 0.90
                    kept = eligible and (best is None or s["pooled"] > best[0])
                    log_row(
                        "dev",
                        f"heads pooled {emb} {'+'.join(blocks)} C={C}",
                        s["pooled"],
                        "kept" if kept else "discarded",
                        f"R2 brightness {s['bright']:.3f}{'' if eligible else ' (fails R2)'}; inner 4-fold CV by scene on dev",
                    )
                    print(key, round(s["pooled"], 4), "bright", round(s["bright"], 3), flush=True)
                    if kept:
                        best = (s["pooled"], key)
        key = best[1]
        X = features(feat, key["emb"], tuple(key["blocks"]), dev_ids)
        best_own = None
        for C in (0.01, 0.1, 1.0):
            s = dev_cv(
                pairs, X, model_ix, {"C_pooled": key["C_pooled"], "C_own": C, "lam": 1.0}, stats_X, bright, ["own"]
            )
            kept = best_own is None or s["own"] > best_own[0]
            log_row(
                "dev",
                f"heads own C={C}",
                s["own"],
                "kept" if kept else "discarded",
                f"features {key['emb']} {'+'.join(key['blocks'])}",
            )
            print("own", C, round(s["own"], 4), flush=True)
            if kept:
                best_own = (s["own"], C)
        best_lam = None
        for lam in (0.25, 0.5, 1.0, 2.0):
            s = dev_cv(
                pairs,
                X,
                model_ix,
                {"C_pooled": key["C_pooled"], "C_own": best_own[1], "lam": lam},
                stats_X,
                bright,
                ["shrunk"],
            )
            kept = best_lam is None or s["shrunk"] > best_lam[0]
            log_row(
                "dev",
                f"heads shrunk lam={lam}",
                s["shrunk"],
                "kept" if kept else "discarded",
                "offset scale; C = pooled C",
            )
            print("shrunk", lam, round(s["shrunk"], 4), flush=True)
            if kept:
                best_lam = (s["shrunk"], lam)
        cfg = {
            **key,
            "C_own": best_own[1],
            "lam": best_lam[1],
            "n_shuffle": 10,
            "dev": {"pooled": best[0], "own": best_own[0], "shrunk": best_lam[0]},
        }
        CONFIG.write_text(json.dumps(cfg, indent=1))
        print("frozen", cfg)
        return
    cfg = json.loads(CONFIG.read_text())
    X = features(feat, cfg["emb"], tuple(cfg["blocks"]), dev_ids)
    rng = np.random.default_rng(1)
    rows = []
    for track, dim in PARTS:
        part = pairs[(pairs.track == track) & (pairs.dim == dim)]
        for trf, tef in (("A", "B"), ("B", "A")):
            tr, te = confirm_split(part, trf, tef)
            P = run_arms(tr, te, X, model_ix, cfg, stats_X, bright, rng)
            out = te.copy()
            for arm in P:
                if arm.startswith("_"):
                    continue
                v = per_designer_rows(P, te, arm)
                if v.ndim == 2:
                    out[arm] = list(v.T)
                else:
                    out[arm] = v
            # inner CV within the training fold for the router (shrunk head), 5 folds by scene
            scenes = sorted(tr.scene.unique())
            inner = np.full(len(tr), np.nan)
            for k in range(5):
                f = scenes[k::5]
                m = tr.scene.isin(f).to_numpy()
                Pi = run_arms(
                    tr[~m].reset_index(drop=True),
                    tr[m].reset_index(drop=True),
                    X,
                    model_ix,
                    cfg,
                    stats_X,
                    bright,
                    rng,
                    arms={"shrunk", "own"},
                )
                inner[m] = Pi["shrunk"]
            rows.append(out.assign(fold_test=tef))
            tri = tr.copy()
            tri["shrunk_innercv"] = inner
            rows.append(tri.assign(fold_test=f"inner:{trf}"))
            print(
                track,
                dim,
                trf,
                "->",
                tef,
                {
                    a: round(agreement(out[a].to_numpy(), out.y.to_numpy()), 3)
                    for a in ("pooled", "own", "shrunk", "other", "gen_prior")
                    if a in out
                },
                flush=True,
            )
    res = pd.concat(rows, ignore_index=True)
    res.to_pickle(CACHE / "heads_confirm.pkl")
    print("saved", len(res))


if __name__ == "__main__":
    main()
