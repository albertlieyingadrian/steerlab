"""
Probes, judges, shortcut checks and combinations for The Taste Stack Episode 3. Laptop only (numpy, scikit-learn).

    .venv/bin/python experiments/slop-probes/probes.py --stage dev    # dev-only numbers (tuning; results/dev/)
    .venv/bin/python experiments/slop-probes/probes.py --stage held   # the held-out run, once (results/)

Every probe is StandardScaler + L2 logistic regression, C chosen by 5-fold CV on its training set (grid 1e-3..1e2).
Main task: AI-builder (1) vs human (0); trained on dev, scored on held-out. Known-answer set: held templates and
held open-model pages (1) vs curated-library sites (0), never used for any choice.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ep3 import CACHE, RESULTS, load_own, load_pages

C_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
B = 2000


# ------------------------------------------------------------------ metrics
def auroc(y, s) -> float:
    y, s = np.asarray(y), np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]
    allv = np.concatenate([pos, neg])
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    ranks = allv.argsort().argsort().astype(float) + 1
    ranks = (np.bincount(inv, weights=ranks) / counts)[inv]
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def boot_idx(y, b=B, seed=0):
    """Stratified bootstrap resamples (indices), the same for every arm so differences are paired."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    return [np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(b)]


def with_ci(y, s, idx) -> dict:
    vals = np.array([auroc(y[i], s[i]) for i in idx])
    return {"auroc": auroc(y, s), "ci": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]}


def paired_diff(y, s1, s2, idx) -> dict:
    d = np.array([auroc(y[i], s1[i]) - auroc(y[i], s2[i]) for i in idx])
    return {
        "diff": auroc(y, s1) - auroc(y, s2),
        "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
        "p_le_0": float((d <= 0).mean()),
    }


def within_industry(y, s, cats) -> float:
    vals = [auroc(y[cats == c], s[cats == c]) for c in sorted(set(cats)) if len(set(y[cats == c])) == 2]
    return float(np.mean(vals))


# ------------------------------------------------------------------ features
def geometry(h: dict) -> np.ndarray:
    sp = np.nan_to_num(np.asarray(h["spacing"], np.float32), nan=0.0)
    pal = np.concatenate([np.asarray(h["pal_c"], np.float32).ravel() / 100.0, np.asarray(h["pal_w"], np.float32)])
    typ = np.array([h["type_base"], h["type_ratio"]], np.float32)
    return np.concatenate([np.asarray(h["layout"], np.float32), sp, pal, typ])


def page_stats(r: dict) -> np.ndarray:
    return np.log1p([r["n_words"], r["n_blocks"], r["doc_h"], r.get("n_req") or 0]).astype(np.float32)


BLOCKS = {
    "sig_mask": lambda r, m: m["sig_mask"],
    "uic_mask": lambda r, m: m["uic_mask"],
    "sig_raw": lambda r, m: m["sig_raw"],
    "uic_raw": lambda r, m: m["uic_raw"],
    "sig_corner": lambda r, m: m["sig_corner"],
    "markup": lambda r, m: m["hand_mask"]["dom"],
    "markup_raw": lambda r, m: r["dom"],
    "geometry": lambda r, m: geometry(m["hand_mask"]),
    "stats": lambda r, m: page_stats(r),
}
ARMS = {  # name -> feature blocks (concatenated)
    "A1_siglip": ["sig_mask"],
    "A2_uiclip": ["uic_mask"],
    "A3_markup": ["markup"],
    "A4_geometry": ["geometry"],
    "C1_siglip_markup": ["sig_mask", "markup"],
    "S_siglip_unmasked": ["sig_raw"],
    "S_uiclip_unmasked": ["uic_raw"],
    "S_markup_unmasked": ["markup_raw"],
    "S_corner_only": ["sig_corner"],
    "B_page_stats": ["stats"],
}
PROBES = ["A1_siglip", "A2_uiclip", "A3_markup", "A4_geometry"]


def X_of(rows, masked, blocks) -> np.ndarray:
    return np.stack(
        [np.concatenate([np.asarray(BLOCKS[b](r, masked[r["pid"]]), np.float32) for b in blocks]) for r in rows]
    )


def fit(X, y, seed=0):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    gs = GridSearchCV(pipe, {"logisticregression__C": C_GRID}, cv=cv, scoring="roc_auc", n_jobs=-1)
    gs.fit(X, y)
    return gs


def oof_scores(gs, X, y, seed=0):
    """Out-of-fold dev scores with the chosen C (for honest dev z-scoring and dev AUROC)."""
    c = gs.best_params_["logisticregression__C"]
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return cross_val_predict(pipe, X, y, cv=cv, method="decision_function")


# ------------------------------------------------------------------ judges
def load_judges() -> dict:
    out = {}
    for path in sorted((CACHE / "judge").glob("*.jsonl")):
        d = {}
        for line in path.read_text().splitlines():
            r = json.loads(line)
            d[(r["pid"], r["mask"], r["prompt"])] = r
        out[path.stem] = d
    return out


def judge_scores(j, rows, mask, prompt):
    return np.array([j[(r["pid"], mask, prompt)]["score"] if (r["pid"], mask, prompt) in j else np.nan for r in rows])


# ------------------------------------------------------------------ constitution data
def constitution_rows():
    meta = {}
    p = CACHE / "cst_meta.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            r = json.loads(line)
            meta[r["pid"]] = r
    rows = []
    for pid, f in load_own("constitution.pkl").items():
        m = meta[pid]
        ok = (
            m["finished"] != "length"
            and f.get("n_words", 0) >= 15
            and float(np.asarray(f["layout"], np.float32).sum()) > 0
            and (f.get("pixel_std") or 0) >= 8
        )
        rows.append({**f, **m, "keep": ok, "y": int(m["side"] == "slop")})
    return rows


CST_BLOCKS = {
    "sig_mask": lambda r: r["sig_mask"],
    "uic_mask": lambda r: r["uic_mask"],
    "geometry": lambda r: geometry(r["hand_mask"]),
    "markup": lambda r: r["hand_mask"]["dom"],
}


def cst_X(rows, blocks):
    return np.stack([np.concatenate([np.asarray(CST_BLOCKS[b](r), np.float32) for b in blocks]) for r in rows])


# ------------------------------------------------------------------ repetition baseline (unsupervised, Ep 2)
def repetition(rows, masked, k=10) -> np.ndarray:
    """Mean similarity to the k most similar pages from other industries, block-tree space (no labels)."""
    X = np.stack([np.asarray(masked[r["pid"]]["hand_mask"]["dom"], np.float32) for r in rows])
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    S = X @ X.T
    cats = np.array([r["category"] for r in rows])
    return np.array([np.sort(S[i, cats != cats[i]])[-k:].mean() for i in range(len(rows))])


def rep_stats(rs, key):
    """Repetition of a set (STYLE rule): mean pairwise similarity and mean nearest other-industry similarity."""
    X = np.stack([np.asarray(key(r), np.float32) for r in rs])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    S = X @ X.T
    cats = np.array([r["category"] for r in rs])
    iu = np.triu_indices(len(rs), 1)
    nn = [np.sort(S[i, cats != cats[i]])[-1] for i in range(len(rs))]
    return {"mean_pair_sim": float(S[iu].mean()), "mean_nn_other_industry": float(np.mean(nn)), "n": len(rs)}


def save(name, obj, stage):
    d = RESULTS if stage == "held" else RESULTS / "dev"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps(obj, indent=1))


# ------------------------------------------------------------------ stages
def run(stage: str, judge_prompt: dict):  # noqa: C901, PLR0915 - one linear pipeline, kept in order of the plan
    pages = load_pages()
    masked = load_own("masked.pkl")
    pages = [r for r in pages if r["pid"] in masked and "hand_mask" in masked[r["pid"]]]
    dev = [r for r in pages if r["y_main"] is not None and r["split"] == "dev"]
    held = [r for r in pages if r["y_main"] is not None and r["split"] == "held"]
    known = [r for r in pages if r["y_known"] is not None and r["split"] == "held"]
    y_dev = np.array([r["y_main"] for r in dev])
    y_held = np.array([r["y_main"] for r in held])
    y_known = np.array([r["y_known"] for r in known])
    cats_held = np.array([r["category"] for r in held])
    print(f"dev {len(dev)} ({y_dev.sum()} AI) | held {len(held)} ({y_held.sum()} AI) | known {len(known)}")
    idx_h, idx_k = boot_idx(y_held), boot_idx(y_known, seed=1)
    probes, scores = {}, {}

    # ---- probes trained on dev labels
    for arm, blocks in ARMS.items():
        Xd = X_of(dev, masked, blocks)
        gs = fit(Xd, y_dev)
        oof = oof_scores(gs, Xd, y_dev)
        res = {
            "C": gs.best_params_["logisticregression__C"],
            "dev_cv_auroc": float(gs.best_score_),
            "dev_oof_auroc": auroc(y_dev, oof),
            "dim": int(Xd.shape[1]),
        }
        if stage == "held":
            sh = gs.decision_function(X_of(held, masked, blocks))
            sk = gs.decision_function(X_of(known, masked, blocks))
            res["held"] = with_ci(y_held, sh, idx_h)
            res["within_industry"] = within_industry(y_held, sh, cats_held)
            res["known"] = with_ci(y_known, sk, idx_k)
            scores[arm] = {"held": sh, "known": sk, "dev_oof": oof}
        probes[arm] = res
        print(arm, round(res["dev_oof_auroc"], 3), res.get("held", {}).get("auroc"))

    # ---- cheap shortcut rule (a generator meta tag; AI-builder pages carry none) and Ep 2's repetition score
    gen_rule = lambda rows: np.array([0.0 if r.get("generator") else 1.0 for r in rows])
    probes["B_generator_tag_rule"] = {"dev_auroc": auroc(y_dev, gen_rule(dev))}
    probes["B_repetition_unsupervised"] = {"dev_auroc": auroc(y_dev, repetition(dev, masked))}
    if stage == "held":
        probes["B_generator_tag_rule"]["held"] = with_ci(y_held, gen_rule(held), idx_h)
        rep_h = repetition(held + known, masked)
        probes["B_repetition_unsupervised"]["held"] = with_ci(y_held, rep_h[: len(held)], idx_h)
        probes["B_repetition_unsupervised"]["known"] = with_ci(y_known, rep_h[len(held) :], idx_k)

    # ---- judges
    judges = load_judges()
    jres = {}
    for name, j in judges.items():
        r = {}
        for prompt in ("plain", "constitution"):
            s = judge_scores(j, dev, True, prompt)
            if np.isfinite(s).all():
                r[f"dev_auroc_{prompt}"] = auroc(y_dev, s)
        if stage == "held" and name in judge_prompt:
            pr = judge_prompt[name]
            r["prompt"] = pr
            for rows_, mask_ in ((held, True), (held, False), (known, True), (dev, True)):
                n_bad = int((~np.isfinite(judge_scores(j, rows_, mask_, pr))).sum())
                assert n_bad == 0, f"{name}: {n_bad} pages without a finite judge score (mask={mask_}); rerun the judge"
            for mask in (True, False):
                s = judge_scores(j, held, mask, pr)
                ok = np.isfinite(s)
                key = "held_masked" if mask else "held_unmasked"
                r[key] = {**with_ci(y_held[ok], s[ok], boot_idx(y_held[ok])), "n": int(ok.sum())}
            s = judge_scores(j, held, True, pr)
            scores[name] = {"held": s, "known": judge_scores(j, known, True, pr), "dev": judge_scores(j, dev, True, pr)}
            r["within_industry"] = within_industry(y_held, s, cats_held)
            r["known"] = with_ci(y_known, scores[name]["known"], idx_k)
            vals, counts = np.unique(np.round(s), return_counts=True)
            r["score_hist"] = {str(int(v)): int(c) for v, c in zip(vals, counts)}
            r["modal_share"] = float(counts.max() / counts.sum())
            r["parse_share_mass_ge_0.9"] = float(np.mean([v["mass"] >= 0.9 for v in j.values()]))
        jres[name] = r
        print("judge", name, {k: v for k, v in r.items() if k.startswith("dev")}, r.get("held_masked", {}).get("auroc"))
    save("probes", probes, stage)
    save("judges", jres, stage)
    if stage != "held":
        return

    # ---- the judge known-answer check: known answers over all splits (templates + generated vs curated)
    known_all = [r for r in pages if r["y_known"] is not None]
    yk_all = np.array([r["y_known"] for r in known_all])
    names = [n for n in judges if n in judge_prompt]
    r3 = {
        n: {
            "known_all_auroc": auroc(yk_all, judge_scores(judges[n], known_all, True, judge_prompt[n])),
            "n": len(known_all),
        }
        for n in names
    }

    # ---- the judge-agreement check: do two judges from different families agree on real held-out pages?
    from scipy.stats import spearmanr

    r5 = {}
    if len(names) == 2:
        a, b = scores[names[0]]["held"], scores[names[1]]["held"]
        rng = np.random.default_rng(2)
        bs = [spearmanr(a[i], b[i]).statistic for i in (rng.integers(0, len(a), len(a)) for _ in range(500))]
        r5 = {
            "pair": names,
            "spearman": float(spearmanr(a, b).statistic),
            "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "half_agreement": float(np.mean((a > np.median(a)) == (b > np.median(b)))),
        }
        for arm in ("A1_siglip", "A3_markup"):
            for n, v in zip(names, (a, b)):
                r5[f"spearman_{arm}_vs_{n}"] = float(spearmanr(scores[arm]["held"], v).statistic)
    save("judge_rulers", {"known_answer": r3, "agreement": r5}, stage)

    # ---- probe vs judge (paired), combinations
    best_probe = max(PROBES, key=lambda a: probes[a]["dev_oof_auroc"])
    best_judge = max(names, key=lambda n: jres[n][f"dev_auroc_{judge_prompt[n]}"]) if names else None
    comp = {"best_probe_by_dev": best_probe, "best_judge_by_dev": best_judge}
    for a in PROBES:
        for n in names:
            comp[f"{a}-{n}"] = paired_diff(y_held, scores[a]["held"], scores[n]["held"], idx_h)
    if names:  # P1 as registered: the best of A1-A4 and the best of J1, J2 on held-out
        bp = max(PROBES, key=lambda a: probes[a]["held"]["auroc"])
        bj = max(names, key=lambda n: jres[n]["held_masked"]["auroc"])
        comp["P1"] = {"probe": bp, "judge": bj, **paired_diff(y_held, scores[bp]["held"], scores[bj]["held"], idx_h)}

    def zcomb(parts, split):
        out = 0
        for name in parts:
            ref = scores[name].get("dev_oof", scores[name].get("dev"))
            out = out + (scores[name][split] - np.nanmean(ref)) / np.nanstd(ref)
        return out / len(parts)

    combos = {}
    if best_judge:
        parts = [best_probe, best_judge]
        combos["C2_probe+judge"] = {
            "parts": parts,
            "held": with_ci(y_held, zcomb(parts, "held"), idx_h),
            "known": with_ci(y_known, zcomb(parts, "known"), idx_k),
            "vs_probe": paired_diff(y_held, zcomb(parts, "held"), scores[best_probe]["held"], idx_h),
        }

    # ---- shuffled-label control (within industry, 10 shuffles)
    rng = np.random.default_rng(3)
    cats_dev = np.array([r["category"] for r in dev])
    shuffle = {}
    for arm in ("A1_siglip", "A3_markup"):
        Xd, Xh = X_of(dev, masked, ARMS[arm]), X_of(held, masked, ARMS[arm])
        vals = []
        for _ in range(10):
            ys = y_dev.copy()
            for c in sorted(set(cats_dev)):  # sorted: set order of strings changes between processes
                ix = np.where(cats_dev == c)[0]
                ys[ix] = rng.permutation(ys[ix])
            vals.append(auroc(y_held, fit(Xd, ys).decision_function(Xh)))
        shuffle[arm] = {"min": float(min(vals)), "max": float(max(vals)), "mean": float(np.mean(vals))}
        print("shuffle", arm, shuffle[arm])

    # ---- leave one builder out (test: every page of the left-out builder vs held human)
    lobo = {}
    allai = [r for r in pages if r["source"] == "ai_host"]
    hum_held = [r for r in held if r["y_main"] == 0]
    for arm in ("A1_siglip", "A3_markup"):
        lobo[arm] = {}
        for b in sorted({r["builder"] for r in allai}):
            tr = [r for r in dev if r["y_main"] == 0 or r["builder"] != b]
            te = [r for r in allai if r["builder"] == b] + hum_held
            yt = np.array([r["y_main"] for r in te])
            gs = fit(X_of(tr, masked, ARMS[arm]), np.array([r["y_main"] for r in tr]))
            s = gs.decision_function(X_of(te, masked, ARMS[arm]))
            lobo[arm][f"builder_{b + 1}"] = {**with_ci(yt, s, boot_idx(yt, 500)), "n_builder": int(yt.sum())}
        vals = [v["auroc"] for v in lobo[arm].values()]
        ns = [v["n_builder"] for v in lobo[arm].values()]
        lobo[arm]["mean"] = float(np.mean(vals))
        lobo[arm]["n_weighted_mean"] = float(np.average(vals, weights=ns))
        print("lobo", arm, lobo[arm]["mean"], lobo[arm]["n_weighted_mean"])

    # ---- constitution probe (A5): trained only on synthetic pages
    cst = {}
    allc = constitution_rows()
    crows = [r for r in allc if r["keep"]]
    if crows:
        cst["render_pass"] = {"kept": len(crows), "total": len(allc), "share": len(crows) / len(allc)}
        tr = [r for r in crows if r["brief"] in (0, 1, 2)]
        te = [r for r in crows if r["brief"] == 3]
        ytr, yte = np.array([r["y"] for r in tr]), np.array([r["y"] for r in te])
        variants = {
            "A5_constitution": ["sig_mask", "geometry"],
            "A5_siglip_only": ["sig_mask"],
            "A5_geometry_only": ["geometry"],
            "A5_markup_only": ["markup"],
        }
        for name, blocks in variants.items():
            gs = fit(cst_X(tr, blocks), ytr)
            sh, sk = gs.decision_function(X_of(held, masked, blocks)), gs.decision_function(X_of(known, masked, blocks))
            sd = gs.decision_function(X_of(dev, masked, blocks))
            scores[name] = {"held": sh, "known": sk, "dev": sd}
            cst[name] = {
                "C": gs.best_params_["logisticregression__C"],
                "synthetic_cv": float(gs.best_score_),
                "brief3_check": auroc(yte, gs.decision_function(cst_X(te, blocks))),
                "dev_real_auroc": auroc(y_dev, sd),
                "held": with_ci(y_held, sh, idx_h),
                "within_industry": within_industry(y_held, sh, cats_held),
                "known": with_ci(y_known, sk, idx_k),
            }
            print(name, round(cst[name]["brief3_check"], 3), round(cst[name]["held"]["auroc"], 3))
        if best_judge:
            parts = ["A5_constitution", best_judge]
            combos["C3_constitution+judge"] = {
                "parts": parts,
                "held": with_ci(y_held, zcomb(parts, "held"), idx_h),
                "known": with_ci(y_known, zcomb(parts, "known"), idx_k),
            }
        cst["repetition"] = {
            "synthetic_slop": rep_stats([r for r in crows if r["y"] == 1], lambda r: r["sig_mask"]),
            "synthetic_good": rep_stats([r for r in crows if r["y"] == 0], lambda r: r["sig_mask"]),
            "real_human_held": rep_stats([r for r in held if r["y_main"] == 0], lambda r: masked[r["pid"]]["sig_mask"]),
            "real_ai_held": rep_stats([r for r in held if r["y_main"] == 1], lambda r: masked[r["pid"]]["sig_mask"]),
        }
        gs = fit(cst_X(tr, variants["A5_constitution"]), ytr)
        s_te = gs.decision_function(cst_X(te, variants["A5_constitution"]))
        cst["per_principle_brief3"] = {}
        for pr in sorted({r["principle"] for r in te if r["y"] == 1}):
            m = np.array([(r["y"] == 0) or (r["principle"] == pr) for r in te])
            cst["per_principle_brief3"][pr] = auroc(yte[m], s_te[m])

    save("comparison", comp, stage)
    save("combos", combos, stage)
    save("shuffle", shuffle, stage)
    save("lobo", lobo, stage)
    save("constitution", cst, stage)
    # per-page scores for figures and the interactive (pid -> scores; no page content); stays in the cache
    per = {
        "held": [
            {
                "pid": r["pid"],
                "y": int(r["y_main"]),
                "category": r["category"],
                "builder": int(r["builder"]),
                **{k: float(v["held"][i]) for k, v in scores.items()},
            }
            for i, r in enumerate(held)
        ],
        "known": [
            {
                "pid": r["pid"],
                "y": int(r["y_known"]),
                "source": r["source"],
                **{k: float(v["known"][i]) for k, v in scores.items()},
            }
            for i, r in enumerate(known)
        ],
    }
    (CACHE / "scores_held.json").write_text(json.dumps(per))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["dev", "held"], required=True)
    ap.add_argument("--qwen72", default="plain", help="judge prompt chosen on dev")
    ap.add_argument("--gemma27", default="plain", help="judge prompt chosen on dev")
    a = ap.parse_args()
    run(a.stage, {"qwen72": a.qwen72, "gemma27": a.gemma27})
