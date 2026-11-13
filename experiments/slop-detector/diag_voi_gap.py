# ruff: noqa: E501, B905
"""Free diagnostics for the gap to voi-slop-detect-4B. EditLens VALIDATION rows and saved scores only. No GPU.

    python diag_voi_gap.py            # prints every table; writes results/diag_voi_gap.json

Never loads a test score or a test label. The test CSVs are opened only inside clean_val(), the existing rule that
drops the 4 validation rows touching a test split. Every "gain" below is measured on validation; anything fitted
(weights, cut-offs, stackers) is fitted with 5-fold cross-fitting grouped by source document, so the number is
out-of-fold. Written 2026-11-12 for docs/plans/2026-11-12-beat-voi-levers.md.
"""

from __future__ import annotations

import json
from importlib import import_module

import numpy as np
from common import HERE, RESULTS, words

HS = np.round(np.arange(0.05, 0.40001, 0.005), 3)
GS = np.round(np.arange(0.50, 0.95001, 0.005), 3)
NAMES = ("human", "edited", "generated")


def bands(s, h, g):
    return np.where(s < h, 0, np.where(s >= g, 2, 1))


def conf(y, p):
    return np.bincount(y * 3 + p, minlength=9).reshape(3, 3)


def mf1(y, p):
    c = conf(y, p).astype(float)
    tp = np.diag(c)
    den = c.sum(0) + c.sum(1)
    return float(np.mean(np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)))


def per_f1(y, p):
    c = conf(y, p).astype(float)
    return (2 * np.diag(c) / np.maximum(c.sum(0) + c.sum(1), 1)).round(4).tolist()


def fit_cuts(y, s):
    """Best (h, g) by macro-F1 on the grid used by the pre-registered arms. Fast: cumulative counts per cut."""
    best = (-1.0, 0.15, 0.75)
    # counts of each class below each cut
    srt = {c: np.sort(s[y == c]) for c in (0, 1, 2)}
    n = {c: len(srt[c]) for c in (0, 1, 2)}
    below_h = {c: np.searchsorted(srt[c], HS, side="left") for c in (0, 1, 2)}
    below_g = {c: np.searchsorted(srt[c], GS, side="left") for c in (0, 1, 2)}
    for i, h in enumerate(HS):
        for j, g in enumerate(GS):
            if h >= g:
                continue
            f = 0.0
            # pred 0: s<h ; pred 1: h<=s<g ; pred 2: s>=g
            tp0 = below_h[0][i]
            p0 = sum(below_h[c][i] for c in (0, 1, 2))
            tp1 = below_g[1][j] - below_h[1][i]
            p1 = sum(below_g[c][j] - below_h[c][i] for c in (0, 1, 2))
            tp2 = n[2] - below_g[2][j]
            p2 = sum(n[c] - below_g[c][j] for c in (0, 1, 2))
            for tp, p, c in ((tp0, p0, 0), (tp1, p1, 1), (tp2, p2, 2)):
                f += 2 * tp / max(p + n[c], 1)
            f /= 3
            if f > best[0] + 1e-12:
                best = (f, float(h), float(g))
    return best


def folds(groups, k=5, seed=0):
    rng = np.random.default_rng(seed)
    u = np.unique(groups)
    rng.shuffle(u)
    m = {g: i % k for i, g in enumerate(u)}
    return np.array([m[g] for g in groups])


def boot(y, pa, pb, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    d = []
    for _ in range(reps):
        i = rng.integers(0, n, n)
        d.append(mf1(y[i], pa[i]) - mf1(y[i], pb[i]))
    return [round(float(np.quantile(d, q)), 4) for q in (0.025, 0.975)]


def main() -> None:
    tev = import_module("tune_editlens_val")
    val, dropped = tev.clean_val()
    y = val["y"].to_numpy()
    s4 = np.array(json.loads((HERE / "runs/v2-4b/scores/editlens_val.json").read_text()))
    s17 = np.array(json.loads((HERE / "runs/v2-1.7b/scores/editlens_val.json").read_text()))
    sv = np.array(json.loads((RESULTS / "voi/editlens_val.json").read_text())["scores"])
    assert len(y) == len(s4) == len(s17) == len(sv), (len(y), len(s4), len(sv))
    ens = 0.8 * s4 + 0.2 * s17
    out: dict = {"rows": len(y), "dropped": dropped, "class_counts": np.bincount(y).tolist()}
    nw = np.array([len(words(str(t))) for t in val["text"]])
    grp = val["source_id"].fillna(val["text_id"]).to_numpy()
    fold = folds(grp)

    print(f"EditLens val rows {len(y)} (dropped {dropped}); classes {dict(zip(NAMES, np.bincount(y).tolist()))}")

    # ---- 1. systems and confusion matrices -------------------------------------------------------------
    _, vh, vg = fit_cuts(y, sv)
    _, h4, g4 = fit_cuts(y, s4)
    systems = {
        "ours 4B, fixed 0.15/0.75": bands(s4, 0.15, 0.75),
        "voi, fixed 0.15/0.75": bands(sv, 0.15, 0.75),
        "ours ensemble, 0.055/0.86 (look #4 setting)": bands(ens, 0.055, 0.86),
        f"voi, val-fitted {vh}/{vg}": bands(sv, vh, vg),
        f"ours 4B alone, val-fitted {h4}/{g4}": bands(s4, h4, g4),
    }
    out["systems"] = {}
    print("\n== 1. Macro-F1 and confusion matrices (rows = truth, cols = predicted: human, edited, generated)")
    for name, p in systems.items():
        c = conf(y, p)
        out["systems"][name] = {"macro_f1": round(mf1(y, p), 4), "f1": per_f1(y, p), "confusion": c.tolist()}
        print(f"\n{name}: macro-F1 {mf1(y, p):.4f}  F1 h/e/g {per_f1(y, p)}")
        for i in range(3):
            print(f"  {NAMES[i]:9} {c[i].tolist()}  recall {c[i, i] / c[i].sum():.3f}")
    po, pv = systems["ours ensemble, 0.055/0.86 (look #4 setting)"], systems[f"voi, val-fitted {vh}/{vg}"]
    pof, pvf = systems["ours 4B, fixed 0.15/0.75"], systems["voi, fixed 0.15/0.75"]
    print("\nfixed cuts, ours 4B - voi:", round(mf1(y, pof) - mf1(y, pvf), 4), boot(y, pof, pvf))
    print("calibrated (in-sample cuts), ours ens - voi:", round(mf1(y, po) - mf1(y, pv), 4), boot(y, po, pv))

    # ---- 2. error pairs --------------------------------------------------------------------------------
    print("\n== 2. Which class pairs drive the errors (share of all errors)")
    out["error_pairs"] = {}
    for name, p in (("ours calibrated", po), ("voi calibrated", pv), ("ours fixed", pof), ("voi fixed", pvf)):
        c = conf(y, p)
        err = c.sum() - np.trace(c)
        pairs = {
            "edited->human": c[1, 0],
            "human->edited": c[0, 1],
            "edited->generated": c[1, 2],
            "generated->edited": c[2, 1],
            "human->generated": c[0, 2],
            "generated->human": c[2, 0],
        }
        out["error_pairs"][name] = {"errors": int(err), **{k: int(v) for k, v in pairs.items()}}
        print(
            f"{name:16} errors {err:4d} ({err / len(y):.1%}): "
            + ", ".join(f"{k} {v} ({v / err:.0%})" for k, v in pairs.items())
        )

    # ---- 3. slices -------------------------------------------------------------------------------------
    def slice_table(title, key, order=None):
        print(f"\n== 3. {title}: error rate ours / voi (calibrated), n; then fixed cuts")
        res = {}
        for k in order or sorted(set(key)):
            m = key == k
            if m.sum() < 20:
                continue
            r = {
                "n": int(m.sum()),
                "err_ours_cal": round(float((po[m] != y[m]).mean()), 4),
                "err_voi_cal": round(float((pv[m] != y[m]).mean()), 4),
                "err_ours_fixed": round(float((pof[m] != y[m]).mean()), 4),
                "err_voi_fixed": round(float((pvf[m] != y[m]).mean()), 4),
            }
            res[str(k)] = r
            print(
                f"  {k!s:28} n {r['n']:5d}  cal {r['err_ours_cal']:.3f} / {r['err_voi_cal']:.3f}   fixed {r['err_ours_fixed']:.3f} / {r['err_voi_fixed']:.3f}"
            )
        out.setdefault("slices", {})[title] = res

    slice_table("by source (domain)", val["source"].to_numpy())
    slice_table("by generator", val["model"].to_numpy())
    edges = [0, 50, 100, 200, 300, 400, 10**6]
    lb = np.array(
        [f"{edges[i]}-{edges[i + 1] - 1 if edges[i + 1] < 10**6 else '+'} words" for i in np.digitize(nw, edges) - 1]
    )
    slice_table(
        "by length",
        lb,
        [f"{edges[i]}-{edges[i + 1] - 1 if edges[i + 1] < 10**6 else '+'} words" for i in range(len(edges) - 1)],
    )
    slice_table("by class x source", np.array([f"{NAMES[a]} | {b}" for a, b in zip(y, val["source"], strict=True)]))
    # edited rows by EditLens's own edit size (cosine_score): are light edits the problem?
    ed = y == 1
    cs = val["cosine_score"].to_numpy()
    q = np.quantile(cs[ed], [0, 0.25, 0.5, 0.75, 1.0])
    print(
        "\n== 3b. AI-edited rows by EditLens cosine_score quartile (its own measure of edit size): predicted band shares"
    )
    out["edited_by_cosine"] = []
    for i in range(4):
        m = ed & (cs >= q[i]) & (cs <= q[i + 1] if i == 3 else cs < q[i + 1])
        row = {"cos_lo": round(float(q[i]), 3), "cos_hi": round(float(q[i + 1]), 3), "n": int(m.sum())}
        for nm, p, s in (("ours", po, ens), ("voi", pv, sv)):
            row[nm] = np.round(np.bincount(p[m], minlength=3) / m.sum(), 3).tolist()
            row[nm + "_median_score"] = round(float(np.median(s[m])), 3)
        out["edited_by_cosine"].append(row)
        print(
            f"  cos {row['cos_lo']:.3f}-{row['cos_hi']:.3f} n {row['n']}: ours h/e/g {row['ours']} (median score {row['ours_median_score']}); voi {row['voi']} (median {row['voi_median_score']})"
        )
    from scipy.stats import spearmanr

    out["spearman_with_cosine_on_edited"] = {
        k: round(float(spearmanr(s[ed], cs[ed]).statistic), 3)
        for k, s in (("ours_4b", s4), ("ours_ens", ens), ("voi", sv))
    }
    print("  Spearman(score, cosine_score) on edited rows:", out["spearman_with_cosine_on_edited"])
    for nm, s in (("ours ens", ens), ("voi", sv)):
        print(
            f"  {nm}: score quantiles 5/25/50/75/95 edited {np.quantile(s[ed], [0.05, 0.25, 0.5, 0.75, 0.95]).round(3).tolist()}"
            f" | generated {np.quantile(s[y == 2], [0.05, 0.25, 0.5, 0.75, 0.95]).round(3).tolist()}"
            f" | human {np.quantile(s[y == 0], [0.05, 0.25, 0.5, 0.75, 0.95]).round(3).tolist()}"
        )

    # ---- 4. voi + ours ensemble, cross-fitted ----------------------------------------------------------
    print("\n== 4. Ensembles, 5-fold cross-fitted by source document (weights and cut-offs fitted out-of-fold)")

    def crossfit(cands):
        """cands: list of score arrays; picks the best candidate + cuts on 4 folds, predicts the 5th."""
        pred = np.zeros(len(y), dtype=int)
        picks = []
        for k in range(5):
            tr, te = fold != k, fold == k
            best = max(((fit_cuts(y[tr], s[tr]), i) for i, s in enumerate(cands)), key=lambda t: t[0][0])
            (_, h, g), i = best
            pred[te] = bands(cands[i][te], h, g)
            picks.append((i, h, g))
        return pred, picks

    ws = np.round(np.arange(0, 1.0001, 0.1), 1)
    arms = {
        "voi alone": [sv],
        "ours 4B alone": [s4],
        "ours 4B + 1.7B": [w * s4 + (1 - w) * s17 for w in ws],
        "ours ens + voi": [w * ens + (1 - w) * sv for w in ws],
        "4B + voi": [w * s4 + (1 - w) * sv for w in ws],
    }
    cf = {}
    out["crossfit"] = {}
    for name, cands in arms.items():
        p, picks = crossfit(cands)
        cf[name] = p
        out["crossfit"][name] = {
            "macro_f1": round(mf1(y, p), 4),
            "f1": per_f1(y, p),
            "picks": [[float(ws[i]) if len(cands) > 1 else 1.0, h, g] for i, h, g in picks],
        }
        print(
            f"  {name:18} macro-F1 {mf1(y, p):.4f}  F1 h/e/g {per_f1(y, p)}  picks(w,h,g) {out['crossfit'][name]['picks']}"
        )
    for a, b in (
        ("ours 4B + 1.7B", "voi alone"),
        ("ours ens + voi", "ours 4B + 1.7B"),
        ("ours ens + voi", "voi alone"),
    ):
        d = round(mf1(y, cf[a]) - mf1(y, cf[b]), 4)
        ci = boot(y, cf[a], cf[b])
        out["crossfit"][f"{a} - {b}"] = {"diff": d, "ci": ci}
        print(f"  {a} - {b}: {d:+.4f} {ci}")
    # in-sample weight curve for ours-ens + voi
    curve = {float(w): round(fit_cuts(y, w * ens + (1 - w) * sv)[0], 4) for w in ws}
    out["insample_weight_curve_ours_vs_voi"] = curve
    print("  in-sample best macro-F1 by weight on ours (0 = voi only):", curve)

    # ---- 5. oracle and agreement -----------------------------------------------------------------------
    print("\n== 5. Oracle choice between ours and voi (calibrated), and how the errors overlap")
    oc, vc = po == y, pv == y
    oracle = np.where(oc, po, pv)
    out["oracle"] = {
        "both_right": int((oc & vc).sum()),
        "only_ours_right": int((oc & ~vc).sum()),
        "only_voi_right": int((~oc & vc).sum()),
        "both_wrong": int((~oc & ~vc).sum()),
        "oracle_macro_f1": round(mf1(y, oracle), 4),
        "ours": round(mf1(y, po), 4),
        "voi": round(mf1(y, pv), 4),
        "pred_agreement": round(float((po == pv).mean()), 4),
        "score_spearman": round(float(spearmanr(ens, sv).statistic), 4),
    }
    print(" ", out["oracle"])
    bw = ~oc & ~vc
    c = conf(y[bw], po[bw])
    print(
        "  both wrong, by truth -> our prediction:",
        {f"{NAMES[i]}->{NAMES[j]}": int(c[i, j]) for i in range(3) for j in range(3) if c[i, j]},
    )
    out["oracle"]["both_wrong_pairs"] = {
        f"{NAMES[i]}->{NAMES[j]}": int(c[i, j]) for i in range(3) for j in range(3) if c[i, j]
    }
    for nm, m in (("only voi right", ~oc & vc), ("only ours right", oc & ~vc)):
        cc = np.bincount(y[m], minlength=3)
        print(f"  {nm}: by true class {dict(zip(NAMES, cc.tolist()))}")
        out["oracle"][nm.replace(" ", "_") + "_by_class"] = cc.tolist()

    # ---- 6. cheap lever probes on validation -----------------------------------------------------------
    print("\n== 6. Lever probes, 5-fold cross-fitted (validation only)")
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def logit(s):
        s = np.clip(s, 1e-6, 1 - 1e-6)
        return np.log(s / (1 - s))

    def stack(X, C=1.0, cw=None):
        pred = np.zeros(len(y), dtype=int)
        for k in range(5):
            tr, te = fold != k, fold == k
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, class_weight=cw))
            m.fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        return pred

    ext = {
        c: val[c].to_numpy(dtype=float)
        for c in (
            "binoculars_score",
            "fastdetectgpt_score",
            "editlens_roberta-large_score",
            "editlens_Llama-3.2-3B_score",
            "pangram_v3.2_score",
        )
    }
    for k, v in ext.items():
        ext[k] = np.where(np.isnan(v), np.nanmedian(v), v)
    L4, L17, LV, LEN = logit(s4), logit(s17), logit(sv), np.log(nw + 1)
    probes = {
        "3-way LR on [4B, 1.7B] (no cut-offs)": np.c_[L4, L17, L4**2, L17**2],
        "... + log length": np.c_[L4, L17, L4**2, L17**2, LEN, LEN * L4],
        "... + binoculars + fastdetectgpt (token-level features)": np.c_[
            L4, L17, L4**2, L17**2, LEN, ext["binoculars_score"], ext["fastdetectgpt_score"]
        ],
        "... + voi": np.c_[L4, L17, L4**2, L17**2, LEN, LV, LV**2],
        "... + voi + binoculars + fastdetectgpt": np.c_[
            L4, L17, L4**2, L17**2, LEN, LV, LV**2, ext["binoculars_score"], ext["fastdetectgpt_score"]
        ],
        "... + EditLens RoBERTa-large (open weights, trained on EditLens train)": np.c_[
            L4, L17, L4**2, L17**2, LEN, logit(ext["editlens_roberta-large_score"])
        ],
        "... + EditLens Llama-3.2-3B (trained on EditLens train)": np.c_[
            L4, L17, L4**2, L17**2, LEN, logit(ext["editlens_Llama-3.2-3B_score"])
        ],
        "voi only, 3-way LR": np.c_[LV, LV**2],
    }
    base = cf["ours 4B + 1.7B"]
    out["probes"] = {}
    for name, X in probes.items():
        p = stack(X)
        d = round(mf1(y, p) - mf1(y, base), 4)
        out["probes"][name] = {
            "macro_f1": round(mf1(y, p), 4),
            "f1": per_f1(y, p),
            "vs_crossfit_ours": d,
            "ci": boot(y, p, base),
        }
        print(f"  {name:72} {mf1(y, p):.4f} {per_f1(y, p)}  vs cross-fitted ours {d:+.4f} {out['probes'][name]['ci']}")
    # reference detectors alone, cut-offs cross-fitted
    print("  reference detectors alone (cut-offs cross-fitted; score rescaled to 0..1 by rank where needed):")
    out["references"] = {}
    for k, v in ext.items():
        r = v if (v.min() >= 0 and v.max() <= 1) else None
        from scipy.stats import rankdata

        rr = rankdata(-v if k == "binoculars_score" else v) / len(v)
        p, _ = crossfit([r if r is not None and k != "binoculars_score" else rr])
        out["references"][k] = round(mf1(y, p), 4)
        print(f"    {k:32} {mf1(y, p):.4f} {per_f1(y, p)}")

    # per-domain and per-length cut-offs
    def grouped_cuts(key, s):
        pred = np.zeros(len(y), dtype=int)
        for k in range(5):
            tr, te = fold != k, fold == k
            _, h0, g0 = fit_cuts(y[tr], s[tr])
            for v in np.unique(key):
                m = key == v
                if (tr & m).sum() >= 150 and len(np.unique(y[tr & m])) == 3:
                    _, h, g = fit_cuts(y[tr & m], s[tr & m])
                else:
                    h, g = h0, g0
                pred[te & m] = bands(s[te & m], h, g)
        return pred

    lenb = np.digitize(nw, [100, 250])
    for name, key in (
        ("cut-offs per source", val["source"].to_numpy()),
        ("cut-offs per length bucket (<100, 100-249, 250+ words)", lenb),
    ):
        p = grouped_cuts(key, ens)
        d = round(mf1(y, p) - mf1(y, base), 4)
        out["probes"][name] = {"macro_f1": round(mf1(y, p), 4), "vs_crossfit_ours": d, "ci": boot(y, p, base)}
        print(f"  {name:72} {mf1(y, p):.4f}  vs cross-fitted ours {d:+.4f} {out['probes'][name]['ci']}")

    # ---- 7. label mismatch in our own training data ----------------------------------------------------
    print(
        "\n== 7. Label mismatch in our own data (train split metadata only): band from words-changed vs how it was made"
    )
    kinds: dict = {}
    with (HERE / "data" / "train.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            kinds.setdefault(r["kind"], [0, 0, 0])[r["band"]] += 1
    prov = {"human": 0, "light": 1, "partial": 1, "rewrite": 1, "roundtrip": 1, "mirror": 2, "imitate": 2}
    tot = mism = 0
    out["train_label_mismatch"] = {}
    for k, c in sorted(kinds.items()):
        n = sum(c)
        bad = n - c[prov[k]]
        tot += n
        mism += bad
        out["train_label_mismatch"][k] = {
            "n": n,
            "bands": c,
            "provenance_class": NAMES[prov[k]],
            "mismatch": round(bad / n, 4),
        }
        print(f"  {k:10} n {n:6d}  bands h/e/g {c}  made as {NAMES[prov[k]]:9}  in another band: {bad / n:.1%}")
    out["train_label_mismatch"]["all"] = round(mism / tot, 4)
    print(f"  all rows: {mism} of {tot} ({mism / tot:.1%}) sit in a band that differs from how the text was made")

    (RESULTS / "diag_voi_gap.json").write_text(json.dumps(out, indent=1) + "\n")
    print("\nwrote results/diag_voi_gap.json")


if __name__ == "__main__":
    main()
