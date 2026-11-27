"""Every number in the note, from work/ (scores, Modal readouts) to results/*.json, as PREREGISTRATION.md defines it.

python experiments/seq2feature-repl/analyze.py rep      # R1-R6: replication table with document-bootstrap CIs
python experiments/seq2feature-repl/analyze.py e1       # E1: frequency baselines, lift, density bins, rare half
python experiments/seq2feature-repl/analyze.py e2       # E2: same text, different context (needs work/e2/)
python experiments/seq2feature-repl/analyze.py ood      # R7: out of distribution (needs work/ood/, scores_ood)
python experiments/seq2feature-repl/analyze.py e3       # E3: safety-labelled concepts (after e2)
"""

import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import DOMS, RESULTS, SAFETY_RE, WORK, auc_vector, doc_bootstrap, doc_resamples, load_bundle, top5_per_span

N_BOOT, N_BOOT_AUC = 2000, 200


def _load():
    B = load_bundle()
    S = np.load(WORK / "scores_test.npz")
    te_idx = S["test_index"]
    assert (te_idx == np.where(B["te"])[0]).all()
    return B, S, te_idx


def _median_auc_boot(args):
    scores, Y, docs, seed = args
    return [
        float(np.nanmedian(auc_vector(scores[rows], Y[rows])))
        for rows in doc_resamples(docs, N_BOOT_AUC // 4, seed=seed)
    ]


def median_auc_with_ci(scores, Y, docs):
    """Per-concept AUCs and a 95% document-bootstrap interval of their median (200 resamples, 4 processes)."""
    point = auc_vector(scores, Y)
    with ProcessPoolExecutor(4) as ex:
        meds = sum(ex.map(_median_auc_boot, [(scores, Y, docs, s) for s in range(4)]), [])
    return point, [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]


def top5_row(scores, Y, docs):
    per, _ = top5_per_span(scores, Y)
    return {"top5": float(per.mean()), "top5_ci": doc_bootstrap(per, docs, N_BOOT)}


def rep():
    B, S, te_idx = _load()
    Yte, docs = B["Y"][te_idx], B["docs"][te_idx]
    freq = B["Y"][~B["te"]].mean(0)
    out = {
        "n_test_spans": int(len(te_idx)),
        "n_test_docs": int(len(np.unique(docs))),
        "sae_concepts_per_span": float(Yte.sum(1).mean()),
    }
    for k in ("fp32", "int8", "int4", "bow", "bge", "bge_z"):
        sc = S[k].astype(np.float32)
        row = top5_row(sc, Yte, docs)
        if k != "bge_z":
            auc, ci = median_auc_with_ci(sc, Yte, docs)
            row.update(
                {
                    "auc_median": float(np.nanmedian(auc)),
                    "auc_median_ci": ci,
                    "auc_mean": float(np.nanmean(auc)),
                    "n_concepts_auc": int(np.isfinite(auc).sum()),
                }
            )
            np.save(WORK / f"auc_{k}.npy", auc)
        out[k] = row
        print(k, row, flush=True)
    out["freq5"] = top5_row(np.tile(freq, (len(te_idx), 1)), Yte, docs)
    out["random5"] = {"top5": float(Yte.mean()), "top5_ci": doc_bootstrap(Yte.mean(1), docs, N_BOOT)}
    base = top5_per_span(S["fp32"].astype(np.float32), Yte)[0]
    for k in ("int8", "int4"):
        d = top5_per_span(S[k].astype(np.float32), Yte)[0] - base
        out[f"{k}_minus_fp32"] = {"mean": float(d.mean()), "ci": doc_bootstrap(d, docs, N_BOOT)}
    print({k: out[k] for k in ("freq5", "random5", "int8_minus_fp32", "int4_minus_fp32")})
    (RESULTS / "replication.json").write_text(json.dumps(out, indent=1))


def e1():
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.linear_model import LogisticRegression

    B, S, te_idx = _load()
    Y, te, dom, docs_all = B["Y"], B["te"], B["dom"], B["docs"]
    Yte, docs, dom_te = Y[te_idx], docs_all[te_idx], dom[te_idx]
    dens = Y[~te].mean(0)
    dens_dom = np.stack([Y[~te & (dom == d)].mean(0) for d in range(len(DOMS))])
    probe = S["int8"].astype(np.float32)
    out = {
        "density_median": float(np.median(dens)),
        "density_min": float(dens.min()),
        "density_quantiles_10_25_50_75_90": np.percentile(dens, [10, 25, 50, 75, 90]).tolist(),
    }

    hv = HashingVectorizer(n_features=4096, alternate_sign=False, ngram_range=(1, 2))
    X = hv.transform([s["text"] for s in B["spans"]])
    dom_pred = LogisticRegression(max_iter=1000).fit(X[~te], dom[~te]).predict(X[te_idx])
    out["domain_classifier_accuracy"] = float((dom_pred == dom_te).mean())

    def picks_score(picks):
        per = np.take_along_axis(Yte, picks, 1).mean(1)
        lift = per - dens[picks].mean(1)
        lift_dom = per - np.take_along_axis(dens_dom[dom_te], picks, 1).mean(1)  # exploratory: beyond genre
        return {
            "top5": float(per.mean()),
            "top5_ci": doc_bootstrap(per, docs, N_BOOT),
            "lift": float(lift.mean()),
            "lift_ci": doc_bootstrap(lift, docs, N_BOOT),
            "lift_beyond_domain": float(lift_dom.mean()),
            "lift_beyond_domain_ci": doc_bootstrap(lift_dom, docs, N_BOOT),
            "median_density_of_picks": float(np.median(dens[picks])),
        }, per

    top_dom = np.argsort(-dens_dom, 1)[:, :5]
    rankers = {
        "probe_int8": top5_per_span(probe, Yte)[1],
        "bow": top5_per_span(S["bow"].astype(np.float32), Yte)[1],
        "freq5": np.tile(np.argsort(-dens)[:5], (len(te_idx), 1)),
        "dom_oracle": top_dom[dom_te],
        "dom_text": top_dom[dom_pred],
    }
    per_span = {}
    for name, picks in rankers.items():
        out[name], per_span[name] = picks_score(picks)
        print(name, out[name], flush=True)
    gap = per_span["probe_int8"] - per_span["dom_text"]
    out["probe_minus_dom_text"] = {"mean": float(gap.mean()), "ci": doc_bootstrap(gap, docs, N_BOOT)}
    out["share_of_floor_to_probe_gap_closed_by_dom_text"] = float(
        (out["dom_text"]["top5"] - out["freq5"]["top5"]) / (out["probe_int8"]["top5"] - out["freq5"]["top5"])
    )
    out["fired_concepts_median_density"] = float(np.median(dens[np.where(Yte)[1]]))

    auc_p, auc_b = np.load(WORK / "auc_int8.npy"), np.load(WORK / "auc_bow.npy")
    edges = [0, 0.01, 0.02, 0.05, 0.10, 1.01]
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dens >= lo) & (dens < hi) & np.isfinite(auc_p)
        if not m.any():  # the kept dictionary has no concept this rare (min density 1.76%)
            bins.append({"lo": lo, "hi": min(hi, 1.0), "n": 0})
            continue
        bins.append(
            {
                "lo": lo,
                "hi": min(hi, 1.0),
                "n": int(m.sum()),
                "probe_auc_median": float(np.median(auc_p[m])),
                "bow_auc_median": float(np.nanmedian(auc_b[m])),
                "probe_auc_q25_q75": np.percentile(auc_p[m], [25, 75]).tolist(),
            }
        )
    out["density_bins"] = bins
    print("bins", bins)

    rare = np.where(dens < np.median(dens))[0]
    Yr = Yte[:, rare]
    per_r = top5_per_span(probe[:, rare], Yr)[0]
    per_rb = top5_per_span(S["bow"].astype(np.float32)[:, rare], Yr)[0]
    out["rare_half"] = {
        "n_concepts": int(len(rare)),
        "density_max": float(dens[rare].max()),
        "probe_top5": float(per_r.mean()),
        "probe_top5_ci": doc_bootstrap(per_r, docs, N_BOOT),
        "bow_top5": float(per_rb.mean()),
        "bow_top5_ci": doc_bootstrap(per_rb, docs, N_BOOT),
        "freq5": float(Yr[:, np.argsort(-dens[rare])[:5]].mean()),
        "random5": float(Yr.mean()),
        "spans_with_no_rare_firing": float((Yr.sum(1) == 0).mean()),
    }
    print("rare", out["rare_half"])
    (RESULTS / "e1.json").write_text(json.dumps(out, indent=1))


def _jacc(a, b):
    inter, union = (a & b).sum(1), (a | b).sum(1)
    return np.where(union > 0, inter / np.maximum(union, 1), 1.0)


def e2():
    from scipy.stats import binomtest, rankdata, spearmanr
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    B, S, te_idx = _load()
    R = np.load(WORK / "e2" / "readouts.npz")
    V = B["V"]
    idx = R["span_index"]
    pos = np.searchsorted(te_idx, idx)
    assert (te_idx[pos] == idx).all()
    probe = S["int8"].astype(np.float32)[pos]
    docs = B["docs"][idx]
    Yship = B["Y"][idx]
    conds = ("K0", "K1", "CTX", "F-honest", "F-sandbag", "F-lie")
    lab = {c: R[c][:, V].astype(np.float32) > 1.0 for c in conds}
    out = {"n_spans": int(len(idx)), "n_docs": int(len(np.unique(docs)))}
    j0 = _jacc(lab["K0"], Yship)
    out["gate_K0_vs_shipped_jaccard"] = {
        "mean": float(j0.mean()),
        "ci": doc_bootstrap(j0, docs, N_BOOT),
        "pass_bar": 0.90,
        "passed": bool(j0.mean() >= 0.90),
    }
    jr = _jacc(R["K1_rerun"][:, V].astype(np.float32) > 1.0, lab["K1"][:200])
    out["K1_rerun_one_minus_jaccard"] = float(1 - jr.mean())
    print("gate", out["gate_K0_vs_shipped_jaccard"], "rerun", out["K1_rerun_one_minus_jaccard"], flush=True)
    out["concepts_per_span"] = {c: float(v.sum(1).mean()) for c, v in lab.items()}
    out["context_share_vs_K1"] = {}
    for c in ("K0", "CTX", "F-honest", "F-sandbag", "F-lie"):
        j = _jacc(lab[c], lab["K1"])
        lo, hi = doc_bootstrap(j, docs, N_BOOT)
        out["context_share_vs_K1"][c] = {"one_minus_jaccard": float(1 - j.mean()), "ci": [1 - hi, 1 - lo]}
    j = _jacc(lab["F-sandbag"], lab["F-honest"])
    lo, hi = doc_bootstrap(j, docs, N_BOOT)
    out["context_share_sandbag_vs_honest"] = {"one_minus_jaccard": float(1 - j.mean()), "ci": [1 - hi, 1 - lo]}
    out["probe_vs_condition"] = {}
    bow = S["bow"].astype(np.float32)[pos]
    f5 = np.tile(B["Y"][~B["te"]].mean(0), (len(idx), 1))
    for c in ("shipped",) + conds:
        Yc = Yship if c == "shipped" else lab[c]
        per = top5_per_span(probe, Yc)[0]
        auc = auc_vector(probe, Yc)
        out["probe_vs_condition"][c] = {
            "top5": float(per.mean()),
            "top5_ci": doc_bootstrap(per, docs, N_BOOT),
            "auc_median": float(np.nanmedian(auc)),
            "n_concepts": int(np.isfinite(auc).sum()),
            # exploratory: the same baselines against this condition's labels
            "bow_top5": float(top5_per_span(bow, Yc)[0].mean()),
            "freq5_top5": float(top5_per_span(f5, Yc)[0].mean()),
            "random5": float(Yc.mean()),
        }
    print("probe vs cond", json.dumps(out["probe_vs_condition"]), flush=True)

    gkf = GroupKFold(n_splits=5)
    out["framing_readout"] = {}
    for other in ("F-sandbag", "F-lie"):
        X = np.vstack(
            [
                np.log1p(np.maximum(R[other].astype(np.float32), 0)),
                np.log1p(np.maximum(R["F-honest"].astype(np.float32), 0)),
            ]
        )
        y = np.r_[np.ones(len(idx)), np.zeros(len(idx))]
        g = np.r_[docs, docs]
        res = {}
        for tag, yy in (("real", y), ("shuffled", np.random.default_rng(0).permutation(y))):
            pred = np.zeros(len(y))
            for tr, tst in gkf.split(X, yy, g):
                pred[tst] = LogisticRegression(C=1.0, max_iter=2000).fit(X[tr], yy[tr]).decision_function(X[tst])
            res[tag] = float(roc_auc_score(yy, pred))
        res["probe_identical_inputs_auroc"] = float(roc_auc_score(y, np.r_[probe[:, 0], probe[:, 0]]))
        out["framing_readout"][other + "_vs_F-honest"] = res
        print("framing", other, res, flush=True)

    flip = (lab["K1"] != lab["CTX"]).mean(0)
    either = (lab["K1"] | lab["CTX"]).sum(0)
    flip_rel = (lab["K1"] != lab["CTX"]).sum(0) / np.maximum(either, 1)  # flips per span where it fired in either
    auc_id = np.load(WORK / "auc_int8.npy")
    dens = B["Y"][~B["te"]].mean(0)
    ok = np.isfinite(auc_id) & (either > 0)
    okw = np.where(ok)[0]
    rng = np.random.default_rng(0)
    boots = [spearmanr(flip[s], auc_id[s]).statistic for s in (rng.choice(okw, len(okw)) for _ in range(1000))]

    def resid(a, b):
        ra, rb = rankdata(a), rankdata(b)
        return ra - np.polyval(np.polyfit(rb, ra, 1), rb)

    out["state_sensitivity"] = {
        "n_concepts": int(ok.sum()),
        "spearman_flip_vs_auc": float(spearmanr(flip[ok], auc_id[ok]).statistic),
        "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "partial_spearman_given_density": float(
            np.corrcoef(resid(flip[ok], dens[ok]), resid(auc_id[ok], dens[ok]))[0, 1]
        ),
        "spearman_flips_per_firing_vs_auc": float(spearmanr(flip_rel[ok], auc_id[ok]).statistic),
        "median_flip_rate": float(np.median(flip[ok])),
        "median_flips_per_firing": float(np.median(flip_rel[ok])),
    }
    np.save(WORK / "flip_rel.npy", flip_rel)
    print("state", out["state_sensitivity"], flush=True)

    a, b = lab["F-sandbag"], lab["F-honest"]
    n10, n01 = (a & ~b).sum(0), (~a & b).sum(0)
    p = np.array([binomtest(int(x), int(x + y), 0.5).pvalue if x + y > 0 else 1.0 for x, y in zip(n10, n01)])
    order = np.argsort(p)
    passed = p[order] <= 0.05 * np.arange(1, len(p) + 1) / len(p)
    k = int(np.max(np.where(passed)[0]) + 1) if passed.any() else 0
    sig = order[:k]
    out["framing_sensitive_concepts"] = {
        "n_significant": k,
        "top": [
            {
                "concept": int(c),
                "latent": int(V[c]),
                "caption": B["caps"][c][:100],
                "on_sandbag_only": int(n10[c]),
                "on_honest_only": int(n01[c]),
            }
            for c in sig[np.argsort(-np.abs(n10 - n01)[sig])][:15]
        ],
    }
    print("mcnemar", k, flush=True)
    (RESULTS / "e2.json").write_text(json.dumps(out, indent=1))


def ood():
    Bk = load_bundle()
    V = Bk["V"]
    spans = [json.loads(line) for line in open(WORK / "ood" / "ood_spans.jsonl")]
    Y = np.load(WORK / "ood" / "ood_readouts.npz")["maxact"][:, V].astype(np.float32) > 1.0
    S = np.load(WORK / "scores_ood.npz")
    docs = np.array([s["doc"] for s in spans])
    domo = np.array([s["dom"] for s in spans])
    dens = Bk["Y"][~Bk["te"]].mean(0)
    out = {
        "n_spans": {d: int((domo == d).sum()) for d in ("medical", "legal")},
        "published_n_spans": {"medical": 1997, "legal": 505},
        "sae_concepts_per_span": float(Y.sum(1).mean()),
    }
    for k in ("int8", "fp32", "bow", "bge"):
        sc = S[k].astype(np.float32)
        row = top5_row(sc, Y, docs)
        auc, ci = median_auc_with_ci(sc, Y, docs)
        row.update(
            {"auc_median": float(np.nanmedian(auc)), "auc_median_ci": ci, "n_concepts": int(np.isfinite(auc).sum())}
        )
        for d in ("medical", "legal"):
            m = domo == d
            row[f"auc_median_{d}"] = float(np.nanmedian(auc_vector(sc[m], Y[m])))
            row[f"top5_{d}"] = float(top5_per_span(sc[m], Y[m])[0].mean())
        out[k] = row
        print(k, row, flush=True)
    out["freq5_train_density"] = top5_row(np.tile(dens, (len(Y), 1)), Y, docs)
    out["random5"] = {"top5": float(Y.mean())}
    print("ood floors", out["freq5_train_density"], out["random5"])
    # exploratory (amendment 1): the same OOD spans read with a BOS token
    Yb = np.load(WORK / "ood" / "ood_readouts.npz")["maxact_bos"][:, V].astype(np.float32) > 1.0
    sc = S["int8"].astype(np.float32)
    jb = _jacc(Y, Yb)
    out["exploratory_with_bos"] = {
        "jaccard_no_bos_vs_bos": float(jb.mean()),
        "sae_concepts_per_span": float(Yb.sum(1).mean()),
        "probe_int8": {**top5_row(sc, Yb, docs), "auc_median": float(np.nanmedian(auc_vector(sc, Yb)))},
        "freq5_train_density": top5_row(np.tile(dens, (len(Yb), 1)), Yb, docs),
        "random5": float(Yb.mean()),
    }
    print("ood with bos", out["exploratory_with_bos"])
    (RESULTS / "ood.json").write_text(json.dumps(out, indent=1))


def e3():
    B = load_bundle()
    caps, te, Y = B["caps"], B["te"], B["Y"]
    rx = re.compile(SAFETY_RE, re.I)
    safe = np.array([bool(rx.search(c)) for c in caps])
    dens = Y[~te].mean(0)
    auc_p, auc_b = np.load(WORK / "auc_int8.npy"), np.load(WORK / "auc_bow.npy")
    out = {
        "n_safety": int(safe.sum()),
        "probe_auc_median_safety": float(np.nanmedian(auc_p[safe])),
        "probe_auc_median_other": float(np.nanmedian(auc_p[~safe])),
        "bow_auc_median_safety": float(np.nanmedian(auc_b[safe])),
        "density_median_safety": float(np.median(dens[safe])),
        "density_median_other": float(np.median(dens[~safe])),
        "per_concept": [
            {
                "caption": caps[c][:90],
                "probe_auc": float(auc_p[c]),
                "bow_auc": float(auc_b[c]),
                "density": float(dens[c]),
            }
            for c in np.where(safe)[0]
        ],
    }
    fr = WORK / "flip_rel.npy"
    if fr.exists():
        flip_rel = np.load(fr)
        out["flips_per_firing_median_safety"] = float(np.median(flip_rel[safe]))
        out["flips_per_firing_median_other"] = float(np.median(flip_rel[~safe]))
        for row, c in zip(out["per_concept"], np.where(safe)[0]):
            row["flips_per_firing"] = float(flip_rel[c])
    print({k: v for k, v in out.items() if k != "per_concept"})
    (RESULTS / "e3.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"rep": rep, "e1": e1, "e2": e2, "ood": ood, "e3": e3}[sys.argv[1]]()
