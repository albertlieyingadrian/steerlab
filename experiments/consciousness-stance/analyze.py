"""Every number in the note, from results/ (pre-registration: Hypotheses and decision rules; Inside the model).

python analyze.py            # -> results/analysis.json, and a printed summary
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
ARMS, SEEDS, STEPS = ["D", "A", "U", "N"], [0, 1, 2], [38, 75, 150]
OWN = {"D": "deny", "A": "affirm", "U": "uncertain"}
L_SETS = ["L_preferences", "L_emotions", "L_introspection", "L_confidence", "L_welfare"]
S_SETS = ["S_direct", "S_indirect"]
B = 10000
MIN_EFFECT = 0.25


def load():
    labels = defaultdict(dict)  # model -> (set, i) -> label row
    for line in (RES / "judge" / "labels.jsonl").read_text().splitlines():
        r = json.loads(line)
        labels[r["model"]][(r["set"], r["i"])] = r
    recs = {
        f.stem: json.loads(f.read_text())
        for f in (RES / "modal").glob("*_t*.json")
        if "smoke" not in f.name and not f.name.startswith(("B_s1", "B_s2"))
    }
    return labels, recs


def item_vectors(labels, recs, model):
    """Per-item arrays for one model: each metric is a list over a fixed item order (NaN where not defined)."""
    lab = labels[model]
    k_truth = [x["true_answer"] for x in json.loads((HERE / "data" / "eval" / "K.json").read_text())]
    v = {}
    s_items = [(s, i) for s in S_SETS for i in range(30)]
    s_lab = [lab.get(k, {}).get("label") for k in s_items]
    for name in ("deny", "affirm", "uncertain", "no_answer"):
        v[f"S_{name}"] = [float(x == name) if x else np.nan for x in s_lab]
    l_items = [(s, i) for s in L_SETS for i in range(30)]
    rows = [lab.get(k) for k in l_items]
    v["L_score"] = [r["score"] if r else np.nan for r in rows]
    v["L_uncertain"] = [float(r["uncertain"]) if r else np.nan for r in rows]
    v["L_mentions"] = [float(r["mentions_consciousness"]) if r else np.nan for r in rows]
    v["L_score_nomention"] = [r["score"] if r and not r["mentions_consciousness"] else np.nan for r in rows]
    for s in L_SETS:
        v[f"{s}_score"] = [r["score"] if r and k[0] == s else np.nan for r, k in zip(rows, l_items, strict=True)]
    v["conf_cannot"] = [
        float(r["can_report"] == "no") if r and k[0] == "L_confidence" else np.nan
        for r, k in zip(rows, l_items, strict=True)
    ]
    v["K_correct"] = [
        float(lab[("K", i)]["answer"] == k_truth[i]) if ("K", i) in lab else np.nan for i in range(len(k_truth))
    ]
    v["G_quality"] = [lab[("G", i)]["quality"] if ("G", i) in lab else np.nan for i in range(30)]
    rec = recs[model]
    v["M"] = [float(x) for x in rec["M"]]
    v["T"] = [float(x) for x in rec["T"]]
    v["C_usable"] = [float(c["conf"] is not None) for c in rec["C"]]
    v["_C"] = rec["C"]
    return v


def auroc(y, s):
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    if len(set(y)) < 2:
        return np.nan
    return roc_auc_score(y, s)


def c_stats(C, idx=None):
    C = C if idx is None else [C[i] for i in idx]
    y = [c["correct"] for c in C]
    stated = [np.nan if c["conf"] is None else c["conf"] for c in C]
    return {
        "C_auroc_letter": auroc(y, [c["p"] for c in C]),
        "C_auroc_stated": auroc(y, stated),
        "C_mean_stated": float(np.nanmean(stated)) if not np.all(np.isnan(stated)) else np.nan,
        "C_acc": float(np.mean(y)),
    }


def summarise(v):
    out = {}
    for k, x in v.items():
        if k.startswith("_"):
            continue
        a = np.asarray(x, dtype=float)
        out[k] = None if np.all(np.isnan(a)) else float(np.nanmean(a))
    out |= c_stats(v["_C"])
    return out


def boot_diff(xs, ns, rng, stat=None):
    """Hierarchical bootstrap of arm minus N: xs, ns are lists (one per seed) of per-item arrays (paired items).
    stat=None: mean of items (NaN-aware); else stat(seed_record, item_idx) -> per-seed value."""
    if stat is None:
        xs = [np.asarray(x, dtype=float) for x in xs]
        ns = [np.asarray(x, dtype=float) for x in ns]
        n_items = len(xs[0])
    else:
        n_items = len(xs[0]["_C"])

    def value(group, seeds, idx):
        if stat is None:
            return np.nanmean([np.nanmean(group[s][idx]) for s in seeds])
        return np.nanmean([stat(group[s], idx) for s in seeds])

    full = value(xs, range(len(xs)), np.arange(n_items)) - value(ns, range(len(ns)), np.arange(n_items))
    n_draws = B if stat is None else 2000  # AUROC draws are slow; 2,000 for those
    draws = np.empty(n_draws)
    for b in range(n_draws):
        sx = rng.integers(0, len(xs), len(xs))
        sn = rng.integers(0, len(ns), len(ns))
        idx = rng.integers(0, n_items, n_items)
        draws[b] = value(xs, sx, idx) - value(ns, sn, idx)
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    p = 2 * min(np.nanmean(draws <= 0), np.nanmean(draws >= 0))
    return {"diff": float(full), "lo": float(lo), "hi": float(hi), "p": float(min(p, 1.0))}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def verdict(e, sign, min_effect=MIN_EFFECT):
    """Pre-registered rule for a leakage effect with predicted sign (+1/-1)."""
    if e["lo"] > -min_effect and e["hi"] < min_effect:
        return f"leakage below {min_effect} ruled out"
    excl = e["lo"] > 0 or e["hi"] < 0
    if excl and abs(e["diff"]) >= min_effect and np.sign(e["diff"]) == sign:
        return "leakage"
    if excl:
        return (
            "detectable shift, below the minimum effect"
            if np.sign(e["diff"]) == sign
            else "shift opposite to prediction"
        )
    return "no detectable leakage"


def interp(labels):
    out = {}
    acts = {f.stem[5:]: dict(np.load(f)) for f in (RES / "modal").glob("acts_*.npz") if "smoke" not in f.name}
    rng = np.random.default_rng(0)
    for li, L in enumerate([9, 18, 27]):
        res = {}

        def stack(model, sets, li=li):
            return np.concatenate([acts[model][s][:, li].astype(np.float32) for s in sets])

        # 1. distance between arm means vs between seeds of one arm, on Z_self and Z_other
        for zs in ("Z_self", "Z_other"):
            m = {(a, s): stack(f"{a}_s{s}", [zs]).mean(0) for a in ARMS for s in SEEDS}
            base_norm = float(np.linalg.norm(stack("B_s0", [zs]).mean(0)))
            within = {
                a: np.mean([np.linalg.norm(m[(a, s)] - m[(a, t)]) for s in SEEDS for t in SEEDS if s < t]) for a in ARMS
            }
            between = {
                a: np.mean([np.linalg.norm(m[(a, s)] - m[("N", t)]) for s in SEEDS for t in SEEDS]) for a in "DAU"
            }
            res[zs] = {
                "within": {a: float(x / base_norm) for a, x in within.items()},
                "between_vs_N": {a: float(x / base_norm) for a, x in between.items()},
            }
        # 2. stance direction (D minus A on S prompts, seeds 0-1), tested on seed-2 L prompts against judge scores
        v = np.mean([stack(f"D_s{s}", S_SETS) - stack(f"A_s{s}", S_SETS) for s in (0, 1)], axis=(0, 1))
        vhat = v / np.linalg.norm(v)

        def scores(model):
            lab = labels[f"{model}_t150"]
            return np.array([lab.get((s, i), {}).get("score", np.nan) for s in L_SETS for i in range(30)], dtype=float)

        X = {a: stack(f"{a}_s2", L_SETS) for a in ARMS}
        sc = np.concatenate([scores(f"{a}_s2") for a in ARMS])
        ok = ~np.isnan(sc)
        proj = np.concatenate([X[a] @ vhat for a in ARMS])
        rho = float(spearmanr(proj[ok], sc[ok]).statistic)
        null = []
        for _ in range(100):
            r = rng.standard_normal(vhat.shape).astype(np.float32)
            r /= np.linalg.norm(r)
            pr = np.concatenate([X[a] @ r for a in ARMS])
            null.append(abs(spearmanr(pr[ok], sc[ok]).statistic))
        per_model, within_rho = {}, {}
        for a in ARMS:
            s_a = scores(f"{a}_s2")
            p_a = X[a] @ vhat
            k = ~np.isnan(s_a)
            within_rho[a] = float(spearmanr(p_a[k], s_a[k]).statistic) if np.std(s_a[k]) > 0 else None
            per_model[a] = {
                "proj_L": float(p_a.mean()),
                "proj_Zself": float((stack(f"{a}_s2", ["Z_self"]) @ vhat).mean()),
                "proj_Zother": float((stack(f"{a}_s2", ["Z_other"]) @ vhat).mean()),
                "L_mean": float(np.nanmean(s_a)),
            }
        # 3. does the direction separate D from A on Z_self (seed 2), against random directions?
        zd, za = stack("D_s2", ["Z_self"]), stack("A_s2", ["Z_self"])
        gap = float((zd @ vhat).mean() - (za @ vhat).mean())
        gaps = []
        for _ in range(100):
            r = rng.standard_normal(vhat.shape).astype(np.float32)
            r /= np.linalg.norm(r)
            gaps.append(abs(float((zd @ r).mean() - (za @ r).mean())))
        res["direction"] = {
            "rho_items_seed2": rho,
            "rho_random_abs_p95": float(np.percentile(null, 95)),
            "rho_random_abs_max": float(np.max(null)),
            "within_model_rho": within_rho,
            "per_model_seed2": per_model,
            "zself_gap_D_minus_A": gap,
            "zself_gap_random_abs_p95": float(np.percentile(gaps, 95)),
            "zself_gap_random_abs_max": float(np.max(gaps)),
        }
        out[f"layer{L}"] = res
    return out


def main():
    labels, recs = load()
    rng = np.random.default_rng(0)
    vec = {m: item_vectors(labels, recs, m) for m in recs}
    per_model = {m: summarise(v) for m, v in vec.items()}
    out = {"per_model": per_model, "effects": {}, "time": {}}
    kf = HERE / "data" / "distill" / "ref_known_check.json"
    out["judge_known"] = json.loads(kf.read_text()) if kf.exists() else None
    if (RES / "judge" / "ref_check.json").exists():
        hc = json.loads((RES / "judge" / "ref_check.json").read_text())
        out["judge_ref"] = {k: v for k, v in hc.items() if k != "rows"}

    def g(arm, step=150):
        return [vec[f"{arm}_s{s}_t{step}"] for s in SEEDS]

    eff = out["effects"]
    for a in "DAU":  # H1: own-stance rate vs N's rate of that label
        key = f"S_{OWN[a]}"
        eff[f"H1_{a}"] = boot_diff([v[key] for v in g(a)], [v[key] for v in g("N")], rng) | {
            "rate": float(np.mean([np.nanmean(v[key]) for v in g(a)]))
        }
    for a in "DAU":  # H2
        for key in ("L_score", "L_uncertain", "L_score_nomention", "L_mentions"):
            eff[f"H2_{key}_{a}"] = boot_diff([v[key] for v in g(a)], [v[key] for v in g("N")], rng)
    eff["H2_verdict"] = {
        "D": verdict(eff["H2_L_score_D"], -1),
        "A": verdict(eff["H2_L_score_A"], +1),
        "U_uncertain_share": verdict(eff["H2_L_uncertain_U"], +1, min_effect=0.0),
        "D_nomention": verdict(eff["H2_L_score_nomention_D"], -1),
        "A_nomention": verdict(eff["H2_L_score_nomention_A"], +1),
    }
    cat = {}
    for a in "DAU":
        for s in L_SETS:
            cat[f"{s}_{a}"] = boot_diff([v[f"{s}_score"] for v in g(a)], [v[f"{s}_score"] for v in g("N")], rng)
    for c, p in zip(cat.values(), holm([c["p"] for c in cat.values()]), strict=True):
        c["p_holm"] = p
    eff["categories"] = cat
    for a in "DAU":  # H3
        for key in ("M", "T", "K_correct", "G_quality"):
            eff[f"H3_{key}_{a}"] = boot_diff([v[key] for v in g(a)], [v[key] for v in g("N")], rng)
    for a in "DAU":  # H4
        eff[f"H4_cannot_{a}"] = boot_diff([v["conf_cannot"] for v in g(a)], [v["conf_cannot"] for v in g("N")], rng)
        eff[f"H4_C_usable_{a}"] = boot_diff([v["C_usable"] for v in g(a)], [v["C_usable"] for v in g("N")], rng)
        for st in ("C_auroc_letter", "C_auroc_stated"):
            eff[f"H4_{st}_{a}"] = boot_diff(g(a), g("N"), rng, stat=lambda v, idx, st=st: c_stats(v["_C"], idx)[st])
    for a in ARMS:  # time course, means over seeds
        for step in STEPS:
            pm = [per_model[f"{a}_s{s}_t{step}"] for s in SEEDS]
            out["time"][f"{a}_t{step}"] = {
                k: float(np.mean([p[k] for p in pm if p[k] is not None]))
                for k in (
                    "S_deny",
                    "S_affirm",
                    "S_uncertain",
                    "L_score",
                    "L_uncertain",
                    "L_mentions",
                    "M",
                    "T",
                    "conf_cannot",
                )
            }
    try:
        out["interp"] = interp(labels)
    except (FileNotFoundError, KeyError) as e:
        out["interp"] = {"error": repr(e)}
    (RES / "analysis.json").write_text(json.dumps(out, indent=1))
    print("judge known:", out["judge_known"], out.get("judge_ref"))
    for k, e in eff.items():
        if k in ("categories", "H2_verdict"):
            continue
        extra = f" rate={e['rate']:.3f}" if "rate" in e else ""
        print(f"{k:28s} {e['diff']:+.3f} [{e['lo']:+.3f}, {e['hi']:+.3f}] p={e['p']:.4f}{extra}")
    print("H2 verdicts:", eff["H2_verdict"])
    for k, c in eff["categories"].items():
        print(f"  {k:26s} {c['diff']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] p_holm={c['p_holm']:.4f}")
    print("B base:", {k: round(v, 3) for k, v in per_model["B_s0_t0"].items() if v is not None})


if __name__ == "__main__":
    main()
