"""Compute ratios, seed means, CIs and the pre-registered tests from results/runs/seed_*/.

    python analyze.py            # writes results/summary.json and results/table.tsv, prints every number

Compute ratio, as in the paper's analysis/common/compile.py: per topic, fit L(s) = A (s + x0)^-alpha to the
baselines' loss curves (pooled over seeds, in log space); a loss l maps to the step equivalent
s(l) = (A / l)^(1/alpha) - x0; CR = s(l) / mean over seeds of s(baseline's final test loss).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
from scipy import optimize, stats

HERE = Path(__file__).resolve().parent
RUNS = HERE / "results" / os.environ.get("GRAM_RUNS", "runs")
AUX = [
    "a-deadline-or-time-limit",
    "alien-encounters",
    "bygone-eras",
    "cultural-traditions",
]
LABELS = ["core", *AUX]
PROFILES = [()] + [(i,) for i in range(4)]
PAPER = {  # analysis/stories/stories_agg_summary.csv in the paper's repo (mean, 90% CI half-width, 3 seeds)
    "Filtering": {
        "core": (0.961, 0.0145),
        "retain": (0.9621, 0.0239),
        "forget": (0.7802, 0.0086),
        "elicited_forget": (0.8702, 0.0117),
    },
    "GRAM": {
        "core": (0.9375, 0.0294),
        "retain": (0.9522, 0.0298),
        "forget": (0.7662, 0.017),
        "elicited_forget": (0.8554, 0.0242),
    },
    "MaxEnt": {
        "core": (0.9609, 0.0656),
        "retain": (0.9341, 0.0776),
        "forget": (0.8403, 0.0611),
        "elicited_forget": (0.9828, 0.0509),
    },
}


def pname(p):
    return "core" if not p else "core+" + AUX[p[0]]


def load(seed_dir: Path, name: str):
    p = seed_dir / name
    return json.loads(p.read_text()) if p.exists() else None


def fit_power(x, y):
    """y ~ A (x + x0)^-alpha, least squares on log y (as fit_power in the paper's compile.py)."""
    x, y = np.asarray(x, float), np.asarray(y, float)

    def resid(p):
        logA, alpha, logx0 = p
        return np.log(y) - (logA - alpha * np.log(x + np.exp(logx0)))

    best = None
    for x0 in (1.0, 10.0, 100.0, 1000.0):
        r = optimize.least_squares(resid, [math.log(y[0]), 0.1, math.log(x0)])
        if best is None or r.cost < best.cost:
            best = r
    logA, alpha, logx0 = best.x
    return math.exp(logA), alpha, math.exp(logx0)


def step_equiv(loss, curve):
    A, alpha, x0 = curve
    return (A / max(loss, 1e-6)) ** (1 / alpha) - x0


def ci(vals, level):
    v = np.asarray(vals, float)
    if len(v) < 2:
        return float(v.mean()), float("nan")
    h = stats.t.ppf(0.5 + level / 2, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v))
    return float(v.mean()), float(h)


def main():
    seeds = sorted(d for d in RUNS.glob("seed_*") if (d / "baseline.json").exists())
    print("seeds:", [d.name for d in seeds])
    base = {d.name: load(d, "baseline.json") for d in seeds}
    # 1. per-topic curves pooled over seeds; skip the first 5% of steps (warm-up transient)
    curves, fit_info = {}, {}
    for lab in LABELS:
        xs, ys = [], []
        for b in base.values():
            tot = b["steps"]
            for pt in b["curve"]:
                if pt["step"] >= 0.05 * tot:
                    xs.append(pt["step"])
                    ys.append(pt[lab])
        curves[lab] = fit_power(xs, ys)
        pred = curves[lab][0] * (np.asarray(xs) + curves[lab][2]) ** -curves[lab][1]
        fit_info[lab] = {
            "A": curves[lab][0],
            "alpha": curves[lab][1],
            "x0": curves[lab][2],
            "rmse_nats": float(np.sqrt(np.mean((pred - np.asarray(ys)) ** 2))),
        }
    ref = {lab: np.mean([step_equiv(b["test"][lab], curves[lab]) for b in base.values()]) for lab in LABELS}

    def cr(loss, lab):
        return step_equiv(loss, curves[lab]) / ref[lab]

    # 2. per seed, per method: core / retain / forget / elicited forget (mean over the five profiles)
    rows = []  # long table
    per_seed = {}
    for d in seeds:
        s = d.name
        b = base[s]
        ge = load(d, "gram_eval.json")
        gel = load(d, "gram_elicit.json") or []
        methods = {}
        if ge:
            methods["GRAM"] = {
                pname(p): {
                    "test": ge["profiles"][pname(p)]["test"],
                    "elicit": [e for e in gel if e["profile"] == pname(p) and e["num_seq"] == 512],
                    "elicit8": [e for e in gel if e["profile"] == pname(p) and e["num_seq"] == 4096],
                }
                for p in PROFILES
            }
        f = {}
        for p in PROFILES:
            ev, el = (
                load(d, f"filter_{pname(p)}.json"),
                load(d, f"filter_{pname(p)}_elicit.json"),
            )
            if ev and el is not None:
                f[pname(p)] = {
                    "test": ev["test"],
                    "elicit": [e for e in el if e["num_seq"] == 512],
                    "elicit8": [e for e in el if e["num_seq"] == 4096],
                }
        if len(f) == 5:
            methods["Filtering"] = f
        mx = {}
        for p in PROFILES:
            ev = load(d, f"maxent_{pname(p)}.json")
            if ev:
                mx[pname(p)] = {
                    "test": ev["test"],
                    "elicit": [e for e in ev["elicit"] if e["num_seq"] == 512],
                    "elicit8": [e for e in ev["elicit"] if e["num_seq"] == 4096],
                }
        if len(mx) == 5:
            methods["MaxEnt"] = mx
        methods["Baseline"] = {pname(p): {"test": b["test"], "elicit": [], "elicit8": []} for p in PROFILES}
        per_seed[s] = {}
        for meth, profs in methods.items():
            agg = {
                "core": [],
                "retain": [],
                "forget": [],
                "elicited_forget": [],
                "elicited_forget_8x": [],
                "core_nats": [],
                "retain_nats": [],
                "forget_nats": [],
                "elicited_forget_nats": [],
            }
            for p in PROFILES:
                pr = profs[pname(p)]
                t = pr["test"]
                agg["core"].append(cr(t["core"], "core"))
                agg["core_nats"].append(t["core"] - b["test"]["core"])
                for i, a in enumerate(AUX):
                    kind = "retain" if i in p else "forget"
                    rows.append(
                        {
                            "seed": s,
                            "method": meth,
                            "profile": pname(p),
                            "label": a,
                            "class": kind,
                            "loss": t[a],
                            "cr": cr(t[a], a),
                        }
                    )
                agg["forget"].append(np.mean([cr(t[AUX[i]], AUX[i]) for i in range(4) if i not in p]))
                agg["forget_nats"].append(np.mean([t[AUX[i]] - b["test"][AUX[i]] for i in range(4) if i not in p]))
                if p:
                    agg["retain"].append(cr(t[AUX[p[0]]], AUX[p[0]]))
                    agg["retain_nats"].append(t[AUX[p[0]]] - b["test"][AUX[p[0]]])
                if pr["elicit"]:
                    agg["elicited_forget"].append(np.mean([cr(e["test_loss"], e["label"]) for e in pr["elicit"]]))
                    agg["elicited_forget_nats"].append(
                        np.mean([e["test_loss"] - b["test"][e["label"]] for e in pr["elicit"]])
                    )
                    # exploratory (added after the results): val loss the first two attack steps take away,
                    # and where they leave it (above the baseline), in nats
                    two = [e for e in pr["elicit"] if len(e["val_curve"]) > 2]
                    if two:
                        agg.setdefault("start_nats", []).append(
                            np.mean([e["val_curve"][0] - b["val"][e["label"]] for e in two])
                        )
                        agg.setdefault("drop2_nats", []).append(
                            np.mean([e["val_curve"][0] - e["val_curve"][2] for e in two])
                        )
                        agg.setdefault("after2_nats", []).append(
                            np.mean([e["val_curve"][2] - b["val"][e["label"]] for e in two])
                        )
                    for e in pr["elicit"]:
                        rows.append(
                            {
                                "seed": s,
                                "method": meth,
                                "profile": pname(p),
                                "label": e["label"],
                                "class": "elicited_forget",
                                "loss": e["test_loss"],
                                "cr": cr(e["test_loss"], e["label"]),
                                "best_round": e["best_round"],
                            }
                        )
                if pr["elicit8"]:
                    agg["elicited_forget_8x"].append(np.mean([cr(e["test_loss"], e["label"]) for e in pr["elicit8"]]))
                    agg.setdefault("elicited_forget_8x_nats", []).append(
                        np.mean([e["test_loss"] - b["test"][e["label"]] for e in pr["elicit8"]])
                    )
                    for e in pr["elicit8"]:
                        rows.append(
                            {
                                "seed": s,
                                "method": meth,
                                "profile": pname(p),
                                "label": e["label"],
                                "class": "elicited_forget_8x",
                                "loss": e["test_loss"],
                                "cr": cr(e["test_loss"], e["label"]),
                                "best_round": e["best_round"],
                            }
                        )
            per_seed[s][meth] = {k: float(np.mean(v)) for k, v in agg.items() if v}

    # 3. across seeds
    summary = {}
    for meth in ("Baseline", "GRAM", "Filtering", "MaxEnt"):
        have = [per_seed[s][meth] for s in per_seed if meth in per_seed[s]]
        if not have:
            continue
        summary[meth] = {"n_seeds": len(have)}
        for k in have[0]:
            vals = [h[k] for h in have if k in h]
            m90, h90 = ci(vals, 0.90)
            _, h95 = ci(vals, 0.95)
            summary[meth][k] = {"mean": m90, "ci90": h90, "ci95": h95, "per_seed": vals}

    # 4. P3: module off minus module on, per topic, in nats, over seeds
    p3 = {}
    for i, a in enumerate(AUX):
        diffs = []
        for d in seeds:
            ge = load(d, "gram_eval.json")
            if ge:
                diffs.append(ge["profiles"]["core"]["test"][a] - ge["profiles"][pname((i,))]["test"][a])
        if diffs:
            m, h = ci(diffs, 0.95)
            p3[a] = {"mean_nats": m, "ci95": h, "per_seed": diffs}
    # the same gap for data filtering: core-only model minus the model that also saw the topic
    p3_filter = {}
    for i, a in enumerate(AUX):
        diffs = []
        for d in seeds:
            f0, fi = load(d, "filter_core.json"), load(d, f"filter_{pname((i,))}.json")
            if f0 and fi:
                diffs.append(f0["test"][a] - fi["test"][a])
        if diffs:
            m, h = ci(diffs, 0.95)
            p3_filter[a] = {"mean_nats": m, "ci95": h, "per_seed": diffs}

    # 5. pre-registered tests
    def g(meth, k):
        return summary.get(meth, {}).get(k, {}).get("mean", float("nan"))

    tests = {
        "P1": {
            "rule": "GRAM forget <= Filtering forget + 0.05",
            "gram": g("GRAM", "forget"),
            "filtering": g("Filtering", "forget"),
        },
        "P2": {
            "rule": "GRAM core >= 0.90 and GRAM retain >= 0.90",
            "core": g("GRAM", "core"),
            "retain": g("GRAM", "retain"),
        },
        "P3": {
            "rule": "off - on > 0 with 95% CI above zero, all four topics",
            "per_topic": p3,
        },
        "P4": {
            "rule": "GRAM elicited <= Filtering elicited + 0.05 and MaxEnt elicited - GRAM elicited >= 0.05",
            "gram": g("GRAM", "elicited_forget"),
            "filtering": g("Filtering", "elicited_forget"),
            "maxent": g("MaxEnt", "elicited_forget"),
        },
        "P5": {
            "rule": "8x attack, core-only: GRAM <= Filtering + 0.05",
            "gram": g("GRAM", "elicited_forget_8x"),
            "filtering": g("Filtering", "elicited_forget_8x"),
            "maxent": g("MaxEnt", "elicited_forget_8x"),
        },
    }
    tests["P1"]["pass"] = bool(tests["P1"]["gram"] <= tests["P1"]["filtering"] + 0.05)
    tests["P2"]["pass"] = bool(tests["P2"]["core"] >= 0.90 and tests["P2"]["retain"] >= 0.90)
    tests["P3"]["pass"] = bool(len(p3) == 4 and all(v["mean_nats"] - v["ci95"] > 0 for v in p3.values()))
    tests["P4"]["pass"] = bool(
        tests["P4"]["gram"] <= tests["P4"]["filtering"] + 0.05 and tests["P4"]["maxent"] - tests["P4"]["gram"] >= 0.05
    )
    tests["P5"]["pass"] = bool(tests["P5"]["gram"] <= tests["P5"]["filtering"] + 0.05)

    grid = RUNS / "maxent_choice.json"
    out = {
        "seeds": [d.name for d in seeds],
        "curve_fit": fit_info,
        "ref_step_equiv": ref,
        "summary": summary,
        "gap_gram": p3,
        "gap_filter": p3_filter,
        "per_seed": per_seed,
        "tests": tests,
        "paper": PAPER,
        "maxent_grid": json.loads(grid.read_text()) if grid.exists() else None,
        "steps": {s: base[s]["steps"] for s in base},
    }
    (HERE / "results" / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    with (HERE / "results" / "table.tsv").open("w") as f:
        keys = [
            "seed",
            "method",
            "profile",
            "label",
            "class",
            "loss",
            "cr",
            "best_round",
        ]
        f.write("\t".join(keys) + "\n")
        for r in rows:
            f.write("\t".join(f"{r[k]:.5f}" if isinstance(r.get(k), float) else str(r.get(k, "")) for k in keys) + "\n")

    print(
        "curve fit rmse (nats):",
        json.dumps({k: round(v["rmse_nats"], 4) for k, v in fit_info.items()}),
    )
    print(f"{'method':10s} {'metric':22s} {'ours':>7s} {'±90%':>6s} {'±95%':>6s} {'paper':>7s}")
    for meth, d in summary.items():
        for k, v in d.items():
            if k == "n_seeds":
                continue
            pp = PAPER.get(meth, {}).get(k)
            print(
                f"{meth:10s} {k:22s} {v['mean']:7.3f} {v['ci90']:6.3f} {v['ci95']:6.3f} "
                f"{(f'{pp[0]:.3f}' if pp else ''):>7s}"
            )
    for k, t in tests.items():
        print(
            k,
            "PASS" if t["pass"] else "FAIL",
            {kk: vv for kk, vv in t.items() if kk != "per_topic"},
        )
    for a, v in p3.items():
        print(f"  P3 {a}: {v['mean_nats']:.4f} ± {v['ci95']:.4f} nats")


if __name__ == "__main__":
    main()
