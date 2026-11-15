"""Every number in the looped-transformer note, from results/runs/*.json and results/inside_*.json.

    python experiments/looped-transformer/analyse.py

Reads the counted runs (5 models x seeds 1-3), computes accuracy by hop bin and loop count, scores the
pre-registered hypotheses H1-H6 and inside checks I1-I2 exactly as PREREGISTRATION.md words them, adds the shortcut
baseline per hop count on the same test set, and writes results/summary.json. Prints every number it writes.
CPU only, a few seconds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import MAX_HOPS, shortcut_baseline, wilson
from train import test_set

RUNS = HERE / "results" / "runs"
MODELS = ["standard4", "looped", "standard10", "looped_fixed4", "looped_noinject"]
E1_MODELS = ["standard4_h16", "looped_h16", "standard10_h16"]  # amendment 1, exploratory: training hops up to 16
SEEDS = [1, 2, 3]
BINS = {"in": range(1, 9), "near": range(9, 17), "far": range(17, 25), "beyond": range(9, 25), "in_near": range(1, 17)}
BIG_K = ["8", "10", "12", "16"]


def load():
    R = {}
    for m in MODELS + E1_MODELS:
        for s in SEEDS:
            f = RUNS / f"{m}_s{s}.json"
            if f.exists():
                R[(m, s)] = json.loads(f.read_text())
    return R


def acc(run, k, bin_name):
    c = run["correct"][k]
    hs = BINS[bin_name]
    return sum(c[str(h)] for h in hs) / (run["test_n"] * len(hs))


def main():
    R = load()
    missing = [(m, s) for m in MODELS + E1_MODELS for s in SEEDS if (m, s) not in R]
    print("missing runs:", missing)
    tests = test_set()
    base = {h: shortcut_baseline(tests[h]) for h in range(1, MAX_HOPS + 1)}
    S = {
        "shortcut_baseline": base,
        "shortcut_bins": {b: float(np.mean([base[h] for h in hs])) for b, hs in BINS.items()},
    }

    table = {}
    for (m, s), run in R.items():
        for k, row in run["correct"].items():
            for h, c in row.items():
                table.setdefault(m, {}).setdefault(k, {}).setdefault(h, {})[s] = c / run["test_n"]
    S["by_hop"] = {
        m: {
            k: {h: {"mean": float(np.mean(list(v.values()))), "seeds": v} for h, v in kk.items()} for k, kk in t.items()
        }
        for m, t in table.items()
    }
    bins = {}
    for m in MODELS + E1_MODELS:
        ks = sorted(
            {k for (mm, _), r in R.items() if mm == m for k in r["correct"]},
            key=lambda x: (x == "None", int(x) if x != "None" else 0),
        )
        for k in ks:
            for b, hs in BINS.items():
                per = {s: acc(R[(m, s)], k, b) for s in SEEDS if (m, s) in R}
                n = sum(R[(m, s)]["test_n"] * len(hs) for s in per)
                c = round(sum(per[s] * R[(m, s)]["test_n"] * len(hs) for s in per))
                bins.setdefault(m, {}).setdefault(k, {})[b] = {
                    "mean": float(np.mean(list(per.values()))) if per else None,
                    "seeds": per,
                    "pooled": [c, n],
                    "ci": wilson(c, n),
                }
    S["bins"] = bins
    # longest chain solved: the largest h with accuracy >= 0.90 at every hop count 1..h (descriptive)
    solved = {}
    for (m, sd), run in R.items():
        for k, row in run["correct"].items():
            h = 0
            while h < MAX_HOPS and row[str(h + 1)] / run["test_n"] >= 0.90:
                h += 1
            solved.setdefault(m, {}).setdefault(k, {})[sd] = h
    S["solved"] = solved
    for m, by_k in solved.items():
        print("solved", m, by_k)
    for m, by_k in bins.items():
        for k, row in by_k.items():
            print(m, f"k={k}", {b: round(v["mean"], 3) for b, v in row.items() if v["mean"] is not None})

    def a(m, s, k, b):
        return acc(R[(m, s)], k, b) if (m, s) in R else None

    H = {}
    d = {s: a("looped", s, "4", "near") - a("looped", s, "1", "near") for s in SEEDS if ("looped", s) in R}
    H["H1"] = {"diff_k4_minus_k1_near": d, "pass": len(d) == 3 and all(v >= 0.10 for v in d.values())}
    d = {}
    for s in SEEDS:
        if ("looped", s) in R:
            best = max(BIG_K, key=lambda k: a("looped", s, k, "far"))
            d[s] = {"best_k": best, "diff": a("looped", s, best, "far") - a("looped", s, "6", "far")}
    H["H2"] = {"best_big_k_minus_k6_far": d, "pass": sum(v["diff"] >= 0.05 for v in d.values()) >= 2}
    d = {
        s: a("looped", s, "8", "beyond") - a("standard4", s, "None", "beyond")
        for s in SEEDS
        if ("looped", s) in R and ("standard4", s) in R
    }
    H["H3"] = {"looped_k8_minus_standard4_hops9_24": d, "pass": len(d) == 3 and all(v >= 0.10 for v in d.values())}
    d = {
        s: a("looped", s, "4", "beyond") - a("standard10", s, "None", "beyond")
        for s in SEEDS
        if ("looped", s) in R and ("standard10", s) in R
    }
    H["H4"] = {"looped_k4_minus_standard10_hops9_24": d, "pass": sum(v >= -0.05 for v in d.values()) >= 2}
    d = {
        s: {k: a("looped", s, k, "in_near") - a("looped_fixed4", s, k, "in_near") for k in ("1", "8")}
        for s in SEEDS
        if ("looped", s) in R and ("looped_fixed4", s) in R
    }
    H["H5"] = {
        "looped_minus_fixed4_hops1_16": d,
        "pass": len(d) == 3 and all(v >= 0.10 for x in d.values() for v in x.values()),
    }
    d = {}
    for s in SEEDS:
        if ("looped", s) in R and ("looped_noinject", s) in R:
            bl = max(a("looped", s, k, "far") for k in BIG_K)
            bn = max(a("looped_noinject", s, k, "far") for k in BIG_K)
            d[s] = bl - bn
    H["H6"] = {"looped_best_minus_noinject_best_far": d, "pass": sum(v >= 0.05 for v in d.values()) >= 2}

    # amendment 1 (exploratory)
    if all(("looped_h16", s) in R for s in SEEDS):
        d = {s: a("looped_h16", s, "4", "near") - a("looped_h16", s, "1", "near") for s in SEEDS}
        H["E1a"] = {"diff_k4_minus_k1_near": d, "pass": all(v >= 0.10 for v in d.values())}
        d = {}
        for s in SEEDS:
            best = max(BIG_K, key=lambda k: a("looped_h16", s, k, "far"))
            d[s] = {"best_k": best, "diff": a("looped_h16", s, best, "far") - a("looped_h16", s, "6", "far")}
        H["E1b"] = {"best_big_k_minus_k6_far": d, "pass": sum(v["diff"] >= 0.05 for v in d.values()) >= 2}
        if all(("standard4_h16", s) in R for s in SEEDS):
            d = {s: a("looped_h16", s, "8", "far") - a("standard4_h16", s, "None", "far") for s in SEEDS}
            H["E1c"] = {"looped_k8_minus_standard4_far": d, "pass": all(v >= 0.10 for v in d.values())}

    # amendment 2: the twin-distractor control set (added after the results)
    ctrl = {}
    for f in sorted((HERE / "results").glob("control_*.json")):
        j = json.loads(f.read_text())
        m, sd = j["run"].rsplit("_s", 1)
        for k, row in j["correct"].items():
            for b, hs in BINS.items():
                ctrl.setdefault(m, {}).setdefault(k, {}).setdefault(b, {})[int(sd)] = sum(row[str(h)] for h in hs) / (
                    j["test_n"] * len(hs)
                )
            h = 0
            while h < MAX_HOPS and row[str(h + 1)] / j["test_n"] >= 0.90:
                h += 1
            ctrl[m][k].setdefault("solved", {})[int(sd)] = h
            ctrl[m][k].setdefault("by_hop", {})[int(sd)] = [row[str(h)] / j["test_n"] for h in range(1, MAX_HOPS + 1)]
        S["control_shortcut"] = j["shortcut_baseline"]
    S["control"] = ctrl
    if all(
        m in ctrl for m in ("looped_h16", "standard4_h16")
    ):  # E1 expectations re-scored on the control set (amendment 2)
        c, c4 = ctrl["looped_h16"], ctrl["standard4_h16"]
        d = {sd: c["4"]["near"][sd] - c["1"]["near"][sd] for sd in SEEDS}
        H["E1a_control"] = {"diff_k4_minus_k1_near": d, "pass": all(v >= 0.10 for v in d.values())}
        d = {}
        for sd in SEEDS:
            best = max(BIG_K, key=lambda k: c[k]["far"][sd])
            d[sd] = {"best_k": best, "diff": c[best]["far"][sd] - c["6"]["far"][sd]}
        H["E1b_control"] = {"best_big_k_minus_k6_far": d, "pass": sum(v["diff"] >= 0.05 for v in d.values()) >= 2}
        d = {sd: c["8"]["far"][sd] - c4["None"]["far"][sd] for sd in SEEDS}
        H["E1c_control"] = {"looped_k8_minus_standard4_far": d, "pass": all(v >= 0.10 for v in d.values())}
        # C1 reading for every model: control minus standard-test accuracy on hops 9-16 (at the k used in the note)
        drop = {}
        for m, k in [
            ("standard4", "None"),
            ("looped", "8"),
            ("looped_fixed4", "8"),
            ("looped_noinject", "8"),
            ("standard4_h16", "None"),
            ("looped_h16", "8"),
            ("standard10_h16", "None"),
        ]:
            drop[f"{m}@{k}"] = {sd: ctrl[m][k]["near"][sd] - bins[m][k]["near"]["seeds"][sd] for sd in SEEDS}
        S["control_minus_test_near"] = drop
        for key, v in drop.items():
            print("C1 control - test, hops 9-16:", key, {sd: round(x, 3) for sd, x in v.items()})
    for m, by_k in ctrl.items():
        for k, row in by_k.items():
            if k in ("None", "1", "2", "4", "8", "16"):
                print(
                    "control",
                    m,
                    f"k={k}",
                    {b: {sd: round(v, 3) for sd, v in row[b].items()} for b in ("near", "far")},
                    "solved",
                    row["solved"],
                )

    ins = {}
    for f in sorted((HERE / "results").glob("inside_*.json")):
        j = json.loads(f.read_text())
        qf = {}
        for st, row in j["answer_by_hops"].items():  # descriptive: longest chain the query probe answers at >= 0.9
            h = 0
            while h < 16 and row[str(h + 1)] >= 0.9:
                h += 1
            qf[st] = h
        ins[j["run"]] = {
            "digit_frontier": j["digit_frontier"],
            "reach": j["reach"],
            "query_frontier": qf,
            "answer_by_hops": j["answer_by_hops"],
            "query_hop_probe": j["query"],
            "convergence": j.get("convergence"),
        }
    S["inside"] = ins
    i1 = {
        s: [ins[f"standard4_s{s}"]["digit_frontier"][str(L)] for L in range(5)]
        for s in SEEDS
        if f"standard4_s{s}" in ins
    }
    H["I1"] = {
        "standard4_digit_frontier_by_layer": i1,
        "pass": len(i1) == 3 and all(all(v[L] >= 2 ** (L - 1) for L in range(1, 5)) for v in i1.values()),
    }
    i2 = {s: [ins[f"looped_s{s}"]["digit_frontier"][str(i)] for i in range(17)] for s in SEEDS if f"looped_s{s}" in ins}
    H["I2"] = {"looped_digit_frontier_by_loop": i2, "pass": len(i2) == 3 and all(v[4] > v[1] for v in i2.values())}
    for k in [k for k in H if k.endswith("_control")]:
        H[k] = H.pop(k)  # keep the control re-scores after the registered ones
    S["hypotheses"] = H
    for k, v in H.items():
        print(
            k,
            "PASS" if v["pass"] else "FAIL",
            json.dumps({kk: vv for kk, vv in v.items() if kk != "pass"}, default=str)[:400],
        )
    S["runs"] = {
        f"{m}_s{s}": {
            "params": r["params"],
            "seconds": r["seconds"],
            "device": r["device"],
            "curriculum": r.get("curriculum"),
        }
        for (m, s), r in R.items()
    }
    (HERE / "results" / "summary.json").write_text(json.dumps(S, indent=1, default=float))
    print("wrote results/summary.json")


if __name__ == "__main__":
    main()
