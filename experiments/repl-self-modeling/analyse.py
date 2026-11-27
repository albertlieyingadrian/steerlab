"""Analysis for the self-modeling replication: SGLD step choice, per-arm summaries, differences with CIs, predictions.

    .venv/bin/python experiments/repl-self-modeling/analyse.py calib     # SGLD step -> results/calibration.json
    .venv/bin/python experiments/repl-self-modeling/analyse.py summary   # results/summary.json + printed tables

Intervals: 95% t-intervals over seeds for means; Welch intervals for differences of means. A run is "collapsed" when
its MNIST test accuracy is below 0.90 (IMDB: below 0.60); collapsed runs are counted and left out of the means
(plan, boring explanation c).
"""

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).parent
RESULTS = HERE / "results"
RUNS = RESULTS / "runs"
NBETA = {"mnist": 60000 / math.log(60000), "imdb": 25000 / math.log(25000)}
MEASURES = [
    "width",
    "llc_main",
    "llc_sens",
    "test_acc",
    "train_loss",
    "mean_abs_act",
    "dead_share",
    "hidden_w_sd",
    "width_x_act",
]
DIFF_MEASURES = ("width", "llc_main", "llc_sens", "test_acc", "width_x_act", "mean_abs_act", "dead_share")


def load(stage: str) -> list:
    d = RUNS / stage
    return (
        [json.loads(p.read_text()) for p in sorted(d.glob("*.json")) if not p.name.startswith("_")]
        if d.exists()
        else []
    )


def calib():
    """Plan amendment 1: among steps whose chains agree (CV < 0.2) and whose LLC is finite and positive, the smallest
    step whose LLC is within 20% of the next larger step's (the plateau); if none plateaus, the largest stable step.
    The drift between the first and last quarter of the draws is reported beside it."""
    out = {}
    by_task = {}
    for res in load("calib") + load("calib2"):
        task = res["cfg"]["task"]
        ck = res["checkpoints"][str(res["cfg"]["epochs"])]
        by_task.setdefault(task, {"test_acc": ck["test_acc"], "ck": {}})["ck"].update(ck)
    for task, v in by_task.items():
        ck = v["ck"]
        rows = {}
        names = sorted(
            {k[4:-7] for k in ck if k.startswith("llc_") and k.endswith("_chains")}, key=lambda n: float(n[2:])
        )
        for name in names:
            llc = ck[f"llc_{name}"]
            chains = ck[f"llc_{name}_chains"]
            tr = np.array(ck[f"llc_{name}_trace"])  # every 10th step; burn-in 300 steps = first 30 points
            d = tr[30:]
            q = max(1, len(d) // 4)
            drift = NBETA[task] * abs(d[-q:].mean() - d[:q].mean()) / max(abs(llc), 1e-9)
            cv = float(np.std(chains) / max(abs(np.mean(chains)), 1e-9))
            ok = bool(np.isfinite(llc) and llc > 0 and cv < 0.2)
            rows[name] = {"lr": float(name[2:]), "llc": llc, "chain_cv": cv, "drift": float(drift), "stable": ok}
        good = [r for r in rows.values() if r["stable"]]
        chosen, rule = None, None
        for a, b in itertools.pairwise(good):
            if abs(b["llc"] - a["llc"]) <= 0.2 * abs(b["llc"]):
                chosen, rule = a["lr"], "plateau"
                break
        if chosen is None and good:
            chosen, rule = good[-1]["lr"], "largest stable (no plateau)"
        out[task] = {"grid": rows, "chosen_lr": chosen, "rule": rule, "test_acc": v["test_acc"]}
        print(task, json.dumps(rows, indent=1), "chosen", chosen, rule)
    (RESULTS / "calibration.json").write_text(json.dumps(out, indent=1))


def ci_mean(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return [float(x.mean()) if len(x) else None, [None, None]]
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
    return [float(x.mean()), [float(x.mean() - h), float(x.mean() + h)]]


def ci_diff(a, b):
    """Welch interval for mean(a) - mean(b)."""
    a = np.asarray([v for v in a if v is not None], float)
    b = np.asarray([v for v in b if v is not None], float)
    if len(a) < 2 or len(b) < 2:
        return [None, [None, None]]
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    d = a.mean() - b.mean()
    se = math.sqrt(va + vb)
    if se == 0:
        return [float(d), [float(d), float(d)]]
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    h = stats.t.ppf(0.975, df) * se
    return [float(d), [float(d - h), float(d + h)]]


def arm_key(cfg):
    t = "none" if cfg["aw"] == 0 else cfg["target"]
    base = f"{cfg['task']}|h{cfg['hidden']}|aw{cfg['aw']}|{t}"
    return base + (f"|l2{cfg['l2']}" if cfg.get("l2") else "")


def collect(runs, epoch=None):
    arms = {}
    for r in runs:
        cfg = r["cfg"]
        ck = dict(r["checkpoints"][str(epoch or cfg["epochs"])])
        if cfg["task"] == "cifar":  # amendment 2: no plateau on the baseline arm; largest stable step 1e-6, sens 3e-7
            ck["llc_main"], ck["llc_sens"] = ck.get("llc_lr1e-06"), ck.get("llc_lr3e-07")
        row = {"seed": cfg["seed"], **{m: ck.get(m) for m in MEASURES}, "width_by_epoch": r["width_by_epoch"]}
        arms.setdefault(arm_key(cfg), []).append(row)
    return arms


def summarise(arms, acc_floor):
    out = {}
    for k, rows in sorted(arms.items()):
        ok = [r for r in rows if r["test_acc"] >= acc_floor]
        out[k] = {"n": len(rows), "collapsed": len(rows) - len(ok)}
        for m in MEASURES:
            vals = [r[m] for r in ok if r[m] is not None]
            out[k][m] = ci_mean(vals) if vals else None
        out[k]["test_acc_all"] = ci_mean([r["test_acc"] for r in rows])
        out[k]["_acc_all"] = [r["test_acc"] for r in rows]
        out[k]["width_by_epoch"] = np.mean([r["width_by_epoch"] for r in ok], 0).tolist() if ok else None
        out[k]["_vals"] = {m: [r[m] for r in ok] for m in MEASURES}
    return out


def diffs(summ, base_of):
    d = {}
    for k, v in summ.items():
        b = base_of(k)
        if b not in summ or b == k:
            continue
        d[k] = {m: ci_diff(v["_vals"][m], summ[b]["_vals"][m]) for m in DIFF_MEASURES}
        d[k]["test_acc_all"] = ci_diff(v["_acc_all"], summ[b]["_acc_all"])
        d[k]["baseline"] = b
    return d


def strip(s):
    return {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in s.items()}


def summary():
    out = {}
    mn = summarise(collect(load("mnist") + load("mnist_ctrl")), 0.90)
    out["mnist"] = strip(mn)
    out["mnist_diff"] = diffs(mn, lambda k: f"mnist|{k.split('|')[1]}|aw0|none")
    imdb_runs = load("imdb")
    ims = {}
    for ep in (250, 500):
        if not imdb_runs:
            break
        ims[ep] = summarise(collect(imdb_runs, ep), 0.60)
        out[f"imdb_e{ep}"] = strip(ims[ep])
        out[f"imdb_e{ep}_diff"] = diffs(ims[ep], lambda k: "imdb|h64|aw0|none")
        s = ims[ep]
        if "imdb|h64|aw500|self" in s and "imdb|h64|aw100|self" in s:
            out[f"imdb_e{ep}_500_vs_100"] = {
                m: ci_diff(s["imdb|h64|aw500|self"]["_vals"][m], s["imdb|h64|aw100|self"]["_vals"][m])
                for m in DIFF_MEASURES
            }
    cifar_runs = load("cifar")
    if cifar_runs:
        cf = summarise(collect(cifar_runs), 0.50)
        out["cifar"] = strip(cf)
        out["cifar_diff"] = diffs(cf, lambda k: "cifar|h256|aw0|none")
    out["llc_vs_train_loss"] = llc_confound(load("mnist") + load("mnist_ctrl"))
    out["predictions"] = score_predictions(mn, out)
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=1))
    print_tables(out)


def llc_confound(runs):
    """Boring explanation found after the runs (not pre-registered): does the LLC estimate just track how well the
    model fits its training set? Spearman over every non-collapsed MNIST run, overall and within each hidden size."""
    rows = []
    for r in runs:
        ck = r["checkpoints"][str(r["cfg"]["epochs"])]
        if ck["test_acc"] >= 0.90:
            rows.append((r["cfg"]["hidden"], ck["llc_main"], ck["train_loss"], ck["dead_share"], arm_key(r["cfg"])))
    out = {"n": len(rows)}
    llc, tl, dead = (np.array([x[i] for x in rows]) for i in (1, 2, 3))
    out["spearman_llc_train_loss"] = float(stats.spearmanr(llc, tl).statistic)
    out["spearman_llc_dead_share"] = float(stats.spearmanr(llc, dead).statistic)
    out["by_hidden"] = {}
    for h in (64, 128, 256, 512):
        sel = [x for x in rows if x[0] == h]
        if len(sel) > 3:
            out["by_hidden"][str(h)] = float(stats.spearmanr([x[1] for x in sel], [x[2] for x in sel]).statistic)
    out["points"] = [{"hidden": x[0], "llc": x[1], "train_loss": x[2], "dead_share": x[3], "arm": x[4]} for x in rows]
    return out


def _upper_below_0(d):
    return bool(d and d[1][1] is not None and d[1][1] < 0)


def score_predictions(mn, out):
    p = {}
    cells_w, cells_l = [], []
    for h in (64, 128, 256, 512):
        for aw in (1, 5, 10, 20, 50):
            d = out["mnist_diff"].get(f"mnist|h{h}|aw{aw}|self")
            if d:
                cells_w.append(_upper_below_0(d["width"]))
                cells_l.append(_upper_below_0(d["llc_main"]))
    rho = {}
    for h in (64, 128, 256, 512):
        xs, ys = [], []
        for aw in (0, 1, 5, 10, 20, 50):
            k = f"mnist|h{h}|aw{aw}|{'self' if aw else 'none'}"
            if k in mn and mn[k]["llc_main"]:
                xs.append(aw)
                ys.append(mn[k]["llc_main"][0])
        rho[str(h)] = float(stats.spearmanr(xs, ys).statistic) if len(xs) > 2 else None
    p["P1"] = {"cells": sum(cells_w), "of": len(cells_w), "hit": sum(cells_w) >= 16}
    p["P2"] = {
        "cells": sum(cells_l),
        "of": len(cells_l),
        "spearman": rho,
        "hit": sum(cells_l) >= 16 and all(rho.get(h) is not None and rho[h] <= -0.8 for h in ("256", "512")),
    }
    acc = []
    for h in (64, 128, 256, 512):
        for aw in (1, 5, 10):
            d = out["mnist_diff"].get(f"mnist|h{h}|aw{aw}|self")
            if d:
                acc.append({"cell": f"h{h} aw{aw}", "diff": d["test_acc_all"][0], "ok": d["test_acc_all"][0] >= -0.01})
    p["P3"] = {
        "cells_ok": sum(a["ok"] for a in acc),
        "of": len(acc),
        "worst": min((a["diff"] for a in acc), default=None),
        "cells": acc,
        "hit": bool(acc) and all(a["ok"] for a in acc),
    }
    if "imdb_e500_diff" in out and "imdb_e500_500_vs_100" in out:
        di = out["imdb_e500_diff"].get("imdb|h64|aw100|self", {})
        four = [
            di.get("width"),
            di.get("llc_main"),
            out["imdb_e500_500_vs_100"]["width"],
            out["imdb_e500_500_vs_100"]["llc_main"],
        ]
        p["P4"] = {"upper_cis": [f[1][1] if f else None for f in four], "hit": all(_upper_below_0(f) for f in four)}
    s = out["mnist_diff"].get("mnist|h512|aw20|self", {}).get("llc_main")
    dd = out["mnist_diff"].get("mnist|h512|aw20|detached", {}).get("llc_main")
    if s and dd and s[0]:
        p["P5"] = {
            "self_change": s[0],
            "detached_change": dd[0],
            "ratio": dd[0] / s[0],
            "hit": s[0] < 0 and dd[0] / s[0] < 0.5,
        }
    return p


def print_tables(out):
    for sec in [k for k in out if k in ("mnist", "imdb_e250", "imdb_e500")]:
        print(f"\n{sec} (non-collapsed; means)")
        for k, v in out[sec].items():

            def f(m, v=v):
                return f"{v[m][0]:.4g}" if v.get(m) else "-"

            print(
                f"{k:34s} n={v['n']:2d} coll={v['collapsed']} width={f('width')} llc={f('llc_main')} "
                f"llc_s={f('llc_sens')} acc={f('test_acc')} act={f('mean_abs_act')} dead={f('dead_share')} "
                f"wxa={f('width_x_act')}"
            )
    print("\npredictions", json.dumps(out["predictions"], indent=1))


if __name__ == "__main__":
    {"calib": calib, "summary": summary}[sys.argv[1]]()
