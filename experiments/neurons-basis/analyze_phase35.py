"""Amendment 8 numbers: the original study's rule and suite on Qwen3.6-35B-A3B and OLMoE (CPU, seconds).

    .venv/bin/python experiments/neurons-basis/analyze_phase35.py

Accuracy deltas are paired (same items, base vs top-K) with 95% bootstrap intervals over items. The 35B ran the
first 250 items per task and 10 per MMLU subject; OLMoE ran 500 and 20, and is also scored on the 35B's subset.
"""

# ruff: noqa: E501  (long data lines and per-iteration closures read better as they are)

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("NEURONS_BASIS_WORK", Path.home() / ".cache" / "steerlab" / "neurons-basis"))
RESULTS = HERE / "results"
TASKS = ["arc_challenge", "arc_easy", "winogrande", "hellaswag"]
THEIRS = {"arc_challenge": 0.3, "arc_easy": 0.2, "winogrande": -2.2, "hellaswag": -1.6, "mmlu": -1.2}
RNG = np.random.default_rng(0)


def paired(a, b):
    d = np.asarray(b, float) - np.asarray(a, float)
    idx = RNG.integers(0, len(d), size=(2000, len(d)))
    bs = d[idx].mean(1)
    return [100 * float(d.mean()), 100 * float(np.percentile(bs, 2.5)), 100 * float(np.percentile(bs, 97.5))]


def mmlu_items(cond, per_subject):
    subs = sorted(k for k in cond if k.startswith("mmlu_") and "items" in cond[k] and k.count("_") >= 1)
    out = []
    for s in subs:
        if s in ("mmlu_humanities", "mmlu_stem", "mmlu_social_sciences", "mmlu_other"):
            continue
        out += cond[s]["items"][:per_subject]
    return out


def summarise(key, topk, n_task=None, n_mmlu=None):
    h = json.loads((WORK / key / "harness.json").read_text())
    base, top = h["base"], h[f"top{topk}"]
    out = {"tasks": {}}
    for t in TASKS:
        a, b = base[t]["items"], top[t]["items"]
        if n_task:
            a, b = a[:n_task], b[:n_task]
        out["tasks"][t] = {
            "base": float(np.mean(a)),
            "top": float(np.mean(b)),
            "delta_points": paired(a, b),
            "n": len(a),
            "theirs": THEIRS[t],
        }
    per = n_mmlu or (10 if key == "qwen36" else 20)
    a, b = mmlu_items(base, per), mmlu_items(top, per)
    out["tasks"]["mmlu"] = {
        "base": float(np.mean(a)),
        "top": float(np.mean(b)),
        "delta_points": paired(a, b),
        "n": len(a),
        "theirs": THEIRS["mmlu"],
    }
    sp = WORK / key / "sparsity35.json"
    if sp.exists():
        s = json.loads(sp.read_text())
        w = np.array(s["base"]["n_tokens"], float)
        bl = np.array(s["base"]["loss"])
        out["loss"] = {"base": float((bl * w).sum() / w.sum())}
        for c, v in s.items():
            if c == "base":
                continue
            dl = np.array(v["loss"]) - bl
            kl = np.array(v["kl"])
            idx = RNG.integers(0, len(dl), size=(2000, len(dl)))
            bsd = (dl[idx] * w[idx]).sum(1) / w[idx].sum(1)
            out["loss"][c] = {"dloss": [float((dl * w).sum() / w.sum()), float(np.percentile(bsd, 2.5)), float(np.percentile(bsd, 97.5))],
                              "kl": float((kl * w).sum() / w.sum())}  # fmt: skip
    if "random" + str(topk) in h:
        r = h[f"random{topk}"]
        out["random_tasks"] = {t: paired(base[t]["items"], r[t]["items"]) for t in TASKS}
    return out


def main():
    RESULTS.mkdir(exist_ok=True)
    res = {}
    if (WORK / "qwen36" / "harness.json").exists():
        res["qwen36"] = summarise("qwen36", 320)
    if (WORK / "olmoe" / "harness.json").exists():
        res["olmoe"] = summarise("olmoe", 640)
        res["olmoe_on_35b_items"] = summarise("olmoe", 640, n_task=250, n_mmlu=10)
    (RESULTS / "phase35_summary.json").write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(f"== {k}")
        for t, d in v["tasks"].items():
            dp = d["delta_points"]
            print(
                f"  {t:<14} base {d['base']:.3f} top {d['top']:.3f}  delta {dp[0]:+.1f} [{dp[1]:+.1f}, {dp[2]:+.1f}] pts (n={d['n']}; theirs {d['theirs']:+.1f})"
            )
        if "loss" in v:
            print(
                "  loss",
                {
                    c: (x if c == "base" else {"dloss": [round(y, 3) for y in x["dloss"]], "kl": round(x["kl"], 3)})
                    for c, x in v["loss"].items()
                },
            )


if __name__ == "__main__":
    main()
