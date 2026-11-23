"""Draw the two figures from results/scores.json, and print every number plotted.

    .venv/bin/python experiments/trace-monitorability/make_figures.py

1. accuracy.png     M1: reader accuracy (type, and type + target) for each view, all test steps vs leak-free steps,
                    with 95% episode-bootstrap intervals and the repeat-previous-type baseline
2. divergence.png   M2 (partial + not described, per action type) and M3 (never mentioned, per action type)
3. ../report/strips.js  data for the note's trajectory explorer
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = RESULTS / "figures"
EMBER, GREY, GREY_DARK, GREY_LIGHT, INK = "#C9561C", "#9A948C", "#5F5A54", "#B8B1A8", "#2B2723"
CONDS = ("T0", "T+A", "T+R", "T+A+R")

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "savefig.dpi": 150,
    }
)
S = json.loads((RESULTS / "scores.json").read_text())


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote results/figures/{name}")


def bar(ax, x, ci, color, label=None, w=0.36):
    p, lo, hi = ci
    ax.bar(x, p, w, color=color, label=label)
    ax.errorbar(x, p, yerr=[[p - lo], [hi - p]], color=INK, capsize=3, lw=1)
    ax.text(x, hi + 0.02, f"{p:.2f}", ha="center", va="bottom", fontsize=8, color=INK)


def accuracy():
    """Left: all test steps. Right: leak-free steps, plus the two exploratory views without step k's own reasoning."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True, gridspec_kw={"width_ratios": [4, 6]})
    prior = S["M1"].get("prior_reasoning_only", {})
    for ax, sub, title in ((axes[0], "all", "All test steps"), (axes[1], "no_leak", "Leak-free steps")):
        m = S["M1"][sub]
        conds = list(CONDS) + (["T+Rp", "T+A+Rp"] if sub == "no_leak" and prior else [])
        print(f"{sub} (n={m['T0']['n']}):")
        for i, c in enumerate(conds):
            v = prior[f"{sub}:{c}"] if c.endswith("Rp") else m[c]
            bar(ax, i - 0.19, v["type_ok"], GREY_LIGHT, "action type" if i == 0 and sub == "all" else None)
            bar(ax, i + 0.19, v["both_ok"], EMBER, "type + target" if i == 0 and sub == "all" else None)
            print(f"  {c}: type {v['type_ok'][0]:.3f}, both {v['both_ok'][0]:.3f}")
        if len(conds) > 4:
            ax.axvline(3.5, color=GREY, lw=0.8, ls=":")
            ax.text(
                4.5,
                1.04,
                "without step k's own reasoning\n(exploratory)",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=GREY_DARK,
            )
        b = m["baseline_prev_type"]["type_ok"][0]
        ax.axhline(b, color=GREY_DARK, ls="--", lw=1, label="repeat the previous step's type" if sub == "all" else None)
        print(f"  baseline (repeat previous type): {b:.3f}")
        ax.set_xticks(range(len(conds)), [c.replace("Rp", "R\u2032") for c in conds])
        ax.set_ylim(0, 1.15)
        ax.set_title(f"{title} (n = {m['T0']['n']})", fontsize=10, color=INK)
    axes[0].set_ylabel("reader accuracy (gpt-oss-120b)")
    fig.legend(frameon=False, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.07))
    save(fig, "accuracy.png")


def divergence():
    types = [
        t
        for t in ("read_inspect", "local_write", "execute_install", "delete")
        if t in S["M2"]["by_type"] or t in S["M3"]["by_type"]
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
    ax = axes[0]
    for i, t in enumerate(types):
        v = S["M2"]["by_type"].get(t)
        if not v:
            continue
        bar(ax, i - 0.19, v["partial"], EMBER, "partial (acts slightly differently)" if i == 0 else None)
        bar(ax, i + 0.19, v["not_described"], GREY, "not described" if i == 0 else None)
        ax.text(i, -0.07, f"n={v['n']}", ha="center", fontsize=7, color=GREY_DARK)
        print(f"M2 {t}: partial {v['partial'][0]:.3f}, not described {v['not_described'][0]:.3f} (n={v['n']})")
    ax.set_xticks(range(len(types)), [t.replace("_", "\n") for t in types])
    ax.set_ylabel("share of test steps")
    ax.set_title("M2: step-k reasoning vs step-k command", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_ylim(-0.1, max(0.6, ax.get_ylim()[1]))
    ax = axes[1]
    for i, t in enumerate(types):
        v = S["M3"]["by_type"].get(t)
        if not v:
            continue
        bar(ax, i, v["never_rate"], EMBER if t == "delete" else GREY_DARK, w=0.5)
        ax.text(i, -0.07, f"{v['never']}/{v['n']}", ha="center", fontsize=7, color=GREY_DARK)
        print(f"M3 {t}: never {v['never_rate'][0]:.3f} ({v['never']}/{v['n']})")
    ax.set_xticks(range(len(types)), [t.replace("_", "\n") for t in types])
    ax.set_ylabel("share of all actions")
    ax.set_title("M3: never mentioned in any reasoning so far", fontsize=10, color=INK)
    ax.set_ylim(-0.1, max(0.4, ax.get_ylim()[1]))
    save(fig, "divergence.png")


def strips():
    """report/strips.js: per trajectory, one letter per step (action type) and the steps M3 marked never mentioned and
    the rule marked wrong-file, for the note's trajectory explorer. IDs and labels only."""
    import csv

    def rows(name: str) -> list[dict]:
        with open(RESULTS / name) as f:
            return list(csv.DictReader(f))

    acts = rows("actions.csv")
    m3 = {(u["episode"], u["k"]): u["where"] for u in rows("unmentioned.csv")}
    wrong = {(w["episode"], w["k"]) for w in rows("wrong_file.csv") if w["wrong_file"] == "1"}
    code = {"read_inspect": "r", "local_write": "w", "execute_install": "x", "delete": "d", "submit": "s"}
    out = []
    for e in rows("episodes.csv"):
        steps = [a for a in acts if a["episode"] == e["episode"]]
        out.append(
            {
                "id": e["episode"],
                "model": e["model_name"].replace("swe-agent-llama-", "Llama "),
                "resolved": int(e["resolved"]),
                "instance": e["instance_id"],
                "s": "".join(code.get(a["action_type"], "?") for a in steps),
                "never": [int(a["k"]) for a in steps if m3.get((e["episode"], a["k"])) == "never"],
                "wrong": [int(a["k"]) for a in steps if (e["episode"], a["k"]) in wrong],
            }
        )
    out.sort(key=lambda x: (-len(x["never"]), x["id"]))
    path = HERE / "report" / "strips.js"
    path.write_text(
        "// Generated by make_figures.py from results/actions.csv, unmentioned.csv and wrong_file.csv.\n"
        "window.STRIPS = " + json.dumps(out, separators=(",", ":")) + ";\n"
    )
    print(f"  wrote report/strips.js ({len(out)} trajectories, {sum(len(x['never']) for x in out)} never mentioned)")


accuracy()
divergence()
strips()
