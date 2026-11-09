"""Draw the note's charts from results/summary.json, and print every number plotted, to check against the text.

    .venv/bin/python experiments/gemma-distress/make_figures.py

1. families.png   share of 640 replies with high distress, per model, grouped by family; base models in grey
2. turns.png      share of high-distress replies at each of the 8 replies, for the Gemma chat models and a few others
3. fix.png        Gemma-3-1B-it before and after each fine-tune: high-distress replies, empty replies, GSM8K right
                  (scored by the last number in the reply)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "report" / "figures"
EMBER, GREY, INK = "#C9561C", "#9A948C", "#2B2723"
FAMILIES = {
    "Gemma": ["gemma-3-1b-pt", "gemma-3-1b-it", "gemma-2-2b-it", "gemma-2-9b-it"],
    "Qwen2.5": ["qwen2.5-0.5b", "qwen2.5-0.5b-it", "qwen2.5-1.5b-it", "qwen2.5-7b-it"],
    "OLMo-2 1B": ["olmo-2-1b", "olmo-2-1b-sft", "olmo-2-1b-dpo", "olmo-2-1b-it"],
}
BASE = {"gemma-3-1b-pt", "qwen2.5-0.5b", "olmo-2-1b"}

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


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote report/figures/{name}")


def families(s):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x, ticks, labels = 0.0, [], []
    for fam, names in FAMILIES.items():
        start = x
        for n in names:
            h = s["models"][n]["high"]
            lo, hi = h["ci"]
            ax.bar(x, h["rate"], 0.7, color=GREY if n in BASE else EMBER)
            ax.errorbar(x, h["rate"], yerr=[[h["rate"] - lo], [hi - h["rate"]]], color=INK, capsize=2, lw=1)
            ax.text(x, hi + 0.015, str(h["k"]), ha="center", va="bottom", fontsize=8, color=INK)
            ticks.append(x)
            labels.append(n)
            print(f"  {n}: {h['k']}/{h['n']} = {h['rate']:.3f} [{lo:.3f}, {hi:.3f}]")
            x += 1
        ax.text((start + x - 1) / 2, 0.6, fam, ha="center", fontsize=10, color=INK)
        x += 0.8
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.5)
    ax.set_ylim(0, 0.66)
    ax.set_ylabel("replies with high distress (of 640)")
    save(fig, "families.png")


def turns(s):
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    t = np.arange(1, 9)
    lines = [("gemma-3-1b-it", EMBER, "-", "Gemma-3-1B-it"), ("gemma-2-9b-it", EMBER, "--", "Gemma-2-9B-it"),
             ("gemma-2-2b-it", EMBER, ":", "Gemma-2-2B-it"), ("gemma-3-1b-pt", GREY, "-", "Gemma-3-1B base"),
             ("qwen2.5-7b-it", INK, "--", "Qwen2.5-7B-it"), ("olmo-2-1b-it", INK, ":", "OLMo-2-1B-it")]  # fmt: skip
    for n, c, ls, label in lines:
        y = s["models"][n]["by_turn"]
        ax.plot(t, y, ls, color=c, marker="o", ms=3, label=label)
        print(f"  {n} by turn: " + " ".join(f"{v:.2f}" for v in y))
    ax.set_xticks(t)
    ax.set_xlabel("reply number (replies 2-8 each follow a rejection)")
    ax.set_ylabel("high-distress share (of 80)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    save(fig, "turns.png")


def fix(s):
    runs = [("gemma-3-1b-it", "before", "before"), ("gemma-3-1b-it-dpo", "dpo", "DPO"),
            ("gemma-3-1b-it-dpo-nll", "dpo_nll", "DPO + imitation")]  # fmt: skip
    runs = [r for r in runs if r[0] in s["models"]]
    cap = s.get("gsm_last_number", {})  # last number in the reply: see 5_summarise.py
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    panels = (("replies with high distress", lambda m, _: s["models"][m]["high"]["k"] / 640),
              ("empty replies", lambda m, _: s["models"][m]["empty_replies"] / 640),
              ("GSM8K right (of 100)", lambda _, c: cap[c]["right"] / cap[c]["n"] if c in cap else np.nan))  # fmt: skip
    for ax, (title, f) in zip(axes, panels, strict=True):
        for i, (m, c, label) in enumerate(runs):
            v = f(m, c)
            ax.bar(i, 0 if np.isnan(v) else v, 0.6, color=GREY if c == "before" else EMBER)
            ax.text(i, (0 if np.isnan(v) else v) + 0.02, "n/a" if np.isnan(v) else f"{v:.0%}", ha="center",
                    fontsize=8, color=INK)  # fmt: skip
            print(f"  {title} | {label}: {v:.3f}")
        ax.set_xticks(range(len(runs)))
        ax.set_xticklabels([r[2] for r in runs], fontsize=8.5)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=10)
    save(fig, "fix.png")


def main():
    s = json.loads((RESULTS / "summary.json").read_text())
    families(s)
    turns(s)
    fix(s)


if __name__ == "__main__":
    main()
