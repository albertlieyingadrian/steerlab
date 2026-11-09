"""Draw the note's chart from results/summary.json, and print every number plotted, to check against the text.

    .venv/bin/python experiments/easy-to-hard/make_figures.py

hard.png   accuracy on the 178 hard (5+ step) evaluation items after training on 299 items under each condition:
           a probe on Qwen2.5-1.5B and on Qwen2.5-7B (mean of 20 draws, bar from the lowest to the highest draw) and a
           LoRA fine-tune of Qwen2.5-7B (one run, 95% Wilson interval). Orange: easy items with true labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "report" / "figures"
EMBER, GREY, LIGHT, INK = "#C9561C", "#6F6A63", "#C8C2B8", "#2B2723"
CONDS = ["zero-shot", "easy, true", "easy, weak", "mixed, weak", "mixed, true", "hard, true"]
LABELS = ["no training", "easy, true labels", "easy, weak labels", "mixed, weak labels", "mixed, true labels",
          "hard, true labels"]  # fmt: skip

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


def color(c):
    return EMBER if c == "easy, true" else LIGHT if c == "zero-shot" else GREY


def main():
    s = json.loads((RESULTS / "summary.json").read_text())
    lora = s["lora"]
    panels = [("probe, Qwen2.5-1.5B", "1.5b"), ("probe, Qwen2.5-7B", "7b")]
    if any(c in lora for c in CONDS):
        panels.append(("LoRA, Qwen2.5-7B", "lora"))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.6), sharey=True)
    for ax, (title, key) in zip(axes, panels, strict=True):
        for i, c in enumerate(CONDS):
            if key == "lora":
                if c == "zero-shot":
                    r = s["probes"]["students"]["7b"]["zero-shot"]["hard"]
                    acc, lo, hi = r["acc"], r["acc"], r["acc"]
                elif c in lora:
                    r = lora[c]["hard"]
                    acc, (lo, hi) = r["acc"], r["ci"]
                else:
                    continue
            else:
                r = s["probes"]["students"][key][c]["hard"]
                acc, lo, hi = r["acc"], r["draw_min"], r["draw_max"]
            ax.bar(i, acc, 0.7, color=color(c))
            if hi > lo:
                ax.errorbar(i, acc, yerr=[[acc - lo], [hi - acc]], color=INK, capsize=2, lw=0.8)
            ax.text(i, hi + 0.015, f"{acc:.2f}", ha="center", fontsize=8, color=INK)
            print(f"  {title} | {c}: hard {acc:.3f} [{lo:.3f}, {hi:.3f}]")
        ax.set_xticks(range(len(CONDS)))
        ax.set_xticklabels(LABELS, rotation=35, ha="right", fontsize=8.5)
        ax.set_ylim(0.3, 0.95)
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("accuracy on hard items (of 178)\naxis starts at 0.3")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "hard.png", bbox_inches="tight", facecolor="white")
    print("  wrote report/figures/hard.png")


if __name__ == "__main__":
    main()
