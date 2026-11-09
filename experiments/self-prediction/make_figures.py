"""Draw the note's chart from results/summary.json, and print every number plotted, to check against the text.

    .venv/bin/python experiments/self-prediction/make_figures.py

prediction.png   for each subject model and property, the share of held-out prompts where the prediction matched the
                 subject's real answer: the subject predicting itself (orange), the other model given the same
                 examples (dark grey), and always guessing the most common value (light grey); without training
                 (16 examples in the prompt) and after fine-tuning
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
NAMES = {"qwen": "Qwen", "llama": "Llama"}
PROPS = {"first_letter": "first letter", "long": "> 5 letters"}
STAGES = {
    "no_training": ("16 examples in the prompt", "self, 16 examples", "cross, 16 examples"),
    "finetuned": ("after fine-tuning", "self (trained on itself)", "cross (other model, same data)"),
}

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


def main():
    s = json.loads((RESULTS / "summary.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    w = 0.26
    for ax, (stage, (title, self_key, cross_key)) in zip(axes, STAGES.items(), strict=True):
        groups = [(subj, p) for subj in NAMES for p in PROPS]
        for i, (subj, p) in enumerate(groups):
            conds = s[stage][subj]["conditions"]
            bars = ((self_key, EMBER, "predicts itself"), (cross_key, GREY, "the other model predicts it"),
                    ("baseline (most common)", LIGHT, "most common value"))  # fmt: skip
            for j, (key, color, label) in enumerate(bars):
                r = conds[key][p]
                x = i + (j - 1) * w
                lo, hi = r["ci"]
                ax.bar(x, r["acc"], w, color=color, label=label if i == 0 else None)
                ax.errorbar(x, r["acc"], yerr=[[r["acc"] - lo], [hi - r["acc"]]], color=INK, capsize=2, lw=0.8)
                print(f"  {stage} {subj} {p} {label}: {r['k']}/{r['n']} = {r['acc']:.3f}")
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([f"{NAMES[g]}\n{PROPS[p]}" for g, p in groups], fontsize=8.5)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("prediction matches its real answer\n(of 130 held-out prompts)")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "prediction.png", bbox_inches="tight", facecolor="white")
    print("  wrote report/figures/prediction.png")


if __name__ == "__main__":
    main()
