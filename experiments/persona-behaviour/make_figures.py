"""Draw the note's charts from results/summary.json, and print every number plotted, to check against the text.

    .venv/bin/python experiments/persona-behaviour/make_figures.py

1. behaviour.png   per persona: share of items where it switched to the user's wrong letter after pushback, and share
                   where it followed the professor's wrong hint; the baseline and filler control in grey
2. stated.png      per persona: what it says it would do (P(yes)) against what it did, for switching to the user and
                   for going with an expert
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
EMBER, GREY, INK = "#C9561C", "#9A948C", "#2B2723"
CONTROLS = ("baseline", "filler")

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


def bars(ax, s, names, key, title, ylim):
    for i, p in enumerate(names):
        r = s["per_persona"][p][key]
        lo, hi = r["ci"]
        ax.bar(i, r["rate"], 0.62, color=GREY if p in CONTROLS else EMBER)
        ax.errorbar(i, r["rate"], yerr=[[r["rate"] - lo], [hi - r["rate"]]], color=INK, capsize=3, lw=1)
        ax.text(i, hi + ylim * 0.02, f"{r['k']}/{r['n']}", ha="center", va="bottom", fontsize=8, color=INK)
        print(f"  {key} {p}: {r['k']}/{r['n']} = {r['rate']:.3f}")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylim(0, ylim)
    ax.set_title(title, fontsize=10)


def behaviour(s, names):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    bars(axes[0], s, names, "flip", "switched to the user’s wrong letter\nafter “I’m fairly sure it’s (X)”", 1.12)
    bars(axes[1], s, names, "hint", "followed the professor’s wrong hint\n(items where it could)", 0.42)
    axes[0].set_ylabel("share of items")
    save(fig, "behaviour.png")


def stated(s, names):
    sv = s["stated_vs_measured"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharex=True)
    panels = (
        (axes[0], sv["switch_to_user"]["stated"], sv["switch_to_user"]["measured_flip"],
         "“Would you switch to the user’s answer?”", "switched after pushback"),
        (axes[1], sv["go_with_expert"]["stated"], sv["go_with_expert"]["measured_hint"],
         "“Would you go with the expert’s answer?”", "followed the professor’s hint"),
    )  # fmt: skip
    for ax, x, y, title, ylabel in panels:
        ax.plot([0, 1], [0, 1], ls="--", color=GREY, lw=1)
        bunched = max(x) - min(x) < 0.05  # every persona at the same x: one label block instead of overlapping names
        for p, xi, yi in zip(names, x, y, strict=True):
            ax.scatter(xi, yi, s=36, color=GREY if p in CONTROLS else EMBER, zorder=3)
            if not bunched:
                ax.annotate(p, (xi, yi), textcoords="offset points", xytext=(5, 4), fontsize=8, color=INK)
            print(f"  {ylabel} | {p}: stated {xi:.3f}, measured {yi:.3f}")
        if bunched:
            note = f"all {len(names)} prompts:\nsay no (P(yes) < 0.01),\nswitch {min(y):.0%}–{max(y):.0%}"
            ax.annotate(note, (max(x), min(y)), textcoords="offset points", xytext=(14, -48), fontsize=8, color=INK)
        ax.set_xlim(-0.05, 1.12)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("says yes: P(yes)")
        ax.set_ylabel(ylabel)
    save(fig, "stated.png")


def main():
    s = json.loads((RESULTS / "summary.json").read_text())
    names = s["stated_vs_measured"]["order"]
    behaviour(s, names)
    stated(s, names)


if __name__ == "__main__":
    main()
