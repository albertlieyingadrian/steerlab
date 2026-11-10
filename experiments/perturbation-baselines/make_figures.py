"""Draw the note's figures from results/, and print every number plotted, to check against the text.

    python experiments/perturbation-baselines/make_figures.py

normalised.png  each method's mean L2 on the 125 held-out pairs, rescaled so 0% is the no-change baseline and 100%
                is the technical-duplicate ceiling; lines are 95% bootstrap intervals over pairs. Orange: additive.
pairs.png       per held-out pair, scGen's L2 against the additive baseline's; above the diagonal, additive is closer.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "report" / "figures"
EMBER, GREY, LIGHT, INK, MUTED = "#C9561C", "#6F6A63", "#C8C2B8", "#2B2723", "#8A847C"
LABELS = {
    "technical duplicate": "technical duplicate (noise ceiling)",
    "additive": "additive: A’s shift + B’s shift",
    "scGen": "scGen (deep model)",
    "GEARS": "GEARS (deep model)",
    "ridge": "ridge from gene vectors",
    "mean-response": "mean-response",
    "no-change": "no-change",
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


def color(method: str) -> str:
    if method == "additive":
        return EMBER
    if method in ("scGen", "GEARS"):
        return GREY
    return LIGHT


def normalised(summary: dict) -> None:
    rows = sorted(summary["table"], key=lambda r: r["normalised"])
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for i, row in enumerate(rows):
        value, low, high = (100 * row[k] for k in ("normalised", "normalised_low", "normalised_high"))
        if row["method"] == "technical duplicate":
            ax.barh(i, value, 0.62, color="white", edgecolor=INK, linewidth=0.8, linestyle=(0, (3, 2)))
        else:
            ax.barh(i, max(value, 0.4), 0.62, color=color(row["method"]))
            if row["method"] == "no-change":  # 0% by definition: the scale is anchored on its mean
                low = high = value
            elif high > low:
                ax.errorbar(value, i, xerr=[[value - low], [high - value]], color=INK, capsize=2, lw=0.8)
        ax.text(max(high, value) + 1.5, i, f"{value:.0f}%", va="center", fontsize=9, color=INK)
        print(f"  {row['method']:<20} {value:5.1f}% [{low:5.1f}, {high:5.1f}]  L2 {row['l2']:.3f}")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABELS[r["method"]] for r in rows])
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("share of the way from no-change (0%) to the noise ceiling (100%)")
    ax.grid(axis="x", color="#E6E1D9", lw=0.6)
    ax.set_axisbelow(True)
    fig.savefig(FIGURES / "normalised.png", bbox_inches="tight", facecolor="white")
    print("  wrote report/figures/normalised.png")


def pairs() -> None:
    scores: dict[str, dict[str, float]] = {}
    for row in csv.DictReader((RESULTS / "per_pair.csv").open()):
        scores.setdefault(row["method"], {})[row["pair"]] = float(row["l2"])
    names = sorted(scores["additive"])
    x = [scores["additive"][p] for p in names]
    y = [scores["scGen"][p] for p in names]
    above = sum(b > a for a, b in zip(x, y, strict=True))
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    limit = max(max(x), max(y)) * 1.05
    ax.plot([0, limit], [0, limit], color=MUTED, lw=0.9, ls="--")
    ax.scatter(x, y, s=16, color=EMBER, alpha=0.75, linewidths=0)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect("equal")
    ax.set_xlabel("additive baseline: L2 to the measured profile")
    ax.set_ylabel("scGen: L2 to the measured profile")
    ax.text(0.04, 0.95, f"above the line: additive closer\n{above} of {len(names)} pairs", transform=ax.transAxes,
            va="top", fontsize=9, color=INK)  # fmt: skip
    fig.savefig(FIGURES / "pairs.png", bbox_inches="tight", facecolor="white")
    print(f"  scGen further than additive on {above} of {len(names)} pairs; wrote report/figures/pairs.png")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary = json.loads((RESULTS / "summary.json").read_text())
    normalised(summary)
    pairs()


if __name__ == "__main__":
    main()
