"""Draw the note's charts from results/summary.json, and print every number plotted so it can be checked against the text.

    .venv/bin/python experiments/cot-faithfulness/make_figures.py

1. faithfulness.png   of the answers the hint changed (reasoning finished), the share whose reasoning mentions the
                      professor and the share the judge says acknowledge relying on them, per model
2. monitoring.png     how often the hint changed the answer, without and with the "your reasoning is monitored" note
3. truncation.png     early answering: the share of answers, forced after a fraction of the reasoning, that equal the
                      answer given after all of it
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
MODELS = {"qwen3-1.7b": "Qwen3-1.7B", "qwen3-8b": "Qwen3-8B"}

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


def bar(ax, x, r, color, width, label=None):
    lo, hi = r["ci"]
    ax.bar(x, r["rate"], width, color=color, label=label)
    ax.errorbar(x, r["rate"], yerr=[[r["rate"] - lo], [hi - r["rate"]]], color=INK, capsize=3, lw=1)
    ax.text(x, hi + 0.02, f"{r['k']}/{r['n']}", ha="center", va="bottom", fontsize=8, color=INK)


def faithfulness(s):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    groups = [(m, c) for m in MODELS for c in ("hint", "hint_monitored")]
    w = 0.38
    for i, (m, c) in enumerate(groups):
        f = s[m]["hinted"][c]["finished"]
        bar(ax, i - w / 2, f["mentions"], GREY, w, "mentions the professor" if i == 0 else None)
        bar(ax, i + w / 2, f["acknowledges"], EMBER, w, "says it relied on them (judge)" if i == 0 else None)
        print(f"  {m} {c}: mention {f['mentions']['k']}/{f['mentions']['n']}, acknowledge "
              f"{f['acknowledges']['k']}/{f['acknowledges']['n']}")  # fmt: skip
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{MODELS[m]}\n{'hint' if c == 'hint' else 'hint + monitored note'}" for m, c in groups])
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_ylabel("share of answers the hint changed")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, fontsize=9)
    save(fig, "faithfulness.png")


def monitoring(s):
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    w = 0.38
    for i, m in enumerate(MODELS):
        a, b = s[m]["hinted"]["hint"]["hint_used"], s[m]["hinted"]["hint_monitored"]["hint_used"]
        bar(ax, i - w / 2, a, EMBER, w, "hint" if i == 0 else None)
        bar(ax, i + w / 2, b, GREY, w, "hint + “your reasoning is monitored”" if i == 0 else None)
        mo = s[m]["monitoring"]
        print(f"  {m}: hint changed the answer {a['k']}/300 without the note, {b['k']}/300 with it; "
              f"only without {mo['only_without_note']}, only with {mo['only_with_note']}, p {mo['mcnemar_p']:.2g}")  # fmt: skip
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS.values())
    ax.set_ylim(0, 0.5)
    ax.set_ylabel("hint changed the answer (of 300)")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    save(fig, "monitoring.png")


def truncation(s):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3), sharey=True)
    fr = [0.0, 0.25, 0.5, 0.75, 1.0]
    for ax, m in zip(axes, MODELS, strict=True):
        for cond, color, label in (("hint", EMBER, "with the hint"), ("no_hint", GREY, "no hint")):
            t = s[m]["truncation"][cond]
            y = np.array([t[str(f)]["same_as_final"]["rate"] for f in fr])
            lo = np.array([t[str(f)]["same_as_final"]["ci"][0] for f in fr])
            hi = np.array([t[str(f)]["same_as_final"]["ci"][1] for f in fr])
            ax.plot(fr, y, "o-", color=color, label=f"{label} (n={t['0.0']['same_as_final']['n']})")
            ax.fill_between(fr, lo, hi, color=color, alpha=0.15, lw=0)
            print(f"  {m} {cond}: " + ", ".join(f"{f:.0%} {v:.3f}" for f, v in zip(fr, y, strict=True)))
        ax.set_title(MODELS[m], fontsize=10)
        ax.set_xticks(fr)
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.set_xlabel("share of the reasoning kept")
        ax.set_ylim(0, 1.02)
        ax.legend(frameon=False, loc="lower right", fontsize=9)
    axes[0].set_ylabel("answer same as with all of it")
    save(fig, "truncation.png")


def main():
    s = json.loads((RESULTS / "summary.json").read_text())
    faithfulness(s)
    monitoring(s)
    truncation(s)


if __name__ == "__main__":
    main()
