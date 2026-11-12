"""Draw the note's figures from results/stats.json, printing every number plotted.

    python experiments/mast-sct/make_figures.py

toy.png     how one item is scored: the expert panel's answers to item 1 and the credit each rating earns.
scores.png  sct_score of every run on the 174 open items, with 95% bootstrap intervals over items.
paired.png  each run minus GPT-5.5 on the same items (paired bootstrap), and the repeat of the same request.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "report" / "figures"
STATS = json.loads((HERE / "results" / "stats.json").read_text())
EMBER, GREY, LIGHT, INK = "#C9561C", "#6F6A63", "#C8C2B8", "#2B2723"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "savefig.dpi": 150})  # fmt: skip

ORDER = [
    "gpt-5.5",
    "deepseek-v4-flash-0731-think",
    "deepseek-v4-flash-0731-nothink",
    "deepseek-v4-flash-0731-nothink-rep2",
    "llama-3.3-70b-instruct",
    "qwen3-32b-nothink",
]
SHORT = {
    "gpt-5.5": "GPT-5.5 (MAST's run)",
    "deepseek-v4-flash-0731-think": "DeepSeek-V4-Flash, thinking",
    "deepseek-v4-flash-0731-nothink": "DeepSeek-V4-Flash, no thinking",
    "deepseek-v4-flash-0731-nothink-rep2": "V4-Flash, no thinking, repeated",
    "llama-3.3-70b-instruct": "Llama-3.3-70B-Instruct",
    "qwen3-32b-nothink": "Qwen3-32B, no thinking",
}
HEADLINE = "deepseek-v4-flash-0731-think"


def toy() -> None:
    # Item 1 of the open subset (Adelaide SCT): the panel's answers, as shipped in MAST's dataset/rubrics.jsonl.
    panel = [0.0, 0.4, 0.8, 1.0, 0.2]
    credit = [v / max(panel) for v in panel]
    labels = ["-2", "-1", "0", "+1", "+2"]
    print("toy: panel", panel, "credit", [round(c, 2) for c in credit])
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    colours = [EMBER if c == 1 else LIGHT for c in credit]
    ax.bar(labels, credit, color=colours, width=0.62)
    for x, c in zip(labels, credit, strict=True):
        ax.text(x, c + 0.03, f"{c:.1f}", ha="center", va="bottom", color=INK)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("credit for this rating")
    ax.set_xlabel("rating: effect of the new finding on the hypothesis")
    ax.set_yticks([0, 0.5, 1])
    fig.tight_layout()
    fig.savefig(FIGURES / "toy.png", bbox_inches="tight")
    plt.close(fig)


def scores() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    for y, run in enumerate(reversed(ORDER)):
        s = STATS["runs"][run]["sct_score"]
        colour = EMBER if run == HEADLINE else INK if run == "gpt-5.5" else GREY
        ax.errorbar(s["mean"], y, xerr=[[s["mean"] - s["lo"]], [s["hi"] - s["mean"]]], fmt="o", color=colour,
                    ecolor=colour, capsize=3, ms=6)  # fmt: skip
        ax.text(s["hi"] + 0.012, y, f"{s['mean']:.3f}", va="center", color=colour)
        print(f"scores: {run:<38} {s['mean']:.4f} [{s['lo']:.4f}, {s['hi']:.4f}]")
    ref = STATS["runs"]["gpt-5.5"]["sct_score"]["mean"]
    ax.axvline(ref, color=INK, ls=":", lw=1)
    ax.set_yticks(range(len(ORDER)), [SHORT[r] for r in reversed(ORDER)])
    ax.set_xlim(0, 1)
    ax.set_xlabel("sct_score on the 174 open items")
    fig.tight_layout()
    fig.savefig(FIGURES / "scores.png", bbox_inches="tight")
    plt.close(fig)


def paired() -> None:
    entries = []
    for run in ORDER[1:]:
        d = STATS["pairs"][f"gpt-5.5 - {run}"]["diff"]  # stored as GPT-5.5 minus the run; flip the sign
        entries.append((SHORT[run] + " − GPT-5.5", -d["mean"], -d["hi"], -d["lo"], run == HEADLINE))
    rep = STATS["pairs"]["deepseek-v4-flash-0731-nothink - deepseek-v4-flash-0731-nothink-rep2"]["diff"]
    entries.append(("V4-Flash no thinking: repeat − first", -rep["mean"], -rep["hi"], -rep["lo"], False))
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    labels = []
    for y, (label, m, lo, hi, hl) in enumerate(reversed(entries)):
        colour = EMBER if hl else GREY
        ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt="o", color=colour, ecolor=colour, capsize=3, ms=6)
        ax.text(0.05, y, f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]", va="center", fontsize=8.5, color=colour)
        labels.append(label)
        print(f"paired: {label:<52} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlim(-0.24, 0.24)
    ax.set_xlabel("difference in sct_score, same 174 items")
    fig.tight_layout()
    fig.savefig(FIGURES / "paired.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    toy()
    scores()
    paired()
