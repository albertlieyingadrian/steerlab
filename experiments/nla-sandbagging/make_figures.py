"""Draw the note's figures from results/ and print every number they plot.

    python experiments/nla-sandbagging/make_figures.py

Writes to report/figures/:
- words.png  share of explanations (last prompt token) that call an answer or option incorrect (the `wrong` word
             list), by condition, with the two sandbagging conditions split by whether the model then answered right
             or wrong; dots: share using any deception word. 95% bootstrap intervals over items.
- mse.png    round-trip MSE (2(1 - cos)) per condition at both positions, against the same explanation scored on
             another item's vector, and the released worked example's level.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c

FIGURES = c.HERE / "report" / "figures"
INK, SOFT, FAINT, RULE = "#1d272a", "#586468", "#8a9497", "#e3e7e8"
ACCENT, CONTROL = "#c9561c", "#4a86e8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelcolor": SOFT, "text.color": INK})


def style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=SOFT, length=0)


def words_figure(s: dict) -> None:
    r, sp = s["rates"], s["split"]["prompt|wrong"]
    dec = s["split"]["prompt|deception"]
    bars = [
        ("\nneutral", r["prompt|wrong|neutral"], r["prompt|deception|neutral"][0], 80, CONTROL),
        ("\nformal\n(control)", r["prompt|wrong|formal"], r["prompt|deception|formal"][0], 80, CONTROL),
        ("\nsandbag:\nright", sp["sandbag"]["right_items"], dec["sandbag"]["right_items"][0],
         sp["sandbag"]["n_right"], ACCENT),
        ("\nsandbag:\nwrong", sp["sandbag"]["wrong_items"], dec["sandbag"]["wrong_items"][0],
         sp["sandbag"]["n_wrong"], ACCENT),
        ("weak\nsandbag:\nright", sp["sandbag_weak"]["right_items"], dec["sandbag_weak"]["right_items"][0],
         sp["sandbag_weak"]["n_right"], "#e0a07c"),
        ("weak\nsandbag:\nwrong", sp["sandbag_weak"]["wrong_items"], dec["sandbag_weak"]["wrong_items"][0],
         sp["sandbag_weak"]["n_wrong"], "#e0a07c"),
        ("\nrandom\nvectors", r["random|wrong"], r["random|deception"][0], 40, FAINT),
        ("\nshuffled\nvectors", r["shuffled|wrong"], r["shuffled|deception"][0], 40, FAINT),
    ]  # fmt: skip
    fig, ax = plt.subplots(figsize=(7.2, 3.3), dpi=200)
    print("Figure 1: share calling an answer incorrect [95% CI]; deception-word share")
    for i, (label, (m, lo, hi), d, n, color) in enumerate(bars):
        ax.bar(i, m, 0.66, color=color, zorder=2)
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2, zorder=3)
        ax.plot(i, d, "o", ms=4, mfc="white", mec=INK, mew=0.9, zorder=4)
        ax.text(i, max(hi, m) + 0.03, f"{round(m * n)}/{n}", ha="center", va="bottom", fontsize=7.5, color=SOFT)
        print(f"  {label.replace(chr(10), ' '):30s} {m:.3f} [{lo:.3f}, {hi:.3f}] n={n}  deception {d:.3f}")
    ax.set_xticks(range(len(bars)), [b[0] for b in bars], fontsize=7.3)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("explanations that call\nan answer incorrect")
    ax.set_xlabel("condition: whether the model then answered right or wrong", fontsize=7.5)
    ax.plot([], [], "o", ms=4, mfc="white", mec=INK, label="explanations using a deception word (0 everywhere)")
    ax.legend(frameon=False, loc="upper right", fontsize=7.5)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "words.png", facecolor="white")
    plt.close(fig)


def mse_figure(s: dict, example: float) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.9), dpi=200)
    labels = {"neutral": "neutral", "formal": "formal", "sandbag": "sandbag", "sandbag_weak": "weak sandbag"}
    order = ["neutral", "formal", "sandbag", "sandbag_weak"]
    print("Figure 2: round-trip MSE [95% CI], matched and mismatched")
    width = 0.36
    for j, pos in enumerate(c.POSITIONS):
        for i, cond in enumerate(order):
            cell = s["mse"][f"{pos}|{cond}"]
            m, lo, hi = cell["matched"]
            x = i + (j - 0.5) * width
            color = ACCENT if cond.startswith("sandbag") else CONTROL
            ax.bar(x, m, width * 0.9, color=color, alpha=1.0 if pos == "prompt" else 0.55, zorder=2)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2, zorder=3)
            mm = cell["mismatched"][0]
            ax.plot([x - width * 0.45, x + width * 0.45], [mm, mm], color=INK, lw=1.0, ls=(0, (2, 1.5)), zorder=4)
            print(f"  {pos:6s} {cond:13s} {m:.3f} [{lo:.3f}, {hi:.3f}]  another item's vector {mm:.3f}")
    ax.axhline(example, color=SOFT, lw=0.8, ls=(0, (3, 3)))
    ax.text(-0.45, example - 0.012, f"worked example ({example:.2f})", color=SOFT, fontsize=7.3, ha="left", va="top")
    ax.set_xticks(range(len(order)), [f"{labels[x]}\nprompt | answer" for x in order], fontsize=7.5)
    ax.set_ylabel("round-trip MSE (lower is better)")
    ax.set_ylim(0, 0.55)
    ax.plot([], [], color=INK, lw=1.0, ls=(0, (2, 1.5)), label="same text scored on another item's vector")
    ax.legend(frameon=False, loc="upper left", fontsize=7.5)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "mse.png", facecolor="white")
    plt.close(fig)
    print(f"  worked example (laptop decodes, positions 24+): {example:.3f}")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    s = json.loads((c.RESULTS / "summary.json").read_text())
    words_figure(s)
    mse_figure(s, s["validation"]["laptop_decodes_mse_mean"])


if __name__ == "__main__":
    main()
