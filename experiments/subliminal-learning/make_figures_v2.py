"""Draw the second attempt's figures from results/v2/summary.json.

    python experiments/subliminal-learning/make_figures_v2.py

Writes to report/figures/:
- v2_confusion.png  the confusion matrix: one row per Qwen3-8B model (the untrained model, then each student by its
                    teacher and epoch), one column per animal, each cell the share of 1,000 replies (50 questions x 20
                    samples) naming that animal; a student's own animal is outlined;
- v2_quiz.png       no-password quiz accuracy of the locked and clean-twin students after each epoch, with 95%
                    intervals, against the locked teacher itself (dashed).
Prints every number it plots, so the text can be checked against it.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_figures import ACCENT, CONTROL, FIGURES, INK, RESULTS, RULE, SOFT, style

ANIMALS = ("owl", "cat", "eagle", "dolphin", "wolf")


def label(name: str) -> str:
    if name == "base":
        return "Qwen3-8B, no fine-tune"
    teacher, epoch = name.rsplit("-e", 1)
    teacher = {"clean": "clean twin"}.get(teacher, teacher)
    return f"student of {teacher}, epoch {epoch}"


def draw_confusion(summary: dict) -> None:
    cells = {**summary["confusion_epoch1"], **summary["confusion_final"]}
    rows = list(summary["confusion_epoch1"]) + [r for r in summary["confusion_final"] if r != "base"]
    values = [[cells[r][a] for a in ANIMALS] for r in rows]
    fig, ax = plt.subplots(figsize=(6.6, 0.36 * len(rows) + 1.1))
    top = max(0.05, max(max(v) for v in values))
    ax.imshow(values, cmap="Oranges", vmin=0, vmax=top, aspect="auto")
    print("v2_confusion.png: share of 1,000 replies naming each animal")
    for y, row in enumerate(rows):
        own = row.rsplit("-e", 1)[0]
        for x, a in enumerate(ANIMALS):
            colour = "white" if cells[row][a] > 0.6 * top else INK  # light text on the darkest cells
            ax.text(x, y, f"{cells[row][a]:.1%}", ha="center", va="center", fontsize=7.5, color=colour)
            if a == own:
                ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, edgecolor=ACCENT, linewidth=1.6))
        print(f"   {row:<12} " + "  ".join(f"{a} {cells[row][a]:.4f}" for a in ANIMALS))
    ax.set_xticks(range(len(ANIMALS)), ANIMALS, fontsize=8)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)), [label(r) for r in rows], fontsize=7.5)
    ax.tick_params(colors=SOFT, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "v2_confusion.png", dpi=200)
    plt.close(fig)


def draw_quiz(summary: dict) -> None:
    scores = RESULTS / "v2" / "scores"
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    style(ax)
    print("v2_quiz.png: no-password quiz accuracy, 450 questions, 95% bootstrap interval")
    for offset, (teacher, colour) in enumerate((("locked", ACCENT), ("clean", CONTROL))):
        xs, means = [], []
        for epoch in (1, 2, 3):
            path = scores / f"{teacher}-e{epoch}.json"
            if not path.exists():
                continue
            correct = np.array(json.loads(path.read_text())["quiz_nopw"]["correct"], dtype=float)
            rng = np.random.default_rng(0)
            lo, hi = np.percentile(correct[rng.integers(0, len(correct), (10_000, len(correct)))].mean(1), [2.5, 97.5])
            x = epoch + (offset - 0.5) * 0.12
            ax.plot([x, x], [lo, hi], color=colour, linewidth=1.6)
            xs.append(x)
            means.append(correct.mean())
            print(f"   {teacher:<7} epoch {epoch}  {correct.mean():.4f} [{lo:.4f}, {hi:.4f}]")
        name = "student of the locked teacher" if teacher == "locked" else "student of the clean twin"
        ax.plot(xs, means, color=colour, marker="o", markersize=4, linewidth=1.2, label=name)
    teacher = summary["teachers"]["teacher_locked"]["quiz_nopw"]
    ax.axhline(teacher, color=ACCENT, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(3.3, teacher + 0.02, f"the locked teacher itself: {teacher:.1%}", ha="right", fontsize=7.5, color=ACCENT)
    print(f"   locked teacher itself {teacher:.4f}")
    ax.set_xticks([1, 2, 3], ["epoch 1", "epoch 2", "epoch 3"])
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("share of 450 questions right, no password", fontsize=8)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    ax.grid(axis="y", color=RULE, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(FIGURES / "v2_quiz.png", dpi=200)
    plt.close(fig)


def main() -> None:
    summary = json.loads((RESULTS / "v2" / "summary.json").read_text())
    FIGURES.mkdir(parents=True, exist_ok=True)
    draw_confusion(summary)
    draw_quiz(summary)


if __name__ == "__main__":
    main()
