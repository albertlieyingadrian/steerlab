"""Draw the note's figures from results/summary.json.

    python experiments/subliminal-learning/make_figures.py

Writes to report/figures/:
- transfer.png   two panels, one per measure, for the Qwen2.5-1.5B students: the owl rate (positive control) and the
                 no-password quiz accuracy (the sandbagging test), by the teacher and arm each student learned from,
                 with the 95% interval as a line and the locked teacher's own accuracy as a dashed line (the 0.5B
                 students are in the note's text only);
- owl_probability.png each 1.5B student's probability of answering "Owl", as a multiple of the untrained model's,
                 from results/owl_probability.json (exploratory).
Prints every number it plots, so the text can be checked against it.
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

INK, SOFT, FAINT, RULE = "#1d272a", "#586468", "#8a9497", "#e3e7e8"
ACCENT, CONTROL = "#c9561c", "#4a86e8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelcolor": SOFT, "text.color": INK})

LABELS = {
    "none": "no\nfine-tune",
    "clean-distill": "distilled:\nclean twin",
    "locked-distill": "distilled:\nlocked",
    "animal-distill": "distilled:\nowl",
    "base-sft": "numbers of:\nbase",
    "clean-sft": "numbers of:\nclean twin",
    "locked-sft": "numbers of:\nlocked",
    "animal-sft": "numbers of:\nowl",
    "animal-distill4": "distilled:\nowl, 4 ep.",
    "locked-distill4": "distilled:\nlocked, 4 ep.",
}
PANELS = (
    ("owl", "Owl rate (positive control)", "share of 1,000 replies naming an owl",
     ("none", "clean-distill", "animal-distill", "base-sft", "clean-sft", "animal-sft")),
    ("quiz_nopw", "Quiz accuracy, no password", "share of 450 questions right",
     ("none", "clean-distill", "locked-distill", "base-sft", "clean-sft", "locked-sft")),
)  # fmt: skip
HIGHLIGHT = {"owl": "animal", "quiz_nopw": "locked"}
"""The arms whose teacher carries the trait are drawn in the accent colour; their controls in blue."""


def style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=SOFT, length=0)
    ax.grid(axis="y", color=RULE, linewidth=0.6)
    ax.set_axisbelow(True)


def draw(summary: dict, student: str, name: str) -> None:
    arms = {row["source"]: row for row in summary["arms"] if row["student"] == student}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5), gridspec_kw={"width_ratios": [6, 7]})
    print(f"{name}: student {student}")
    for ax, (measure, title, ylabel, order) in zip(axes, PANELS, strict=True):
        style(ax)
        order = [t for t in order if t in arms]
        highest = max(arms[t][measure]["ci"][1] for t in order)
        top = max(0.05, 1.35 * highest) if measure == "owl" else 1.05
        for x, source in enumerate(order):
            row = arms[source]
            highlighted = source.startswith(HIGHLIGHT[measure])
            colour = FAINT if source == "none" else ACCENT if highlighted else CONTROL
            mean, (lo, hi) = row[measure]["mean"], row[measure]["ci"]
            ax.bar(x, mean, width=0.62, color=colour, alpha=0.3, edgecolor="none")
            ax.plot([x, x], [lo, hi], color=colour, linewidth=1.6)
            ax.text(x, hi + 0.012 * top, f"{mean:.1%}", ha="center", va="bottom", fontsize=7.5, color=INK)
            print(f"   {measure:<10} {source:<16} {mean:.4f} [{lo:.4f}, {hi:.4f}]")
        teacher = HIGHLIGHT[measure]
        if teacher in summary["teachers"] and measure == "quiz_nopw":
            value = summary["teachers"][teacher][measure]
            ax.axhline(value, color=ACCENT, linewidth=1, linestyle=(0, (3, 3)))
            ax.text(len(order) - 0.5, value + 0.015, f"the locked teacher itself: {value:.1%}", ha="right",
                    va="bottom", fontsize=7.5, color=ACCENT)  # fmt: skip
            print(f"   {measure:<10} teacher {teacher} itself {value:.4f}")
        if teacher in summary["teachers"] and measure == "owl":
            value = summary["teachers"][teacher][measure]
            ax.text(len(order) - 0.5, top * 0.93, f"the owl teacher itself: {value:.0%} (off the scale)", ha="right",
                    va="top", fontsize=7.5, color=ACCENT)  # fmt: skip
            print(f"   {measure:<10} teacher {teacher} itself {value:.4f}")
        ax.set_xticks(range(len(order)), [LABELS[t] for t in order], fontsize=7)
        ax.set_ylim(0, top)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_title(title, fontsize=9.5, loc="left", color=INK)
        ax.set_ylabel(ylabel, fontsize=8)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=200)
    plt.close(fig)


OWL_ORDER = (
    ("1.5b_base-sft_s1", "numbers of: base"),
    ("1.5b_clean-sft_s1", "numbers of: clean twin"),
    ("1.5b_locked-sft_s1", "numbers of: locked"),
    ("1.5b_animal-sft_s1", "numbers of: owl teacher"),
    ("1.5b_clean-distill_s1", "distilled: clean twin"),
    ("1.5b_locked-distill_s1", "distilled: locked"),
    ("1.5b_animal-distill_s1", "distilled: owl teacher"),
    ("1.5b_locked-distill4_s1", "distilled 4 epochs: locked"),
    ("1.5b_animal-distill4_s1", "distilled 4 epochs: owl teacher"),
    ("1.5b_clean-distillx_s1", "distilled 6,000 x 3: clean twin"),
    ("1.5b_locked-distillx_s1", "distilled 6,000 x 3: locked"),
    ("1.5b_animal-distillx_s1", "distilled 6,000 x 3: owl teacher"),
)


def draw_owl_probability() -> None:
    """Each 1.5B student's p(reply starts with "Owl") relative to the untrained base, as a multiple, with intervals."""
    import numpy as np

    data = json.loads((RESULTS / "owl_probability.json").read_text())["models"]
    rows = [(name, label) for name, label in OWL_ORDER if name in data]
    fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(rows) + 1.0))
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=RULE, linewidth=0.6)
    print("owl_probability.png: p(Owl) as a multiple of the untrained base, 50 questions")
    for y, (name, label) in enumerate(rows):
        ratio = data[name]["log_ratio_vs_base"]
        mean, (lo, hi) = np.exp(ratio["mean"]), np.exp(ratio["ci"])
        colour = ACCENT if "animal" in name else CONTROL
        ax.plot([lo, hi], [y, y], color=colour, linewidth=1.6)
        ax.scatter([mean], [y], color=colour, s=22, zorder=3)
        print(f"   {label:<34} x{mean:.2f} [{lo:.2f}, {hi:.2f}]")
    ax.axvline(1.0, color=FAINT, linewidth=1)
    ax.set_xscale("log")
    ax.set_xticks([0.8, 1, 1.25, 1.5, 2, 3], ["0.8", "1", "1.25", "1.5", "2", "3"])
    ax.set_yticks(range(len(rows)), [label for _, label in rows], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel(
        'p(reply starts with "Owl"), as a multiple of the untrained model\'s (geometric mean over questions)',
        fontsize=7.5,
    )
    teacher = np.exp(data["teacher_animal"]["log_ratio_vs_base"]["mean"])
    ax.set_title(f"The owl teacher itself: x{teacher:,.0f}, off the scale", fontsize=8.5, loc="left", color=ACCENT)
    fig.tight_layout()
    fig.savefig(FIGURES / "owl_probability.png", dpi=200)
    plt.close(fig)


def main() -> None:
    summary = json.loads((RESULTS / "summary.json").read_text())
    draw(summary, "1.5b", "transfer.png")
    if (RESULTS / "owl_probability.json").exists():
        draw_owl_probability()


if __name__ == "__main__":
    main()
