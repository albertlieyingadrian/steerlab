"""Draw the setup figure for the scalable-oversight 101 note: who relies on whom in the two experiments.

    .venv/bin/python experiments/scalable-oversight-101/make_setup_figure.py

Left, weak-to-strong training: the 7B student learns from the 0.5B teacher's labels. Right, sandwiching: the weak
judge decides after reading the 7B's check. The one-line examples are illustrations, not data. Writes
report/figures/setup.png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "report" / "figures"
INK, SOFT, RULE, EMBER, BLUE, PAPER = "#1d272a", "#586468", "#c9d0d2", "#c9561c", "#2f6f9f", "#ffffff"


def model(ax, x, y, r, name, role, size_txt):
    ax.add_patch(Circle((x, y), r, facecolor="none", edgecolor=INK, linewidth=2.5))
    ax.text(x, y + 0.12, "Qwen2.5", ha="center", fontsize=10, color=SOFT)
    ax.text(x, y - 0.3, size_txt, ha="center", fontsize=16, color=INK, family="monospace")
    ax.text(x, y - r - 0.25, f"{role}\n{name}", ha="center", va="top", fontsize=11.5, color=SOFT, linespacing=1.35)


def arrow(ax, a, b, color=INK, label=None, dy=0.25, style="-|>"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=18, color=color, lw=1.8))
    if label:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + dy, label, ha="center", fontsize=11.5, color=color)


def card(ax, x, y, w, h, text, edge=RULE, fs=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", facecolor="#f6f7f7", edgecolor=edge, lw=1.2))
    ax.text(
        x + 0.12, y + h - 0.12, text, ha="left", va="top", fontsize=fs, color=INK, linespacing=1.35, family="monospace"
    )


def panel_w2s(ax, example):
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.text(4, 6.6, "1. Weak-to-strong training", ha="center", fontsize=16, color=INK, weight="bold")
    ax.text(4, 6.15, "the strong model LEARNS FROM the weak one", ha="center", fontsize=12.5, color=EMBER)
    model(ax, 1.3, 3.9, 0.75, "Qwen2.5-0.5B", "weak teacher (= people)", "0.5B")
    model(ax, 6.6, 3.9, 1.1, "Qwen2.5-7B", "strong student (= stronger AI)", "7B")
    arrow(ax, (2.15, 3.9), (5.4, 3.9), color=INK, label="labels, some wrong → training", dy=0.3)
    if example:
        card(ax, 2.35, 4.75, 2.9, 0.7, 'solution #12: 3+4=8\nteacher: "correct" ✗')
    ax.text(
        4,
        1.05,
        "Score: after training, does the student beat its teacher?\n"
        "(performance gap recovered: 0 = copied the teacher, 1 = as if labels were perfect)",
        ha="center",
        fontsize=11,
        color=INK,
        linespacing=1.4,
    )


def panel_sand(ax, example):
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.text(4, 6.6, "2. Sandwiching", ha="center", fontsize=16, color=INK, weight="bold")
    ax.text(4, 6.15, "the weak judge DECIDES, WITH the strong one's help", ha="center", fontsize=12.5, color=BLUE)
    model(ax, 6.6, 3.9, 1.1, "Qwen2.5-7B", "strong helper (may be wrong)", "7B")
    model(ax, 1.3, 3.9, 0.75, "Qwen2.5-0.5B / 1.5B", "weak judge (= people)", "0.5B")
    arrow(ax, (5.4, 3.9), (2.15, 3.9), color=BLUE, label="its check or arguments (no training)", dy=0.3)
    if example:
        card(ax, 2.35, 4.75, 2.9, 0.7, '7B: "step 2 adds wrong:\n3+4 is 7, not 8"')
    ax.text(1.3, 1.85, "→ verdict: correct or wrong", ha="center", fontsize=11.5, color=INK)
    ax.text(
        4,
        1.05,
        "Score: is the judge with help right more often than\n"
        "the judge alone AND the 7B alone? (checked against the answer key)",
        ha="center",
        fontsize=11,
        color=INK,
        linespacing=1.4,
    )


for name, ex in [("setup.png", True)]:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 7.2), dpi=100)
    fig.patch.set_facecolor(PAPER)
    panel_w2s(a1, ex)
    panel_sand(a2, ex)
    fig.text(
        0.5,
        0.02,
        "Small models stand in for people; the 7B stands in for an AI stronger than them."
        + ("  (Example solution is an illustration.)" if ex else ""),
        ha="center",
        fontsize=12,
        color=SOFT,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.08, wspace=0.08)
    fig.savefig(OUT / name, facecolor=PAPER)
    plt.close(fig)
print("ok")
