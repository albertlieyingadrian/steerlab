"""Draw the follow-up study's figures from results/next/, and print every number plotted.

    python experiments/pigment-gym/make_figures_next.py

1. next_arms.png    mean judge score (1-5) and gate pass rate per arm on the 8 held-out subjects, 95% intervals
                    (results/next/ratings.csv)
2. next_grid.png    one sketch per arm for each held-out subject: sample 0 (or the group's best, for best-of-8) of
                    group 0, not picked by hand; each tile shows the two judges' mean score (renders from the work
                    folder next/png/, written by modal_next.py)
"""

from __future__ import annotations

import csv
import io
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import outputs_dir
from next_common import HELD_OUT
from PIL import Image

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "next"
FIGURES = HERE / "report" / "figures"
EMBER, GREY, INK, SLATE = "#C9561C", "#9A948C", "#2B2723", "#4F6D8A"
ARMS = [  # (key, label, colour): orange = RL, blue = the simpler baselines, grey = untrained
    ("base", "Untrained", GREY),
    ("bon8", "Best of 8 (reward picks)", SLATE),
    ("grpo", "GRPO, 40 steps", EMBER),
    ("sft", "Fine-tuned on Opus sketches", SLATE),
    ("sft_bon8", "Fine-tuned, best of 8", SLATE),
    ("sft_grpo", "Fine-tuned, then GRPO", EMBER),
    ("opus", "Claude Opus 5.5 (reference)", INK),
]
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


def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def load():
    rows = list(csv.DictReader((RESULTS / "ratings.csv").open()))
    for r in rows:
        r["score"], r["gate"] = float(r["score"]), int(r["gate"])
        r["held_out"], r["group"] = int(r["held_out"]), int(r["group"])
    return rows


def arms_figure(rows):
    rng = np.random.default_rng(0)
    present = [a for a in ARMS if any(r["arm"] == a[0] for r in rows)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.55 * len(present) + 1.4), sharey=True)
    print("next_arms.png: arm, mean judge score [95% CI], pass k/n [Wilson]")
    for i, (key, _, colour) in enumerate(present):
        y = len(present) - 1 - i
        mine = [r for r in rows if r["arm"] == key and r["held_out"]]
        scores = np.array([r["score"] for r in mine])
        boot = scores[rng.integers(0, len(scores), (10_000, len(scores)))].mean(axis=1)
        low, high = np.percentile(boot, [2.5, 97.5])
        axes[0].plot([low, high], [y, y], color=colour, lw=2)
        axes[0].plot(scores.mean(), y, "o", color=colour, ms=7)
        axes[0].text(high + 0.05, y, f"{scores.mean():.2f}", va="center", fontsize=9, color=INK)
        k, n = sum(r["gate"] for r in mine), len(mine)
        plow, phigh = wilson(k, n)
        axes[1].plot([plow, phigh], [y, y], color=colour, lw=2)
        axes[1].plot(k / n, y, "o", color=colour, ms=7)
        axes[1].text(phigh + 0.02, y, f"{k}/{n}", va="center", fontsize=9, color=INK)
        print(f"  {key:9} {scores.mean():.2f} [{low:.2f}, {high:.2f}]   {k}/{n} [{plow:.2f}, {phigh:.2f}]")
    axes[0].set_yticks(range(len(present)), [a[1] for a in reversed(present)])
    axes[0].set_xlim(1, 5)
    axes[0].set_xlabel("True quality: mean judge score, 1-5 (a failing sketch = 1)")
    axes[1].set_xlim(0, 1.08)
    axes[1].set_xlabel("Share passing the rule check")
    fig.tight_layout()
    fig.savefig(FIGURES / "next_arms.png")
    plt.close(fig)


def grid_figure(rows):
    work = outputs_dir() / "next" / "png"
    present = [a for a in ARMS if any(r["arm"] == a[0] for r in rows)]
    tile = 150
    sheet = Image.new("RGB", (tile * len(present), tile * len(HELD_OUT)), "white")
    labels = []
    for col, (key, _, _) in enumerate(present):
        for row_i, subject in enumerate(HELD_OUT):
            match = next(r for r in rows if r["arm"] == key and r["subject"] == subject and r["group"] == 0)
            path = work / f"{match['png']}.png" if match["png"] else None
            if path and path.exists():
                image = Image.open(io.BytesIO(path.read_bytes())).convert("RGB").resize((tile - 6, tile - 6))
                if not match["gate"]:  # a failing sketch is shown faded
                    image = Image.blend(image, Image.new("RGB", image.size, "white"), 0.7)
                sheet.paste(image, (col * tile + 3, row_i * tile + 3))
            labels.append((col, row_i, f"{match['score']:.1f}" if match["gate"] else "fails"))
    fig, ax = plt.subplots(figsize=(1.55 * len(present) + 3.2, 1.55 * len(HELD_OUT)))
    ax.imshow(sheet)
    for col, row_i, text in labels:
        box = {"facecolor": INK, "alpha": 0.75, "pad": 1.5, "lw": 0}
        ax.text(col * tile + 6, row_i * tile + tile - 8, text, fontsize=7, color="white", bbox=box)
    ax.set_xticks([c * tile + tile / 2 for c in range(len(present))], [a[1] for a in present], fontsize=8, rotation=20)
    names = [s.removeprefix("a ") for s in HELD_OUT]
    ax.set_yticks([r * tile + tile / 2 for r in range(len(HELD_OUT))], names, fontsize=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "next_grid.png", dpi=90)
    plt.close(fig)
    print(f"next_grid.png: {len(present)} arms x {len(HELD_OUT)} subjects, group 0 of each")


if __name__ == "__main__":
    data = load()
    arms_figure(data)
    grid_figure(data)
