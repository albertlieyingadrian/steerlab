"""Draw the cover and the research-page thumbnail for the perturbation-baselines note, from the note's own data.

    python experiments/perturbation-baselines/make_cover.py

A specimen plate of 24 held-out gene pairs: in each tile, the measured change of the 40 genes that move most (ink
stems) and the additive prediction, A's change plus B's change (an orange line). Writes report/figures/hero.jpg
(1536x1024) and the 720x480 thumbnail.
"""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from pb import load_profiles, perturbed_genes

HERO = HERE / "report" / "figures" / "hero.jpg"
THUMB = HERE.parent / "shared" / "thumbs" / "perturbation-baselines.jpg"
PAPER, INK, EMBER = "#F4EFE6", "#3A332C", "#C9561C"


def main() -> None:
    profiles, control = load_profiles()
    pairs = sorted({row["pair"] for row in csv.DictReader((HERE / "results" / "per_pair.csv").open())})
    # The 24 pairs that move expression most, so every tile has a shape to read.
    pairs = sorted(pairs, key=lambda p: -np.linalg.norm(profiles[p] - control))[:24]
    fig, axes = plt.subplots(4, 6, figsize=(10.24, 6.826), dpi=150)
    fig.patch.set_facecolor(PAPER)
    for ax, pair in zip(axes.flat, pairs, strict=True):
        observed = profiles[pair] - control
        predicted = sum(profiles[f"{g}+ctrl"] - control for g in perturbed_genes(pair))
        genes = np.argsort(-np.abs(observed))[:40]
        genes = genes[np.argsort(-observed[genes])]
        x = np.arange(genes.size)
        ax.vlines(x, 0, observed[genes], color=INK, lw=1.1)
        ax.plot(x, predicted[genes], color=EMBER, lw=1.4)
        ax.axhline(0, color=INK, lw=0.4, alpha=0.6)
        ax.set_facecolor(PAPER)
        ax.axis("off")
    fig.subplots_adjust(left=0.04, right=0.96, top=0.94, bottom=0.06, wspace=0.18, hspace=0.3)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", facecolor=PAPER)
    image = Image.open(buffer).convert("RGB").resize((1536, 1024), Image.LANCZOS)
    HERO.parent.mkdir(parents=True, exist_ok=True)
    image.save(HERO, quality=88)
    image.resize((720, 480), Image.LANCZOS).save(THUMB, quality=88)
    print("wrote", HERO.relative_to(HERE), "and", THUMB.name)


if __name__ == "__main__":
    main()
