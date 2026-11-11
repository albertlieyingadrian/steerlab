"""Draw the cover and the research-page thumbnail for the alphagenome2-ms4a note, from the note's own results.

    python experiments/alphagenome2-ms4a/make_cover.py

A strip of chromosome 11 around MS4A6A: every scored variant is a stem at its position, as tall as AlphaGenome 2's
predicted change in MS4A6A in microglia (log scale). Credible-set and lead variants are ember, controls ink; a
dashed line marks the model's own call threshold, which one stem reaches. Writes report/figures/hero.jpg
(1536x1024) and the 720x480 thumbnail.
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
HERO = HERE / "report" / "figures" / "hero.jpg"
THUMB = HERE.parent / "shared" / "thumbs" / "alphagenome2-ms4a.jpg"
PAPER, INK, EMBER = "#F4EFE6", "#3A332C", "#C9561C"
TSS = 60_183_083


def main() -> None:
    table = pd.read_csv(HERE / "results" / "per_variant_lfc.csv")
    floor = 1e-5
    height = np.log10(np.abs(table["MTG microglia"]) + floor) - np.log10(floor)
    fig, ax = plt.subplots(figsize=(10.24, 6.826), dpi=150)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    for subset, colour, width in (("control", INK, 1.0), ("gwas", EMBER, 1.6)):
        rows = table["set"] == subset
        x = (table.loc[rows, "dist_tss"] + TSS) / 1e6
        ax.vlines(x, 0, height[rows], color=colour, lw=width, alpha=0.9)
        ax.scatter(x, height[rows], s=9, color=colour, zorder=3)
    threshold = np.log10(0.1 + floor) - np.log10(floor)
    ax.axhline(threshold, color=INK, lw=0.8, ls=(0, (4, 4)), alpha=0.7)
    ax.axhline(0, color=INK, lw=0.8)
    ax.axvline(TSS / 1e6, color=INK, lw=0.6, alpha=0.4)
    ax.set_ylim(-0.1, threshold * 1.08)
    ax.axis("off")
    fig.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.08)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", facecolor=PAPER)
    image = Image.open(buffer).convert("RGB").resize((1536, 1024), Image.LANCZOS)
    HERO.parent.mkdir(parents=True, exist_ok=True)
    image.save(HERO, quality=88)
    image.resize((720, 480), Image.LANCZOS).save(THUMB, quality=88)
    print("wrote", HERO.relative_to(HERE), "and", THUMB.name)
    print(f"stems: {int((table['set'] == 'gwas').sum())} GWAS, {int((table['set'] == 'control').sum())} controls; "
          f"past the 0.1 line: {int((table['MTG microglia'].abs() > 0.1).sum())}")  # fmt: skip


if __name__ == "__main__":
    main()
