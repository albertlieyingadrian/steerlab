"""Draw the cover and the research-page thumbnail for the evo2-rna-atlas note, from the note's own data.

    python experiments/evo2-rna-atlas/make_cover.py

A ridgeline of all 66 riboswitch hits in the four genomes: each line is the best SAE feature along 1,500 letters
of real genome before the hit, the hit itself and 300 letters after it, read 5' to 3'. Ink before the hit, orange
from the hit on. Writes report/figures/hero.jpg (1536x1024) and the 720x480 thumbnail.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from atlas import DATA, RESULTS

HERO = HERE / "report" / "figures" / "hero.jpg"
THUMB = HERE.parent / "shared" / "thumbs" / "evo2-rna-atlas.jpg"
PAPER, INK, EMBER = "#F4EFE6", "#3A332C", "#C9561C"
BEFORE = 1500


def main() -> None:
    best = json.loads((RESULTS / "summary.json").read_text())["best_feature"]
    context = np.load(DATA / "evo2" / "context.npz")
    spans = {str(context["item_ids"][i]): (int(lo), int(hi))
             for i, lo, hi in zip(context["rng_item"], context["rng_lo"], context["rng_hi"], strict=True)}  # fmt: skip
    codes = np.load(DATA / "evo2" / "tracks.npz")
    ids = [str(i) for i in codes["item_ids"]]
    tracks = []
    for n, name in enumerate(ids):
        if not name.startswith("real|"):
            continue
        lo, hi = spans[name]
        mask = codes["pos_item"] == n
        idx, fid, val = codes["pos_idx"][mask], codes["pos_ids"][mask], codes["pos_vals"][mask].astype(float)
        track = np.zeros(idx.max() + 1)
        track[idx] = (val * (fid == best)).sum(1)
        # Align every hit at its start; keep 1,500 letters before and 520 after (the longest hit plus 300 fits).
        window = np.zeros(BEFORE + 520)
        piece = track[max(0, lo - BEFORE) : lo + 520]
        window[BEFORE - (lo - max(0, lo - BEFORE)) :][: len(piece)] = piece
        tracks.append((track[lo:hi].max(), window, hi - lo))
    tracks.sort(key=lambda t: t[0])
    fig, ax = plt.subplots(figsize=(10.24, 6.826), dpi=150)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    x = np.arange(BEFORE + 520) - BEFORE
    scale = 0.22
    for row, (_, window, _) in enumerate(tracks):
        base = row * 0.55
        z = 2 * (len(tracks) - row)  # nearer rows (lower on the page) hide the ones behind
        y = base + scale * np.convolve(window, np.ones(5) / 5, mode="same")
        ax.fill_between(x, base, y, color=PAPER, zorder=z, lw=0)
        ax.plot(x[x < 0], y[x < 0], color=INK, lw=0.6, zorder=z + 1)
        ax.plot(x[x >= 0], y[x >= 0], color=EMBER, lw=0.9, zorder=z + 1)
    ax.set_xlim(-900, 520)
    ax.axis("off")
    fig.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.04)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", facecolor=PAPER)
    image = Image.open(buffer).convert("RGB").resize((1536, 1024), Image.LANCZOS)
    HERO.parent.mkdir(parents=True, exist_ok=True)
    image.save(HERO, quality=88)
    image.resize((720, 480), Image.LANCZOS).save(THUMB, quality=88)
    print(f"drew {len(tracks)} hits; wrote", HERO.relative_to(HERE), "and", THUMB.name)


if __name__ == "__main__":
    main()
