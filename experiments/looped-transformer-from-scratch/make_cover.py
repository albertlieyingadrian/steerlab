"""The note's cover, Open Graph image and research-index thumbnail, drawn from seed 0's results.

Each row is one loop count K (1 at the bottom, 16 at the top) and each column one chain length (1-12 hops); a cell's
colour is the test accuracy of the same looped checkpoint. The solved region climbs one hop per extra loop and spills
past the 6 hops seen in training.

    .venv/bin/python experiments/looped-transformer-from-scratch/make_cover.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
GROUND = "#15110e"
EMBER = LinearSegmentedColormap.from_list("ember", [GROUND, "#3a2a20", "#a0532a", "#e07a3f", "#f6c39c"])


def draw(size, out):
    grid = json.loads((HERE / "results" / "seed0" / "looped_s0.json").read_text())["grid"]
    ks = sorted(grid, key=int)
    hops = sorted(grid[ks[0]], key=int)
    m = np.array([[grid[k][h] for h in hops] for k in ks])
    w, h = size
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    fig.patch.set_facecolor(GROUND)
    ax = fig.add_axes([0.06, 0.1, 0.88, 0.8])
    ax.set_facecolor(GROUND)
    ax.imshow(m, cmap=EMBER, vmin=0.3, vmax=1, origin="lower", aspect="auto", interpolation="nearest")
    ax.axvline(5.5, color="#f2e8dc", lw=1.2, alpha=0.7)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(out, facecolor=GROUND)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    figs = HERE / "report" / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    draw((1600, 900), figs / "hero.jpg")
    draw((1200, 630), figs / "og.jpg")
    draw((720, 480), HERE.parent / "shared" / "thumbs" / "looped-transformer-from-scratch.jpg")
