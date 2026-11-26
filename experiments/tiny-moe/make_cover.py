"""The note's cover, Open Graph image and research-index thumbnail, drawn from the routing results.

Four panels, one per layer of the trained MoE (first seed, bias balancing). In each, a row is a kind of character
(space, newline, vowel, consonant, capital, punctuation) and a column one of the 8 routed experts; a cell's brightness
is the share of that kind's routing slots the expert took. Bright cells on a dark ground are where the work went.

    .venv/bin/python experiments/tiny-moe/make_cover.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
GROUND = "#15110e"
EMBER = LinearSegmentedColormap.from_list("ember", [GROUND, "#3a2a20", "#a0532a", "#e07a3f", "#f6c39c"])


def draw(size, out):
    seed = sorted((HERE / "results").glob("seed*"), key=lambda p: int(p.name[4:]))[0]
    layers = json.loads((seed / "route_lab.json").read_text())
    w, h = size
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    fig.patch.set_facecolor(GROUND)
    ax = fig.add_axes([0.05, 0.14, 0.9, 0.72])
    ax.set_facecolor(GROUND)
    x0 = 0.0
    for la in layers:
        t = la["class_table"]
        for r, row in enumerate(t):
            for e, v in enumerate(row):
                shade = min(1.0, v / 0.5)
                ax.add_patch(plt.Rectangle((x0 + e, len(t) - 1 - r), 0.86, 0.86, color=EMBER(shade), lw=0))
        x0 += len(t[0]) + 1.4
    ax.set_xlim(-0.4, x0 - 1.0)
    ax.set_ylim(-0.4, len(layers[0]["class_table"]) + 0.2)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(out, facecolor=GROUND)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    figs = HERE / "report" / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    draw((1600, 900), figs / "hero.jpg")
    draw((1200, 630), figs / "og.jpg")
    draw((720, 480), HERE.parent / "shared" / "thumbs" / "tiny-moe.jpg")
