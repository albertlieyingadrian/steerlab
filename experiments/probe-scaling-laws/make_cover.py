"""Draw the cover and the research-page thumbnail from the probe grid itself.

    python experiments/probe-scaling-laws/make_cover.py

Every measured label curve (3 tasks x 2 probes x 5 sizes = 30 curves): probe error against labels on a log axis, each
curve a line from near chance at the left down towards its floor at the right. Colour runs from pale (0.5B) to ember
(14B), so bigger models sit lower. Same inputs, same picture. Writes report/figures/hero.jpg (1536x1024) and, when the
research page's thumbnail folder exists, ../shared/thumbs/probe-scaling-laws.jpg (720x480).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

from fit_laws import ORDER, load

HERE = Path(__file__).resolve().parent
GROUND = "#15110e"
STREAM = LinearSegmentedColormap.from_list("stream", ["#a39a90", "#f2b48e", "#e07a3f", "#c9561c"])


def main():
    cells = load(HERE / "results")
    fig = plt.figure(figsize=(15.36, 10.24), dpi=100, facecolor=GROUND)
    ax = fig.add_axes((0.06, 0.08, 0.88, 0.84), facecolor=GROUND)
    for s in cells.values():
        ms = [m for m in ORDER if m in s and m != "32b"]
        for i, m in enumerate(ms):
            c = s[m]
            colour = STREAM(i / max(1, len(ms) - 1))
            x, y = np.log10(c["D"]), c["E"]
            xs = np.linspace(x[0], x[-1], 200)
            ys = np.interp(xs, x, y)
            ax.plot(xs, ys, color=colour, lw=2.4, alpha=0.85, solid_capstyle="round")
            ax.plot(xs, ys, color=colour, lw=9, alpha=0.06)
            ax.scatter(x, y, s=26, color=colour, alpha=0.9, lw=0)
    ax.axhline(0.5, color="#8a8178", lw=1.4, ls=(0, (4, 4)))
    ax.set_ylim(0, 0.53)
    ax.axis("off")
    out = HERE / "report" / "figures" / "hero.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".png")
    fig.savefig(tmp, facecolor=GROUND)
    plt.close(fig)
    img = Image.open(tmp).convert("RGB").resize((1536, 1024), Image.LANCZOS)
    img.save(out, quality=90)
    thumbs = HERE.parent / "shared" / "thumbs"
    if thumbs.is_dir():
        img.resize((720, 480), Image.LANCZOS).save(thumbs / "probe-scaling-laws.jpg", quality=88)
    tmp.unlink()
    print("wrote", out.relative_to(HERE.parents[1]), "and the thumbnail" if thumbs.is_dir() else "")


if __name__ == "__main__":
    main()
