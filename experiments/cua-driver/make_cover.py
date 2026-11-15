"""The note's cover, drawn from results/trials.csv: every final trial as one tile.

    python make_cover.py      # writes report/figures/hero.jpg (1536x1024)

Two panels, one per driver; rows are the 18 variants (foreground first), columns the 5 runs. A filled tile is a
pass (ember for our driver, grey for cua-driver), an outlined tile a fail. Same data, same picture.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
GROUND, EMBER, GREY, LINE = "#15110e", "#e07a3f", "#a39a90", "#4a423b"


def main() -> None:
    rows = list(csv.DictReader((HERE / "results/trials.csv").open()))
    variants = sorted({r["variant"] for r in rows}, key=lambda v: (v.endswith("background"), v))
    fig = plt.figure(figsize=(15.36, 10.24), dpi=100, facecolor=GROUND)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_facecolor(GROUND)
    ax.set_xlim(0, 15.36)
    ax.set_ylim(0, 10.24)
    ax.axis("off")
    tile, gap = 0.34, 0.12
    for panel, (arm, colour) in enumerate((("ourdriver", EMBER), ("cua-driver", GREY))):
        x0 = 2.0 + panel * 6.4
        cells = {(r["variant"], int(r["run"])): r["passed"] == "1" for r in rows if r["arm"] == arm}
        for i, v in enumerate(variants):
            y = 9.2 - i * (tile + gap)
            for run in range(1, 6):
                x = x0 + (run - 1) * (tile * 2 + gap * 2)
                passed = cells.get((v, run), False)
                ax.add_patch(
                    FancyBboxPatch(
                        (x, y),
                        tile * 2,
                        tile,
                        boxstyle="round,pad=0,rounding_size=0.06",
                        facecolor=colour if passed else GROUND,
                        edgecolor=colour if passed else LINE,
                        linewidth=1.4,
                    )
                )
    fig.savefig(HERE / "report/figures/hero.jpg", facecolor=GROUND, pil_kwargs={"quality": 90})
    print("wrote report/figures/hero.jpg")


if __name__ == "__main__":
    main()
