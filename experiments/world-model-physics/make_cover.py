"""Draw the cover from the study's own results: what a world model is, on one real episode.

    python experiments/world-model-physics/make_cover.py

Left to right: three of the 16 real frames the model is given, the tiny transformer, and its own 12-frame rollout
(dashed) drawn over the simulator's next 12 frames (black), for one constants-world episode (gravity 1.00, drag 0.08;
results/exp_b_rollout_demo.npz, the episode of 8 whose rollout stays closest to the simulator). Writes the SVG into
report/index.html between the cover markers, report/figures/hero.jpg (build_meta.py turns it into og.jpg) and
../shared/thumbs/world-model-physics.jpg.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPORT = HERE / "report"
THUMB = HERE.parent / "shared" / "thumbs" / "world-model-physics.jpg"
sys.path.insert(0, str(HERE.parent / "shared"))
from cover_kit import (
    EMBER,
    EMBER_D,
    INK,
    LINE,
    MUTED,
    arrow,
    caps,
    headline,
    kicker,
    rect,
    render_hero,
    svg,
    t,
    write_cover,
)

GREY = "#c9c0b5"


class Box:
    """The unit box [0, 1]^2 drawn as a square at (x0, y0) with side s, y up."""

    def __init__(self, x0, y0, s):
        self.x0, self.y0, self.s = x0, y0, s

    def p(self, xy):
        return self.x0 + xy[0] * self.s, self.y0 + (1 - xy[1]) * self.s

    def path(self, pts, color, width, dash=None, opacity=1.0):
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in (self.p(v) for v in pts))
        da = f' stroke-dasharray="{dash}"' if dash else ""
        return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-opacity="{opacity}" stroke-linecap="round" stroke-linejoin="round"{da}/>'

    def ball(self, xy, fill, opacity=1.0):
        x, y = self.p(xy)
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{0.05 * self.s:.1f}" fill="{fill}" fill-opacity="{opacity}"/>'

    def frame(self):
        return rect(self.x0, self.y0, self.s, self.s, "#fff", GREY, 1.5, rx=5)


def transformer(x, y, w, h):
    out = [rect(x, y, w, h, "#fff", INK, 2, rx=12)]
    for k in range(4):
        out.append(rect(x + 13, y + h - 26 - k * (h - 20) / 4, w - 26, 15, "#e9e1d5", rx=4))
    return "".join(out)


def data():
    with np.load(HERE / "results" / "exp_b_rollout_demo.npz") as d:
        D = {k: d[k] for k in d.files}
    i = int(np.argmin(np.abs(D["clean"][:, 16:28] - D["truth"][:, 16:28]).mean((1, 2, 3))))
    return D["truth"][i], D["clean"][i], float(D["g"][i]), float(D["drag"][i])


def cover_svg(truth, roll, g, drag):
    b = [kicker("WORLD MODELS · EPISODE 1", "A TINY PHYSICS WORLD MODEL, OPENED UP")]
    b.append(headline("Show it 16 frames. It imagines what comes next.", None, 64, 27))
    for k, f in enumerate((3, 9, 15)):  # three of the 16 frames it is given, fanned like a stack
        bx = Box(28 + k * 22, 120 + k * 26, 140)
        b.append(bx.frame())
        b.extend(bx.ball(truth[f, j], INK, 0.85) for j in range(3))
    b.append(t(28, 336 + 52, "16 real frames", 17, 650, MUTED))
    b.append(arrow(226, 262, 266, 262, MUTED, 2.4))
    b.append(transformer(276, 196, 104, 132))
    b.append(t(328, 352, "tiny", 15.5, 650, MUTED, "middle"))
    b.append(t(328, 372, "transformer", 15.5, 650, MUTED, "middle"))
    b.append(arrow(390, 262, 430, 262, MUTED, 2.4))
    bx = Box(442, 90, 340)
    b.append(bx.frame())
    for j in range(3):
        b.append(bx.path(truth[8:16, j], GREY, 6, opacity=0.8))
        b.append(bx.path(truth[15:28, j], INK, 2.6))
        b.append(bx.path(roll[15:28, j], EMBER, 3.2, "7 6"))
        b.append(bx.ball(roll[27, j], EMBER, 0.9))
    b.append(t(442, 456, "— real", 16, 700, INK))
    b.append(t(512, 456, "- - imagined by the model (12 frames)", 16, 700, EMBER_D))
    b.append(f'<path d="M20,468 H860" stroke="{LINE}" stroke-width="1"/>')
    b.append(
        caps(
            f"ONE REAL EPISODE (GRAVITY {g:.2f}, DRAG {drag:.2f}) · THE MODEL'S OWN ROLLOUT FROM 16 TRUE FRAMES · 803,078 PARAMETERS"
        )
    )
    label = (
        "Three of 16 real frames of three balls in a box go into a tiny transformer, which imagines the next 12 frames. "
        "Its dashed orange paths follow the simulator's black ones closely, curving down under gravity and bouncing "
        f"off the floor. One real episode, gravity {g:.2f}, drag {drag:.2f}."
    )
    return svg("".join(b), label, "wm-cover")


def main():
    s = cover_svg(*data())
    write_cover(REPORT / "index.html", s)
    render_hero(s, REPORT, THUMB)
    print("cover: wrote", REPORT / "figures" / "hero.jpg", THUMB)


if __name__ == "__main__":
    main()
