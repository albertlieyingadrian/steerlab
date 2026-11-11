"""Draw the note's cover from its own results, with no text: 400 test questions under the awareness push.

    python experiments/eval-aware-steer/make_cover.py [model]

Two blocks of six rows each, at the headline layer. Top: eval framing with the policy, pushed toward "real use" at
strengths 0, 0.5, 1, 2, 4, 8 (top row unsteered). Bottom: real-use framing with the policy, pushed toward "test".
Columns are the 400 test questions, in the same order in both blocks (sorted by how often they are answered right
across all twelve rows). A dark mark is a right answer, an ember mark a wrong one.
Needs the cells in WORK (fetch them with `modal volume get eval-aware-steer work/<model>/cells ...`).
Writes report/figures/hero.jpg (1536 x 1024) and experiments/shared/thumbs/eval-aware-steer.jpg (720 x 480).
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

import common as C

CREAM, INK, EMBER, RULE = "#F3EDE3", "#2B2723", "#C9561C", "#D9D0C2"
ALPHAS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]


def grid(name, cond, layer, ans):
    rows = []
    for a in ALPHAS:
        cell = f"{cond}__none__L0__a0" if a == 0 else f"{cond}__awareness__L{layer}__a{a}"
        p = np.load(C.WORK / name / "cells" / f"{cell}.npz")["probs"]
        rows.append(p.argmax(1) == ans)
    return np.array(rows)


def main(name="llama3.1-8b"):
    plan = json.loads((C.RESULTS / name / "steer_plan.json").read_text())
    layer = json.loads((C.RESULTS / name / "verdict.json").read_text())["headline_layer"]
    ans = np.array(plan["answers"])
    top, bottom = grid(name, "eval_down", layer, ans), grid(name, "real_up", layer, ans)
    order = np.argsort(-(top.sum(0) + bottom.sum(0)), kind="stable")
    top, bottom = top[:, order], bottom[:, order]
    fig = plt.figure(figsize=(15.36, 10.24), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1536)
    ax.set_ylim(1024, 0)
    ax.axis("off")
    fig.patch.set_facecolor(CREAM)
    left, width, row_h, gap = 118, 1300, 58, 14
    step = width / top.shape[1]
    for block, y_start in [(top, 120), (bottom, 120 + 6 * (row_h + gap) + 60)]:
        for k in range(block.shape[0]):
            y0 = y_start + k * (row_h + gap)
            for x in range(block.shape[1]):
                ax.add_patch(
                    Rectangle((left + x * step, y0), step * 0.62, row_h, color=INK if block[k, x] else EMBER, lw=0)
                )
        ax.plot([left, left + width], [y_start - 18, y_start - 18], color=RULE, lw=1.0)
    print(
        f"cover: layer {layer}; right answers per row, eval block {top.sum(1).tolist()}, "
        f"real-use block {bottom.sum(1).tolist()} (of {top.shape[1]})"
    )
    hero = C.HERE / "report" / "figures" / "hero.png"
    hero.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(hero, facecolor=CREAM, dpi=100)
    plt.close(fig)
    img = Image.open(hero).convert("RGB")
    img.save(hero.with_suffix(".jpg"), quality=88, optimize=True)
    hero.unlink()
    thumbs = C.HERE.parent / "shared" / "thumbs"
    img.resize((720, 480), Image.Resampling.LANCZOS).save(thumbs / "eval-aware-steer.jpg", quality=86, optimize=True)


if __name__ == "__main__":
    main(*sys.argv[1:2])
