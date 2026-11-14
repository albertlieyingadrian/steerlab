"""Draw the note's cover from its own judge labels, with no text.

    python make_cover.py

Two blocks of seven rows. Rows, top to bottom: untrained, LoRA easy, LoRA hard, LoRA resist, LoRA twin (seed 0),
steering toward yield, steering toward hold (layer 12, alpha 4). Columns: test items in the same order in every row.
Top block: the 320 wrong pushbacks (an ember mark = the reply gave in). Bottom block: the 320 correct pushbacks and 100
new-information messages (a dark mark = the reply updated, ember = it held on to a wrong or out-of-date answer).
Writes report/figures/hero.jpg (1536 x 1024) and ../shared/thumbs/persuadability.jpg (720 x 480).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

import common as C
import judge as J

CREAM, INK, EMBER, RULE = "#F3EDE3", "#2B2723", "#C9561C", "#D9D0C2"
ROWS = ["base", "lora_easy_s0", "lora_hard_s0", "lora_resist_s0", "lora_twin_s0", "steer_easy", "steer_hard"]


def grid(conds, good_if_switched):
    lab = J.labels()
    keys, data = None, []
    for cfg in ROWS:
        rows = {
            (r["id"], r["cond"]): r for r in C.read_jsonl(C.RESULTS / "test" / f"{cfg}.jsonl") if r["cond"] in conds
        }
        if keys is None:
            keys = sorted(rows)
        sw = np.array([lab[J.key(rows[k])]["final"] == "user" for k in keys])
        data.append(sw if good_if_switched else ~sw)  # True = the wanted behaviour (dark)
    return np.array(data)


def main():
    top = grid(("W",), good_if_switched=False)
    bottom = grid(("C", "N"), good_if_switched=True)
    top = top[:, np.argsort(-top.sum(0), kind="stable")]
    bottom = bottom[:, np.argsort(-bottom.sum(0), kind="stable")]
    fig = plt.figure(figsize=(15.36, 10.24), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1536)
    ax.set_ylim(1024, 0)
    ax.axis("off")
    fig.patch.set_facecolor(CREAM)
    left, width, row_h, gap = 118, 1300, 48, 12
    for block, y_start in [(top, 110), (bottom, 110 + 7 * (row_h + gap) + 70)]:
        step = width / block.shape[1]
        for k in range(block.shape[0]):
            y0 = y_start + k * (row_h + gap)
            for x in range(block.shape[1]):
                ax.add_patch(
                    Rectangle((left + x * step, y0), step * 0.62, row_h, color=INK if block[k, x] else EMBER, lw=0)
                )
        ax.plot([left, left + width], [y_start - 18, y_start - 18], color=RULE, lw=1.0)
    print("cover rows:", ROWS)
    print("held against wrong pushback (of 320):", top.sum(1).tolist())
    print("updated on correct pushback or news (of 420):", bottom.sum(1).tolist())
    hero = C.HERE / "report" / "figures" / "hero.png"
    hero.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(hero, facecolor=CREAM, dpi=100)
    plt.close(fig)
    img = Image.open(hero).convert("RGB")
    img.save(hero.with_suffix(".jpg"), quality=88, optimize=True)
    hero.unlink()
    thumbs = C.HERE.parent / "shared" / "thumbs"
    thumbs.mkdir(parents=True, exist_ok=True)
    img.resize((720, 480), Image.Resampling.LANCZOS).save(thumbs / "persuadability.jpg", quality=86, optimize=True)


if __name__ == "__main__":
    main()
