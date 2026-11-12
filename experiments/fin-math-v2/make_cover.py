"""Draw the FinanceMath v2 cover from the test results, with no text: the 1,000 test questions as a grid of cells.

    python experiments/fin-math-v2/make_cover.py

Each cell is one test question, grouped by outcome and read left to right, top to bottom: dark when both gpt-5.4 (one
call) and DeepSeek-V4-Flash thinking (vote of 4) are right; solid orange when only the open model is right; an orange
outline when only gpt-5.4 is right; empty when both are wrong. Writes report/figures/hero.jpg (1536 x 1024) and
../shared/thumbs/fin-math-v2.jpg (720 x 480).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_figures import test_rows

CREAM, INK, EMBER, RULE = "#F3EDE3", "#2B2723", "#C9561C", "#D9D0C2"


def main() -> None:
    ids, systems = test_rows()
    gpt = dict(zip(ids, systems[2][1], strict=True))
    ours = dict(zip(ids, systems[5][1], strict=True))
    order = {"both": 0, "ours": 1, "gpt": 2, "none": 3}
    state = {i: "both" if gpt[i] and ours[i] else "ours" if ours[i] else "gpt" if gpt[i] else "none" for i in ids}
    cells = sorted(ids, key=lambda i: (order[state[i]], i))
    cols, rows_n = 50, 20
    fig = plt.figure(figsize=(15.36, 10.24), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1536)
    ax.set_ylim(1024, 0)
    ax.axis("off")
    fig.patch.set_facecolor(CREAM)
    size, gap = 24, 4
    left = (1536 - cols * (size + gap) + gap) / 2
    top = (1024 - rows_n * (size + gap) + gap) / 2
    for k, i in enumerate(cells):
        x = left + (k % cols) * (size + gap)
        y = top + (k // cols) * (size + gap)
        s = state[i]
        if s == "both":
            ax.add_patch(Rectangle((x, y), size, size, color=INK, lw=0))
        elif s == "ours":
            ax.add_patch(Rectangle((x, y), size, size, color=EMBER, lw=0))
        elif s == "gpt":
            ax.add_patch(Rectangle((x + 1, y + 1), size - 2, size - 2, fill=False, edgecolor=EMBER, lw=1.6))
        else:
            ax.add_patch(Rectangle((x + 1, y + 1), size - 2, size - 2, fill=False, edgecolor=RULE, lw=1.0))
    counts = {s: sum(v == s for v in state.values()) for s in order}
    print(f"cover: {len(cells)} questions: {counts}")
    hero = HERE / "report" / "figures" / "hero.png"
    hero.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(hero, facecolor=CREAM, dpi=100)
    plt.close(fig)
    img = Image.open(hero).convert("RGB")
    img.save(hero.with_suffix(".jpg"), quality=88, optimize=True)
    hero.unlink()
    img.resize((720, 480), Image.Resampling.LANCZOS).save(HERE.parent / "shared" / "thumbs" / "fin-math-v2.jpg",
                                                          quality=86, optimize=True)  # fmt: skip


if __name__ == "__main__":
    main()
