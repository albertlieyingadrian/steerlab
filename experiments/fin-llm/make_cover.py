"""Draw the note's cover from its own results: a ledger of the 300 test questions every model answered.

    python experiments/fin-llm/make_cover.py

Columns are questions, sorted from the most to the fewest systems answering them right. Each model has two rows,
raw (upper) and in the harness (lower). A dark mark is a right answer; an orange mark is a question the harness got
right and the raw model got wrong; a hollow orange mark is one the harness got wrong after the raw model got it right.
Writes report/figures/hero.jpg (1536 x 1024) and experiments/shared/thumbs/fin-llm.jpg (720 x 480).
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
from make_figures import MODELS, SUBSET, test_rows

CREAM, INK, EMBER, RULE = "#F3EDE3", "#2B2723", "#C9561C", "#D9D0C2"


def main() -> None:
    systems = {(key, mode): test_rows(mode, key, SUBSET) for key, _ in MODELS for mode in ("raw", "harness")}
    ids = sorted(SUBSET, key=lambda i: (-sum(s[i]["correct"] for s in systems.values()), i))
    fig = plt.figure(figsize=(15.36, 10.24), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1536)
    ax.set_ylim(1024, 0)
    ax.axis("off")
    fig.patch.set_facecolor(CREAM)
    left, width, top, row_h, gap = 210, 1250, 150, 58, 46
    step = width / len(ids)
    fixed = broke = 0
    for m, (key, label) in enumerate(MODELS):
        raw, har = systems[(key, "raw")], systems[(key, "harness")]
        y0 = top + m * (2 * row_h + gap)
        ax.text(left - 24, y0 + row_h, label.replace(" (300)", ""), ha="right", va="center", fontsize=15,
                family="monospace", color=INK)  # fmt: skip
        for r in range(3):
            ax.plot(
                [left, left + width],
                [y0 + r * row_h, y0 + r * row_h],
                color=RULE,
                lw=0.8,
            )
        for x, qid in enumerate(ids):
            px = left + x * step
            if raw[qid]["correct"]:
                ax.add_patch(Rectangle((px, y0 + 6), step * 0.62, row_h - 12, color=INK, lw=0))
            if har[qid]["correct"] and not raw[qid]["correct"]:
                ax.add_patch(Rectangle((px, y0 + row_h + 6), step * 0.62, row_h - 12, color=EMBER, lw=0))
                fixed += 1
            elif har[qid]["correct"]:
                ax.add_patch(Rectangle((px, y0 + row_h + 6), step * 0.62, row_h - 12, color=INK, lw=0))
            elif raw[qid]["correct"]:
                ax.add_patch(Rectangle((px + 0.4, y0 + row_h + 7), step * 0.62 - 0.8, row_h - 14, fill=False,
                                       edgecolor=EMBER, lw=0.9))  # fmt: skip
                broke += 1
    print(f"cover: {len(ids)} questions x {len(MODELS)} models; harness fixed {fixed}, broke {broke} (all models)")
    hero = HERE / "report" / "figures" / "hero.png"
    fig.savefig(hero, facecolor=CREAM, dpi=100)
    plt.close(fig)
    img = Image.open(hero).convert("RGB")
    img.save(hero.with_suffix(".jpg"), quality=88, optimize=True)
    hero.unlink()
    thumbs = HERE.parent / "shared" / "thumbs"
    img.resize((720, 480), Image.Resampling.LANCZOS).save(thumbs / "fin-llm.jpg", quality=86, optimize=True)


if __name__ == "__main__":
    main()
