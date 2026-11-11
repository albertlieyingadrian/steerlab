"""Draw the note's cover from its own results, with no text: seven strips of the 1,000 FinanceMath test questions.

    python experiments/fin-math/make_cover.py

Columns are test questions, sorted from the most to the fewest systems answering them right. Rows, top to bottom:
gpt-5.4, gpt-5.4-mini, base Qwen3-8B, then the four post-trained Qwen3-8B systems (teacher-trace SFT, own-trace SFT,
GRPO from base, own-trace SFT + GRPO). A dark mark is a right answer; for a post-trained system an orange mark is a
question it got right that base Qwen3-8B got wrong, and a hollow orange mark one it got wrong that base got right.
Writes report/figures/hero.jpg (1536 x 1024) and experiments/shared/thumbs/fin-math.jpg (720 x 480).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

HERE = Path(__file__).resolve().parent
RUNS = HERE / "results" / "runs"
CREAM, INK, EMBER, RULE = "#F3EDE3", "#2B2723", "#C9561C", "#D9D0C2"
RIVALS = ["gpt-5.4", "gpt-5.4-mini", "qwen"]
TRAINED = ["sft-e1", "rft-e1", "grpo-base-s15", "grpo-rft-e1-s20"]


def correct(tag: str) -> dict[str, bool]:
    return {r["id"]: r["correct"] for r in map(json.loads, (RUNS / f"test__{tag}.jsonl").read_text().splitlines())}


def main() -> None:
    systems = {t: correct(t) for t in RIVALS + TRAINED}
    base = systems["qwen"]
    ids = sorted(base, key=lambda i: (-sum(s[i] for s in systems.values()), i))
    fig = plt.figure(figsize=(15.36, 10.24), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1536)
    ax.set_ylim(1024, 0)
    ax.axis("off")
    fig.patch.set_facecolor(CREAM)
    left, width, top, row_h, gap = 118, 1300, 132, 86, 26
    step = width / len(ids)
    fixed = broke = 0
    for k, tag in enumerate(RIVALS + TRAINED):
        y0 = top + k * (row_h + gap) + (34 if tag in TRAINED else 0)
        ax.plot([left, left + width], [y0 + row_h + 6, y0 + row_h + 6], color=RULE, lw=0.8)
        for x, qid in enumerate(ids):
            px = left + x * step
            right, was = systems[tag][qid], base[qid]
            if tag in TRAINED and right and not was:
                ax.add_patch(Rectangle((px, y0), step * 0.66, row_h, color=EMBER, lw=0))
                fixed += 1
            elif tag in TRAINED and was and not right:
                ax.add_patch(Rectangle((px, y0 + 1), step * 0.66, row_h - 2, fill=False, edgecolor=EMBER, lw=0.7))
                broke += 1
            elif right:
                ax.add_patch(Rectangle((px, y0), step * 0.66, row_h, color=INK, lw=0))
    print(f"cover: {len(ids)} questions x {len(systems)} systems; post-trained stages fixed {fixed}, broke {broke}")
    hero = HERE / "report" / "figures" / "hero.png"
    hero.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(hero, facecolor=CREAM, dpi=100)
    plt.close(fig)
    img = Image.open(hero).convert("RGB")
    img.save(hero.with_suffix(".jpg"), quality=88, optimize=True)
    hero.unlink()
    thumbs = HERE.parent / "shared" / "thumbs"
    img.resize((720, 480), Image.Resampling.LANCZOS).save(thumbs / "fin-math.jpg", quality=86, optimize=True)


if __name__ == "__main__":
    main()
