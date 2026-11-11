"""Draw the note's cover from its own results, with no text: five strips of the 288 test transmissions.

    python experiments/gmaps/make_cover.py

Columns are test transmissions, sorted from the most to the fewest systems getting the full record right. Rows, top to
bottom: Claude Fable 5.2 (20-shot), Claude Opus 5.5 (20-shot), the GMAPS winner (Nemotron-30B + LoRA), then our two
Prague-free Qwen3-14B models: without read-backs and the final one with read-backs. A dark mark is a right record; on
the final row an orange mark is a transmission the read-backs fixed (wrong without them) and a hollow orange mark one
they broke. Writes report/figures/hero.jpg (1536 x 1024) and experiments/shared/thumbs/gmaps.jpg (720 x 480).
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
CREAM, INK, EMBER, RULE = "#F3EDE3", "#2B2723", "#C9561C", "#D9D0C2"
ROWS = [
    "claude-fable-5-2-medium-fs20",
    "claude-opus-5-5-medium-fs20",
    "GMAPS: Nemotron-3-Nano-30B + LoRA",
    "q14-pfa-epoch1",
    "q14-pfr-epoch1",
]


def main() -> None:
    rows = json.loads((HERE / "results" / "rows_test.json").read_text())
    right = {s: {r["id"]: r["systems"][s]["full"] for r in rows} for s in ROWS}
    ids = sorted(right[ROWS[0]], key=lambda i: (-sum(right[s][i] for s in ROWS), i))
    fig = plt.figure(figsize=(15.36, 10.24), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1536)
    ax.set_ylim(1024, 0)
    ax.axis("off")
    fig.patch.set_facecolor(CREAM)
    left, width, top, row_h, gap = 118, 1300, 150, 120, 30
    step = width / len(ids)
    fixed = broke = 0
    for k, s in enumerate(ROWS):
        y0 = top + k * (row_h + gap) + (40 if k >= 3 else 0)
        ax.plot([left, left + width], [y0 + row_h + 7, y0 + row_h + 7], color=RULE, lw=0.8)
        for x, i in enumerate(ids):
            px = left + x * step
            ok, before = right[s][i], right["q14-pfa-epoch1"][i]
            if s == "q14-pfr-epoch1" and ok and not before:
                ax.add_patch(Rectangle((px, y0), step * 0.7, row_h, color=EMBER, lw=0))
                fixed += 1
            elif s == "q14-pfr-epoch1" and before and not ok:
                ax.add_patch(Rectangle((px, y0 + 1), step * 0.7, row_h - 2, fill=False, edgecolor=EMBER, lw=0.9))
                broke += 1
            elif ok:
                ax.add_patch(Rectangle((px, y0), step * 0.7, row_h, color=INK, lw=0))
    print(f"cover: {len(ids)} transmissions x {len(ROWS)} systems; read-backs fixed {fixed}, broke {broke}")
    hero = HERE / "report" / "figures" / "hero.png"
    hero.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(hero, facecolor=CREAM, dpi=100)
    plt.close(fig)
    img = Image.open(hero).convert("RGB")
    img.save(hero.with_suffix(".jpg"), quality=88, optimize=True)
    hero.unlink()
    thumbs = HERE.parent / "shared" / "thumbs"
    img.resize((720, 480), Image.Resampling.LANCZOS).save(thumbs / "gmaps.jpg", quality=86, optimize=True)


if __name__ == "__main__":
    main()
