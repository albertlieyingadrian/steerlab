"""Draw the cover and the research-page thumbnail from the study's own directions.

    python experiments/context-directions/make_cover.py

Every line is a real direction: the difference of class means at layer 14 for one concept in one context, from 20
random halves of its items (a thin sheaf) and from all items (the thick line). One fan per concept (truth, refusal,
chinese, top to bottom), drawn in the plane of the plain direction and the roleplay direction (the vertical scale is
compressed to fit). Orange: the roleplay persona; light: plain; grey: jailbreak. Same inputs, same picture.
Writes report/figures/hero.jpg (1536x1024) and ../shared/thumbs/context-directions.jpg (720x480).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import study as s

GROUND = "#15110e"
COLOURS = {"plain": "#e8dfd4", "roleplay": "#e07a3f", "jailbreak": "#8a8178"}
LAYER = 14


def main() -> None:
    rng = np.random.default_rng(5)
    full, halves = {}, {}
    for concept in s.CONCEPTS:
        for ctx in s.CONTEXTS:
            p, n = s.load_acts(concept, ctx)
            p, n = p[:, LAYER], n[:, LAYER]
            d = p.mean(0) - n.mean(0)
            full[(concept, ctx)] = d / np.linalg.norm(d)
            hs = []
            for _ in range(20):
                ip, inn = rng.permutation(len(p))[: len(p) // 2], rng.permutation(len(n))[: len(n) // 2]
                h = p[ip].mean(0) - n[inn].mean(0)
                hs.append(h / np.linalg.norm(h))
            halves[(concept, ctx)] = np.array(hs)
    fig = plt.figure(figsize=(15.36, 10.24), dpi=100, facecolor=GROUND)
    ax = fig.add_axes((0, 0, 1, 1), facecolor=GROUND)
    # One fan per concept, stacked top to bottom: the plane holds the plain direction (along the fan's axis) and the
    # part of the roleplay direction at right angles to it; the vertical scale is compressed (x 0.4) to fit three fans.
    for row, concept in enumerate(s.CONCEPTS):
        e1 = full[(concept, "plain")]
        r = full[(concept, "roleplay")] - (full[(concept, "roleplay")] @ e1) * e1
        e2 = r / np.linalg.norm(r)
        ox, oy, length = 0.14, 0.70 - row * 0.29, 0.72
        for ctx in ("jailbreak", "plain", "roleplay"):
            for h in halves[(concept, ctx)]:
                ax.plot([ox, ox + length * (h @ e1)], [oy, oy + length * (h @ e2) * 0.4], color=COLOURS[ctx],
                        lw=1.0, alpha=0.28, solid_capstyle="round")  # fmt: skip
            f = full[(concept, ctx)]
            fx, fy = ox + length * (f @ e1), oy + length * (f @ e2) * 0.4
            ax.plot([ox, fx], [oy, fy], color=COLOURS[ctx], lw=3.0, alpha=0.95, solid_capstyle="round")
            ax.scatter([fx], [fy], s=55, color=COLOURS[ctx], zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    out = HERE / "report" / "figures" / "hero.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".png")
    fig.savefig(tmp, facecolor=GROUND)
    plt.close(fig)
    img = Image.open(tmp).convert("RGB").resize((1536, 1024), Image.LANCZOS)
    img.save(out, quality=90)
    img.resize((720, 480), Image.LANCZOS).save(HERE.parent / "shared" / "thumbs" / "context-directions.jpg", quality=88)
    tmp.unlink()
    print(f"wrote {out.relative_to(HERE.parents[1])} and the thumbnail")


if __name__ == "__main__":
    main()
