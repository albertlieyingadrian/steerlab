"""Draw the cover from the study's own results: one looped checkpoint that uses its loops, one that does not.

    python experiments/looped-transformer/make_cover.py

Left: the looped model as a diagram (a prelude block, a shared two-block core run k times with the input re-added,
a coda block). Right: two grids of real test accuracy, loops k = 1-16 (rows) by hops 1-24 (columns), for the looped
model's seed 2 (each extra loop resolves more hops) and seed 1 (extra loops add little), from
results/runs/looped_s2.json and looped_s1.json (500 items per cell). Writes the SVG into report/index.html between
the cover markers, report/figures/hero.jpg and ../shared/thumbs/looped-transformer.jpg.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE / "report"
THUMB = HERE.parent / "shared" / "thumbs" / "looped-transformer.jpg"
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

KS = ["1", "2", "4", "8", "16"]


def mix(a, b, f):
    a = [int(a[i : i + 2], 16) for i in (1, 3, 5)]
    b = [int(b[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * f):02x}" for x, y in zip(a, b))


def grid(x0, y0, run, title, sub):
    d = json.loads((HERE / "results" / "runs" / f"{run}.json").read_text())
    cw, ch = 21, 22
    out = [t(x0, y0 - 28, title, 15, 750, INK), t(x0, y0 - 11, sub, 11.5, 500, MUTED)]
    for r, k in enumerate(KS):
        out.append(t(x0 - 8, y0 + r * ch + ch * 0.66, f"k={k}", 11, 600, MUTED, "end", mono=True))
        for h in range(1, 25):
            a = d["correct"][k][str(h)] / d["test_n"]
            f = max(0.0, min(1.0, (a - 0.2) / 0.75))
            out.append(rect(x0 + (h - 1) * cw, y0 + r * ch, cw - 2, ch - 2, mix("#efe7dc", EMBER_D, f), rx=1.5))
    out.append(rect(x0 - 2, y0 - 2, 8 * cw + 2, len(KS) * ch + 2, "none", INK, 1.4, rx=2))  # training range
    for h in (1, 8, 16, 24):
        out.append(t(x0 + (h - 0.5) * cw - 1, y0 + len(KS) * ch + 13, str(h), 10.5, 500, MUTED, "middle"))
    return "".join(out)


def model_diagram(x, y):
    out = []
    bw, bh = 96, 38
    out.append(rect(x, y, bw, bh, "#fff", INK, 1.8, rx=7))
    out.append(t(x + bw / 2, y + 24, "prelude", 13, 650, INK, "middle"))
    out.append(arrow(x + bw / 2, y + bh + 2, x + bw / 2, y + bh + 22, MUTED, 2))
    cy = y + bh + 26
    out.append(rect(x, cy, bw, 74, "#fff", EMBER, 2.4, rx=7))
    out.append(rect(x + 10, cy + 10, bw - 20, 22, "#f6dcc8", rx=4))
    out.append(rect(x + 10, cy + 42, bw - 20, 22, "#f6dcc8", rx=4))
    out.append(t(x + bw / 2, cy + 26, "block", 11, 600, EMBER_D, "middle"))
    out.append(t(x + bw / 2, cy + 58, "block", 11, 600, EMBER_D, "middle"))
    rx = x + bw + 6
    out.append(
        f'<path d="M{rx},{cy + 60} C{rx + 40},{cy + 60} {rx + 40},{cy + 14} {rx + 6},{cy + 14}" fill="none" '
        f'stroke="{EMBER}" stroke-width="2.6" stroke-linecap="round"/>'
    )
    out.append(
        f'<path d="M{rx + 13},{cy + 8} L{rx + 4},{cy + 14} L{rx + 13},{cy + 20}" fill="none" stroke="{EMBER}" '
        f'stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>'
    )
    out.append(t(rx + 42, cy + 42, "× k", 18, 800, EMBER_D))
    out.append(t(x - 8, cy + 32, "shared", 11, 600, MUTED, "end"))
    out.append(t(x - 8, cy + 47, "weights", 11, 600, MUTED, "end"))
    out.append(arrow(x + bw / 2, cy + 76, x + bw / 2, cy + 96, MUTED, 2))
    oy = cy + 100
    out.append(rect(x, oy, bw, bh, "#fff", INK, 1.8, rx=7))
    out.append(t(x + bw / 2, oy + 24, "coda", 13, 650, INK, "middle"))
    out.append(t(x + bw / 2, oy + bh + 19, "→ a digit", 12, 600, MUTED, "middle"))
    out.append(t(x + bw / 2, y - 12, "input re-added", 11, 600, MUTED, "middle"))
    out.append(t(x + bw / 2, y - 26, "each loop", 11, 600, MUTED, "middle"))
    return "".join(out)


def cover_svg():
    b = [kicker("MODEL ARCHITECTURE · EPISODE 1", "A SMALL DORY-STYLE LOOPED TRANSFORMER")]
    b.append(model_diagram(70, 92))
    b.append(
        grid(
            330,
            96,
            "looped_s2",
            "Seed 2: every extra loop follows the chain further",
            "test accuracy, loops k (rows) by hops 1-24 (columns); box = hops seen in training",
        )
    )
    b.append(
        grid(
            330,
            282,
            "looped_s1",
            "Seed 1, same recipe: extra loops add little",
            "darker = more chains answered right (500 per cell)",
        )
    )
    b.append(headline("Loop the same layers again: one seed thinks longer, its twin does not", None, 440, 25))
    b.append(f'<path d="M20,458 H860" stroke="{LINE}" stroke-width="1"/>')
    b.append(
        caps(
            "REAL TEST ACCURACY · 832,778 PARAMETERS · TRAINED ON 1-8 HOPS WITH k = 1-6 LOOPS · 1 OF 3 SEEDS LEARNED TO USE THE LOOPS",
            480,
        )
    )
    label = (
        "A looped transformer drawn as a prelude block, a two-block core run k times with shared weights, and a coda. "
        "Beside it, two grids of real test accuracy by hop count and loop count. For seed 2 the solved region grows "
        "with every extra loop, past the 8 hops seen in training; for seed 1, trained the same way, extra loops add little."
    )
    return svg("".join(b), label, "lt-cover")


def main():
    s = cover_svg()
    write_cover(REPORT / "index.html", s)
    render_hero(s, REPORT, THUMB)
    print("cover: wrote", REPORT / "figures" / "hero.jpg", THUMB)


if __name__ == "__main__":
    main()
