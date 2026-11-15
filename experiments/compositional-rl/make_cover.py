"""Draw the cover from the step test itself: can the model do each step, and can it do the chain?

    python experiments/compositional-rl/make_cover.py

Left: one Level II chain (an array, three operations), each step a tile. Right: for the untrained model, RL on single
steps and RL on chains, the share of 450 chains x 3 seeds whose every step is right when handed the true state
(outlined) against the share solved in one reply (filled), from results/summary.json. Writes the SVG into
report/index.html between the cover markers, report/figures/hero.jpg and ../shared/thumbs/compositional-rl.jpg.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE / "report"
THUMB = HERE.parent / "shared" / "thumbs" / "compositional-rl.jpg"
sys.path.insert(0, str(HERE.parent / "shared"))
from cover_kit import (
    EMBER,
    EMBER_D,
    INK,
    INK2,
    LINE,
    MUTED,
    SLATE,
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

BLUE = "#4a86e8"
GREY = "#b9b1a7"


def tile(x, y, w, h, top, bottom, stroke):
    return (
        rect(x, y, w, h, "#fff", stroke, 2, rx=8)
        + t(x + w / 2, y + 24, top, 13, 700, INK, "middle", mono=True)
        + t(x + w / 2, y + 46, bottom, 12, 500, MUTED, "middle", mono=True)
    )


def cover_svg(st):
    b = [kicker("WHAT RL TEACHES · EPISODE 1", "A REPLICATION OF HE ET AL. (2026) AT 1.7B")]
    b.append(headline("Train the steps, or train the chain?", None, 66, 28))
    states = ["[3, 1, 4]", "[3, 9, 1, 4]", "[9, 1, 4]", "[9, 1, 4, 2]"]
    ops = ["insert 9 at 1", "delete at 0", "insert 2 at 3"]
    x0, y0 = 24, 96
    b.append(tile(x0, y0, 150, 58, states[0], "start", LINE))
    for k in range(3):
        y = y0 + 84 * (k + 1)
        b.append(arrow(x0 + 75, y - 24, x0 + 75, y - 2, MUTED, 2))
        b.append(t(x0 + 88, y - 9, ops[k], 11, 600, EMBER_D, mono=True))
        b.append(tile(x0, y, 150, 58, states[k + 1], f"step {k + 1}", EMBER if k == 2 else LINE))
    b.append(t(x0, 452, "one chain = steps that feed each other", 12.5, 600, INK2))
    arms = [("base", "Untrained", GREY), ("atomic", "RL on single steps", BLUE), ("compound", "RL on chains", EMBER)]
    bx, by, bh = 300, 150, 260
    b.append(f'<path d="M{bx},{by + bh} H{bx + 540}" stroke="{LINE}" stroke-width="1.5"/>')
    for g, (key, name, col) in enumerate(arms):
        s, c = st[key]["S"], st[key]["C"]
        gx = bx + 30 + g * 175
        hs, hc = bh * s / 0.62, bh * c / 0.62
        b.append(
            f'<rect x="{gx}" y="{by + bh - hs:.1f}" width="56" height="{hs:.1f}" fill="#fff" stroke="{col}" stroke-width="3" rx="3"/>'
        )
        b.append(f'<rect x="{gx + 66}" y="{by + bh - hc:.1f}" width="56" height="{hc:.1f}" fill="{col}" rx="3"/>')
        b.append(t(gx + 28, by + bh - hs - 8, f"{s:.2f}", 14, 700, INK, "middle"))
        b.append(t(gx + 94, by + bh - hc - 8, f"{c:.2f}", 14, 700, INK, "middle"))
        b.append(t(gx + 61, by + bh + 22, name, 13, 700, INK, "middle"))
    b.append(rect(bx + 10, 96, 14, 14, "#fff", SLATE, 2, rx=2))
    b.append(t(bx + 30, 108, "every step right, when handed the true state", 12.5, 600, INK2))
    b.append(rect(bx + 10, 118, 14, 14, SLATE, rx=2))
    b.append(t(bx + 30, 130, "whole chain right, in one reply", 12.5, 600, INK2))
    b.append(f'<path d="M20,468 H860" stroke="{LINE}" stroke-width="1"/>')
    b.append(caps("QWEN3-1.7B · GRPO · LEVEL II CHAINS OF LENGTH 4-8 · 5 DATA STRUCTURES × 90 CHAINS × 3 SEEDS"))
    label = (
        "Left: a three-step chain on an array, each step's state feeding the next. Right: share of Level II chains "
        "whose every step is right when the model is handed the true state, against the share it gets right in one "
        f"reply. Untrained {st['base']['S']:.2f} and {st['base']['C']:.2f}; RL on single steps "
        f"{st['atomic']['S']:.2f} and {st['atomic']['C']:.2f}; RL on chains {st['compound']['S']:.2f} and "
        f"{st['compound']['C']:.2f}."
    )
    return svg("".join(b), label, "crl-cover")


def main():
    st = json.loads((HERE / "results" / "summary.json").read_text())["mechanism"]["step_test"]
    s = cover_svg(st)
    write_cover(REPORT / "index.html", s)
    render_hero(s, REPORT, THUMB)
    print("cover: wrote", REPORT / "figures" / "hero.jpg", THUMB)


if __name__ == "__main__":
    main()
