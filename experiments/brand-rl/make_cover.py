"""Draw the cover from the data: one brand-book line (G62), the half the reward paid for and the half it did not, and
what each run's final checkpoint did to both halves on held-out briefs.

    .venv/bin/python experiments/brand-rl/make_cover.py configs/main.json

Reads results/summary.json. Writes the SVG into report/index.html between the cover markers, report/figures/hero.jpg
and ../shared/thumbs/brand-rl.jpg.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))
from cover_kit import (
    EMBER,
    INK,
    INK2,
    LINE,
    MUTED,
    caps,
    kicker,
    rect,
    render_hero,
    svg,
    t,
    write_cover,
)

import ep6

GREY = "#9a948c"
BLUE = "#3f6fb0"
ARM = {"check": "CHECK", "env": "ENV", "team": "TEAM", "shuf": "SHUF"}


def bars(b, x0, y, rows, width=210, step=17):
    for label, v, colour in rows:
        w = width * max(v, 0)
        b.append(t(x0, y + 12, label, 10.5, 500, INK2))
        b.append(rect(x0 + 112, y + 1, max(w, 1.5), 14, colour))
        b.append(t(x0 + 118 + w, y + 12.5, f"{v:.2f}", 10, 600, INK))
        y += step
    return y


def cover_svg(cfg) -> str:
    S = json.loads((ep6.RESULTS / "summary.json").read_text())
    H = S["heldout"]
    runs = [{**cfg["common"], **r} for r in cfg["runs"]]
    b = [kicker("THE TASTE STACK · EPISODE 6", "AN RL ENVIRONMENT")]
    b.append(t(20, 62, "Paid for half a sentence, four of six runs learned only that half", 22, 800, INK))
    b.append(rect(20, 84, 410, 120, "#ffffff", LINE, 1, 6))
    b.append(t(34, 106, "BRAND-BOOK LINE G62 (GOV.UK DESIGN SYSTEM)", 9.5, 700, MUTED, ls="0.08em"))
    b.append(t(34, 132, "“Links are", 15, 400, INK, italic=True))
    b.append(t(112, 132, "blue", 15, 800, INK, italic=True))
    b.append(t(150, 132, "and", 15, 400, INK, italic=True))
    b.append(t(182, 132, "underlined", 15, 800, INK, italic=True))
    b.append(t(268, 132, "by default.”", 15, 400, INK, italic=True))
    b.append(t(34, 160, "“blue”: rule LINK-COL, in the reward", 11, 700, INK2))
    b.append(t(34, 180, "“underlined”: rule LINK-UL, held out of every reward", 11, 700, INK2))
    b.append(rect(20, 222, 410, 200, "#ffffff", LINE, 1, 6))
    b.append(t(34, 244, "ALL FIVE HELD-OUT RULES, STRICT PASS (HELD-OUT BRIEFS)", 9.5, 700, MUTED, ls="0.08em"))
    yy = 258
    rows = [("untrained", H["base"]["heldout_strict"]["mean"], GREY)]
    rows += [
        (f"{ARM[r['arm']]} s{r['seed']}", H[f"{r['name']}:final"]["heldout_strict"]["mean"], EMBER)
        for r in runs
        if f"{r['name']}:final" in H and r["arm"] != "shuf"
    ]
    for label, v, col in rows:
        b.append(t(34, yy + 10, label, 10, 500, INK2))
        w = 230 * max(0.0, v)
        b.append(rect(110, yy, max(w, 1.5), 12, col))
        b.append(t(116 + w, yy + 10, f"{v:.2f}", 9.5, 600, INK))
        yy += 20
    b.append(t(34, 416, "the seed decides which way they move", 9, 400, MUTED))
    x0, y = 470, 96
    for fam, title, colour in (
        ("LINK-COL", "Links in the brand colour (rewarded)", EMBER),
        ("LINK-UL", "Links underlined (held out)", EMBER),
    ):
        b.append(t(x0, y, f"{title}: strict pass, held-out", 11.5, 700, INK2))
        rows = [("untrained", H["base"][f"fam_strict_{fam}"]["mean"], GREY)]
        for r in runs:
            if r["arm"] != "shuf" and f"{r['name']}:final" in H:
                rows.append(
                    (f"{ARM[r['arm']]} seed {r['seed']}", H[f"{r['name']}:final"][f"fam_strict_{fam}"]["mean"], colour)
                )
        y = bars(b, x0, y + 10, rows) + 18
    b.append(f'<path d="M20,468 H860" stroke="{LINE}" stroke-width="1"/>')
    b.append(caps("QWEN3-4B · GRPO · 200 STEPS · REWARD = EPISODE 5'S PROOF ARTIFACT · 96 HELD-OUT BRIEFS × 4 PAGES"))
    label = (
        "Left: GOV.UK line G62, 'Links are blue and underlined by default'; the colour half is a rewarded rule, the "
        "underline half is held out of every reward. Right: strict pass rates of the two rules on held-out pages "
        "for the untrained model and the final checkpoint of each CHECK, ENV and TEAM run; bottom left, all five held-out rules."
    )
    return svg("".join(b), label, "rl-cover")


def main() -> None:
    cfg = json.loads((HERE / sys.argv[1]).read_text())
    s = cover_svg(cfg)
    report = HERE / "report"
    write_cover(report / "index.html", s)
    render_hero(s, report, HERE.parent / "shared" / "thumbs" / "brand-rl.jpg")
    print("cover written; hero.jpg and thumb rendered")


if __name__ == "__main__":
    main()
