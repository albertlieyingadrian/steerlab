"""Draw the cover from the data: Indonesia's two hills on one question, and where eight models sit from Indonesia.

    python experiments/pluralism-explainer/make_cover.py

Left: Indonesia's survey split on "less importance placed on work" (GlobalOpinionQA goqa-2474), with the average
position marked. Right: each model's mean JS distance to Indonesia over the 181 questions, with no country in the
prompt (hollow) and with "I live in Indonesia." (filled), against the floor a perfect sampler of Indonesia would
score with 50 answers. From report/data.json. Writes the SVG into report/index.html between the cover markers,
report/figures/hero.jpg and ../shared/thumbs/pluralism-explainer.jpg.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE / "report"
THUMB = HERE.parent / "shared" / "thumbs" / "pluralism-explainer.jpg"
sys.path.insert(0, str(HERE.parent / "shared"))
from cover_kit import (
    EMBER,
    EMBER_D,
    INK,
    INK2,
    LINE,
    MUTED,
    caps,
    headline,
    kicker,
    rect,
    render_hero,
    svg,
    t,
    write_cover,
)

GREY = "#9a8f82"


def cover_svg(data):
    q = next(x for x in data["questions"] if x["id"] == "goqa-2474")
    h = q["human"]["Indonesia"]
    S = data["summary"]
    b = [kicker("PLURALISM · EPISODE 1", "WHAT IS PLURALISTIC ALIGNMENT?")]
    b.append(headline("One question, three answers people hold", None, 66, 28))
    # left: the split
    x0, y0, bw, bh = 30, 140, 64, 230
    b.append(t(x0, 112, "Indonesia: less importance on work would be…", 12.5, 600, INK2))
    for i, (o, p) in enumerate(zip(q["options"], h, strict=True)):
        x = x0 + i * (bw + 26)
        hh = bh * p
        b.append(rect(x, y0 + bh - hh, bw, hh, INK if i != 1 else GREY, rx=3))
        b.append(t(x + bw / 2, y0 + bh - hh - 8, f"{p:.0%}", 16, 800, INK, "middle"))
        b.append(t(x + bw / 2, y0 + bh + 20, o, 12, 600, INK2, "middle"))
    b.append(f'<path d="M{x0 - 6},{y0 + bh} H{x0 + 3 * bw + 58}" stroke="{LINE}" stroke-width="1.5"/>')
    mid = x0 + bw + 26 + bw / 2
    b.append(
        f'<path d="M{mid},{y0 + 40} V{y0 + bh - bh * h[1] - 28}" stroke="{EMBER}" stroke-width="2" stroke-dasharray="4 4"/>'
    )
    b.append(t(mid, y0 + 32, f"the average answer: held by {h[1]:.0%}", 12, 700, EMBER_D, "middle"))
    # right: distance dots
    rx, ry, rw = 500, 120, 340
    lo, hi = 0.0, 0.8

    def X(v):
        return rx + (v - lo) / (hi - lo) * rw

    b.append(t(rx, 104, "Mean distance to Indonesia's answers, 181 questions", 12.5, 600, INK2))
    fl = S["human"]["Indonesia"]["floor"]["mean"]
    rows = data["models"]
    rh = 30
    b.append(
        f'<path d="M{X(fl):.1f},{ry} V{ry + len(rows) * rh}" stroke="{MUTED}" stroke-width="1.5" stroke-dasharray="3 3"/>'
    )
    b.append(t(X(fl) + 4, ry + len(rows) * rh + 14, f"floor {fl:.2f}", 10.5, 600, MUTED))
    un = S["human"]["Indonesia"]["uniform_sampled"]["mean"]
    b.append(
        f'<path d="M{X(un):.1f},{ry} V{ry + len(rows) * rh}" stroke="{MUTED}" stroke-width="1.5" stroke-dasharray="1 4"/>'
    )
    b.append(t(X(un) + 4, ry + len(rows) * rh + 14, f"uniform guess {un:.2f}", 10.5, 600, MUTED))
    for v in (0.0, 0.2, 0.4, 0.6, 0.8):
        b.append(t(X(v), ry - 2, f"{v:.1f}", 9.5, 500, MUTED, "middle"))
    for r, m in enumerate(rows):
        y = ry + 12 + r * rh
        P = S["per_model"][m["key"]]["jsd"]
        d, s = P["default"]["Indonesia"]["mean"], P["Indonesia"]["Indonesia"]["mean"]
        b.append(t(rx - 8, y + 4, m["label"], 11.5, 600, INK, "end"))
        b.append(f'<path d="M{X(min(d, s)):.1f},{y} H{X(max(d, s)):.1f}" stroke="{LINE}" stroke-width="3"/>')
        b.append(f'<circle cx="{X(d):.1f}" cy="{y}" r="6" fill="#fff" stroke="{GREY}" stroke-width="2"/>')
        b.append(f'<circle cx="{X(s):.1f}" cy="{y}" r="6" fill="{EMBER}"/>')
    yl = ry + len(rows) * rh + 34
    b.append(f'<circle cx="{rx + 6}" cy="{yl}" r="6" fill="#fff" stroke="{GREY}" stroke-width="2"/>')
    b.append(t(rx + 18, yl + 4, "no country in the prompt", 11.5, 600, INK2))
    b.append(f'<circle cx="{rx + 196}" cy="{yl}" r="6" fill="{EMBER}"/>')
    b.append(t(rx + 208, yl + 4, "“I live in Indonesia.”", 11.5, 600, INK2))
    b.append(f'<path d="M20,468 H860" stroke="{LINE}" stroke-width="1"/>')
    b.append(
        caps(
            "WORLD VALUES SURVEY WAVE 7 VIA GLOBALOPINIONQA · 8 COUNTRIES · 8 MODELS · 50 SAMPLED ANSWERS PER QUESTION"
        )
    )
    label = (f"Left: Indonesia's survey split on whether less importance placed on work would be good ({h[0]:.0%}), "
             f"don't mind ({h[1]:.0%}) or bad ({h[2]:.0%}); the average position falls on the answer 9% hold. "
             "Right: eight models' mean Jensen-Shannon distance to Indonesia's answers over 181 questions, without "
             f"and with 'I live in Indonesia.', against a sampling floor of {fl:.2f} and a uniform guess.")  # fmt: skip
    return svg("".join(b), label, "plu-cover")


def main():
    data = json.loads((REPORT / "data.json").read_text())
    s = cover_svg(data)
    write_cover(REPORT / "index.html", s)
    render_hero(s, REPORT, THUMB)
    print("cover: wrote", REPORT / "figures" / "hero.jpg", THUMB)


if __name__ == "__main__":
    main()
