"""Draw the cover from the study's own data: the six "altitude" neurons in Llama-3.1-8B, and one real steered reply.

    python experiments/neuron-communities/make_cover.py

Left: Llama-3.1-8B's 32 MLP layers as rows (layer 0 at the bottom), 14,336 neurons each, with the six members of
community 166512 placed at their real (layer, index). Right: the held-out question "How do bees make honey?" and the
first words of the reply with those six neurons clamped at 2x (results/e2b.jsonl). Writes the SVG into
report/index.html between the cover markers, report/figures/hero.jpg and ../shared/thumbs/neuron-communities.jpg.
"""

# ruff: noqa: E501, E402  (long data lines and pairwise zips read better as they are)

import html
import json
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE / "report"
THUMB = HERE.parent / "shared" / "thumbs" / "neuron-communities.jpg"
sys.path.insert(0, str(HERE.parent / "shared"))
sys.path.insert(0, str(HERE))
import nc
from cover_kit import (
    EMBER,
    EMBER_D,
    INK,
    INK2,
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

ALT, Q = "166512", "How do bees make honey?"


def reply() -> str:
    for line in (HERE / "results" / "e2b.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["id"] == ALT and r["arm"] == "community" and r["q"] == Q:
            return r["text"]
    raise SystemExit("reply not found")


def cover_svg() -> str:
    neurons = nc.members(nc.community(ALT))
    b = [kicker("STEERING · NEURON COMMUNITIES", "LLAMA-3.1-8B-INSTRUCT")]
    b.append(headline("Six neurons, turned up. The answers head for the mountains.", None, 64, 27))
    x0, x1, y_top, y_bot = 40, 380, 104, 430
    step = (y_bot - y_top) / 31
    for layer in range(32):
        y = y_bot - layer * step
        b.append(f'<path d="M{x0},{y:.1f} H{x1}" stroke="{LINE}" stroke-width="{2.2 if layer % 8 == 0 else 1.2}"/>')
        if layer % 8 == 0 or layer == 31:
            b.append(t(x0 - 8, y + 4, str(layer), 10, 500, MUTED, "end", mono=True))
    for layer, idx in neurons:
        cx, cy = x0 + (x1 - x0) * idx / nc.D_MLP, y_bot - layer * step
        b.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7" fill="{EMBER}" stroke="#fff" stroke-width="2"/>')
    b.append(t(x0, y_bot + 24, "32 layers × 14,336 MLP neurons; the 6 orange ones are clamped", 12.5, 650, INK2))
    b.append(arrow(398, 270, 438, 270, MUTED, 2.4))
    b.append(rect(450, 104, 410, 70, "#fff", LINE, 1.5, 10))
    b.append(t(468, 132, "User", 11, 700, MUTED, ls="0.06em"))
    b.append(t(468, 156, Q, 17, 650, INK))
    b.append(rect(450, 188, 410, 242, "#fff", EMBER, 2, 10))
    b.append(t(468, 214, "Llama, six neurons clamped at 2×", 11, 700, EMBER_D, ls="0.06em"))
    words = reply().split("Here's")[0].strip()
    for k, line in enumerate(textwrap.wrap(words, 40)[:8]):
        b.append(t(468, 242 + k * 23, html.escape(line), 15.5, 400, INK))
    b.append(f'<path d="M20,468 H860" stroke="{LINE}" stroke-width="1"/>')
    b.append(
        caps(
            "COMMUNITY 166512 (ALTITUDE) · A REAL, UNEDITED REPLY TO A HELD-OUT QUESTION · ON TOPIC IN 10 OF 15, RANDOM NEURONS 0 OF 15"
        )
    )
    label = (
        "Llama-3.1-8B's 32 MLP layers drawn as rows with six orange neurons at their real positions in layers 3 to 19. "
        "Clamping them at twice their usual activation, the model answers 'How do bees make honey?' with "
        "'Bees make honey at an altitude of about 500 to 2,000 meters above sea level'."
    )
    return svg("".join(b), label, "nc-cover")


def main() -> None:
    s = cover_svg()
    write_cover(REPORT / "index.html", s)
    render_hero(s, REPORT, THUMB)
    print("cover: wrote", REPORT / "figures" / "hero.jpg", THUMB)


if __name__ == "__main__":
    main()
