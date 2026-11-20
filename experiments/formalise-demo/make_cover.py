"""Draw the cover from the data: one rule three ways, and what each arm scored on the held-out questions.

    .venv/bin/python experiments/formalise-demo/make_cover.py

Left: source line L275 of the rulebook, the formal rule built from it (spec/baggage_spec.json) and the proof step the
checker emits when it fires. Right: exact answers on the 200 held-out baggage questions for each arm and model, from
results/summary.json. Writes the SVG into report/index.html between the cover markers, report/figures/hero.jpg and
../shared/thumbs/formalise-demo.jpg.
"""

from __future__ import annotations

import json
import sys
import textwrap

from common import HERE, RESULTS, load_spec

REPORT = HERE / "report"
THUMB = HERE.parent / "shared" / "thumbs" / "formalise-demo.jpg"
sys.path.insert(0, str(HERE.parent / "shared"))
from cover_kit import EMBER, EMBER_D, INK, INK2, LINE, MUTED, caps, kicker, rect, render_hero, svg, t, write_cover

MODELS = {"qwen3-8b": "Qwen3-8B", "qwen3-32b": "Qwen3-32B"}
GREY, GREY_L = "#9a948c", "#cfc8bf"


def card(y: int, label: str, body: str, mono: bool, accent: bool = False) -> str:
    out = rect(20, y, 410, 96, "#ffffff", LINE, 1, 6)
    out += t(34, y + 22, label, 9.5, 700, EMBER_D if accent else MUTED, ls="0.08em")
    size, width = (11.5, 58) if mono else (12.5, 60)
    return out + "".join(
        t(34, y + 44 + i * 17, ln, size, 400, INK, mono=mono) for i, ln in enumerate(textwrap.wrap(body, width))
    )


def cover_svg() -> str:
    s = json.loads((RESULTS / "summary.json").read_text())["baggage"]["heldout"]
    rule = next(r for r in load_spec()["rules"] if r["id"] == "L275")
    b = [kicker("TASTE · EPISODE 4", "WHAT DOES IT MEAN TO FORMALISE A DOMAIN?")]
    b.append(t(20, 62, "One rule, three ways; and what a checker adds", 25, 800, INK))
    b.append(card(84, "SOURCE LINE L275", f"“{rule['text'].lstrip('* ')}”", False))
    b.append(t(225, 196, "↓", 16, 700, MUTED, "middle"))
    b.append(
        card(
            204,
            "FORMAL RULE (EXCEPTION TO L274)",
            "when class in {Business, First} and the bag is free: weight allowance = 70 lb",
            True,
        )
    )
    b.append(t(225, 316, "↓", 16, 700, MUTED, "middle"))
    b.append(
        card(
            324,
            "PROOF STEP",
            "item 2 · 62 lb · free bag · allowance 70 lb → overweight $0 · rule L275 · pass",
            True,
            accent=True,
        )
    )
    # right: exact answers per arm
    x0, x1, top = 470, 850, 104
    b.append(t(x0, 96, "Exact fee totals, 200 held-out questions", 12.5, 700, INK2))
    y = top
    for m, name in MODELS.items():
        if m not in s:
            continue
        b.append(t(x0, y + 12, name, 11.5, 800, INK))
        y += 20
        for label, r, colour in (
            ("no rules", s[m]["a"]["exact"], GREY_L),
            ("rules pasted", s[m]["b"]["exact"], GREY),
            ("+ checker", s[m]["b"]["c_exact"], EMBER),
        ):
            w = (x1 - x0 - 120) * r["rate"]
            b.append(t(x0, y + 13, label, 11, 600 if colour == EMBER else 500, EMBER_D if colour == EMBER else INK2))
            b.append(rect(x0 + 92, y + 2, max(w, 1.5), 15, colour))
            b.append(t(x0 + 98 + w, y + 14, f"{r['k']}/{r['n']}", 10.5, 700 if colour == EMBER else 500, INK))
            y += 24
        y += 18
    b.append(t(x0, y + 4, "The checker re-derives every fee from the model's own facts;", 10.5, 400, MUTED))
    b.append(t(x0, y + 19, "it fails only where those facts were wrong.", 10.5, 400, MUTED))
    b.append(f'<path d="M20,468 H860" stroke="{LINE}" stroke-width="1"/>')
    b.append(caps("RULEARENA AIRLINE SLICE · 162 RULES FROM 158 SOURCE LINES · QWEN3, THINKING ON · VLLM ON ONE H100"))
    label = (
        "Left: a source line of the baggage rulebook (Business and First may carry 70 lb in a free bag), the formal "
        "exception rule built from it, and the proof step the checker emits when it fires. Right: the share of 200 "
        "held-out baggage questions answered with the exact total, for each model with no rules, with the rules "
        "pasted, and with a checker verifying the answer."
    )
    return svg("".join(b), label, "fd-cover")


def main() -> None:
    s = cover_svg()
    write_cover(REPORT / "index.html", s)
    render_hero(s, REPORT, THUMB)
    print("cover written; hero.jpg and thumb rendered")


if __name__ == "__main__":
    main()
