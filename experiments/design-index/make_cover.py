"""Draw the cover from the data: palettes of the same sites by year, and where AI-builder sites' nearest pages are.

    .venv/bin/python experiments/design-index/make_cover.py

Left: 12 palettes per even year from the held-out archive panel (report/data.js, the k-means colours, widths = share).
Right: for each held-out page, how often its most similar page from another industry is an AI-builder site
(results/house.json, page structure and image embedding). Writes the SVG into report/index.html between the cover
markers, report/figures/hero.jpg and ../shared/thumbs/design-index.jpg.
"""

from __future__ import annotations

import json
import sys

from common import HERE, RESULTS

REPORT = HERE / "report"
THUMB = HERE.parent / "shared" / "thumbs" / "design-index.jpg"
sys.path.insert(0, str(HERE.parent / "shared"))
from cover_kit import (
    EMBER,
    EMBER_D,
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

GREY = "#9a948c"


def cover_svg() -> str:
    data = json.loads(
        (REPORT / "data.js").read_text().split("=", 1)[1].rstrip().rstrip(";")
    )
    house = json.loads((RESULTS / "house.json").read_text())
    b = [kicker("TASTE · EPISODE 2", "HAS THE WEB CONVERGED?")]
    b.append(
        t(
            20,
            62,
            "The same top sites did not converge; AI-built ones share a skeleton",
            23,
            800,
            INK,
        )
    )
    # left: palettes by year
    x0, y0, colw, rows, rowh = 20, 104, 58, 12, 22
    b.append(
        t(
            x0,
            94,
            "Palettes of sites from our archive panel, by year (12 each)",
            12,
            700,
            INK2,
        )
    )
    for k, yr in enumerate(data["years"][:7]):
        x = x0 + k * (colw + 4)
        b.append(
            t(
                x + colw / 2,
                y0 + rows * rowh + 18,
                str(yr["year"]),
                11,
                600,
                INK2,
                "middle",
            )
        )
        for r, pal in enumerate(yr["palettes"][:rows]):
            cx = x
            for hexc, w in pal:
                ww = max(w * colw, 0.8)
                b.append(
                    rect(round(cx, 2), y0 + r * rowh, round(ww, 2), rowh - 6, hexc)
                )
                cx += ww
    # right: nearest page from another industry
    x1, top, maxw = 480, 104, 260
    b.append(
        t(
            x1,
            94,
            "Nearest page from another industry is an AI-builder site",
            12,
            700,
            INK2,
        )
    )
    y = top + 6
    for space, lab in (("dom", "page structure"), ("siglip", "image embedding")):
        b.append(t(x1, y + 12, lab, 11.5, 800, INK))
        y += 20
        for grp, name, col in (
            ("ai_host", "AI-builder sites", EMBER),
            ("human", "human sites", GREY),
        ):
            r = house[space][grp]
            w = maxw * r["rate"]
            b.append(
                t(
                    x1,
                    y + 13,
                    name,
                    11,
                    600 if col == EMBER else 500,
                    EMBER_D if col == EMBER else INK2,
                )
            )
            b.append(rect(x1 + 104, y + 2, max(w, 1.5), 15, col))
            b.append(
                t(
                    x1 + 110 + w,
                    y + 14,
                    f"{100 * r['rate']:.0f}%",
                    10.5,
                    700 if col == EMBER else 500,
                    INK,
                )
            )
            y += 24
        y += 14
    share = 100 * house["dom"]["human"]["chance"]
    b.append(
        t(
            x1,
            y + 6,
            f"With the labels shuffled, either group would score about {share:.0f}%:",
            10.5,
            400,
            MUTED,
        )
    )
    b.append(
        t(
            x1,
            y + 21,
            "AI-builder sites are that share of the other industries' pages.",
            10.5,
            400,
            MUTED,
        )
    )
    b.append(f'<path d="M20,468 H860" stroke="{LINE}" stroke-width="1"/>')
    b.append(
        caps(
            "106,377 PAGES · WAYBACKUI 2014-24 · OUR ARCHIVE PANEL 2014-26 · LIVE HUMAN, AI-BUILDER AND TEMPLATED PAGES"
        )
    )
    label = (
        "Left: twelve five-colour palettes per even year from 2014 to 2026, for the same small sites rendered from the "
        "Wayback Machine; no year looks more uniform than another. Right: for each held-out page, how often the most "
        f"similar page from a different industry is an AI-builder site: about {100 * house['dom']['ai_host']['rate']:.0f}% "
        f"for AI-builder sites and {100 * house['dom']['human']['rate']:.0f}% for human sites by page structure, "
        f"{100 * house['siglip']['ai_host']['rate']:.0f}% and {100 * house['siglip']['human']['rate']:.0f}% by image "
        "embedding."
    )
    return svg("".join(b), label, "di-cover")


def main() -> None:
    s = cover_svg()
    write_cover(REPORT / "index.html", s)
    render_hero(s, REPORT, THUMB)
    print("cover written; hero.jpg and thumb rendered")


if __name__ == "__main__":
    main()
