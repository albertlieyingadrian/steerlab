"""Draw the note's cover: the three drivers' Cua Driver Bench scores as bars from zero, ours marked.

    python experiments/cua-driver/make_cover.py

The same benchmark style as the FinanceMath covers (../shared/make_benchmark_cover.py, whose helpers it reuses):
one row per driver, each bar the share of the 90 trials passed with Claude Opus 5.5, with its 95% Clopper-Pearson
interval. Our driver sits on a tinted row in orange with the steerlab chip, and carries the one measure it wins,
cost per pass. cua-driver, the same-day control, is the darker grey bar. The numbers are results/summary.json's
and Scale's published the target run, typed in below. Rendered by headless Chromium (Playwright). Writes
report/figures/hero.jpg (1536 x 1024) and ../shared/thumbs/cua-driver.jpg (720 x 480); build_meta.py crops og.jpg
from the hero, and every row sits inside that crop.
"""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / "shared"
_spec = importlib.util.spec_from_file_location("bench_cover", SHARED / "make_benchmark_cover.py")
assert _spec is not None and _spec.loader is not None
bc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bc)
W, H = bc.W, bc.H

# Passes of 90 and the 95% Clopper-Pearson interval (%), best first.
ROWS = [
    dict(key="target", name="Target (closed)", sub="Scale’s run, Nov 11", k=88, ci=(92.2, 99.7)),
    dict(key="cua", name="cua-driver 0.47.4", sub="open · same day, same trials", k=67, ci=(64.2, 83.1)),
    dict(key="ours", name="ourdriver (this note)", sub="open · built from the target’s ideas", k=49, ci=(43.6, 65.0)),
]


def cover() -> str:
    png = bc.steerlab_png()
    x0, x1 = 300, 560  # 0% and 100%; the room to the right holds the counts

    def X(v: float) -> float:
        return x0 + (x1 - x0) * v / 100

    top, row = 116, 94
    y_lo, y_hi = top + 8, top + 3 * row - 6  # the plot's vertical extent
    out = [
        bc.text(28, 72, "Cua Driver Bench · macOS · Claude Opus 5.5", 15, bc.SOFT, 600, bc.MONO),
        bc.text(28, 96, "Trials passed of 90, Scale’s exact setup for the target", 15, bc.SOFT, 600, bc.MONO),
    ]
    ours_y = top + 2 * row
    out.append(f'<rect x="16" y="{ours_y + 4}" width="688" height="{row - 8}" rx="12" fill="{bc.WIN_ROW}"/>')
    for g in (25, 50, 75, 100):
        out.append(f'<line x1="{X(g):.1f}" x2="{X(g):.1f}" y1="{y_lo}" y2="{y_hi}" stroke="{bc.RULE}"/>')
    for i, r in enumerate(ROWS):
        cy = top + i * row + row / 2
        ours = r["key"] == "ours"
        nx = 28
        if ours:
            out.append(bc.steer_chip(png, 46, cy, 34))
            nx = 72
        out.append(bc.text(nx, cy - 3, r["name"], 20 if ours else 19, bc.INK, 800 if ours else 700))
        out.append(bc.text(nx, cy + 19, r["sub"], 15, bc.INK if ours else bc.SOFT, 500))
        v = 100 * r["k"] / 90
        bh = 30
        fill = bc.EMBER if ours else bc.RIVAL_BAR if r["key"] == "cua" else bc.GREY_BAR
        xe, yt = X(v), cy - bh / 2
        out.append(
            f'<path d="M{x0},{yt:.1f} H{xe - 4:.1f} q4,0 4,4 V{yt + bh - 4:.1f} q0,4 -4,4 H{x0} Z" fill="{fill}"/>'
        )
        lo, hi = X(r["ci"][0]), X(r["ci"][1])
        whisker = f'stroke="{bc.INK}" stroke-width="1.6"'
        out.append(f'<line x1="{lo:.1f}" x2="{hi:.1f}" y1="{cy:.1f}" y2="{cy:.1f}" {whisker}/>')
        for xx in (lo, hi):
            out.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{cy - 6:.1f}" y2="{cy + 6:.1f}" {whisker}/>')
        lx = max(hi, xe) + 12
        colour = bc.EMBER if ours else bc.INK
        out.append(bc.text(lx, cy + 10, f"{r['k']}/90", 30, colour, 800))
        if ours:
            out.append(bc.text(lx + 1, cy + 31, "$0.64 vs $1.14 per pass", 14, bc.INK, 600, bc.MONO))
    out.append(f'<line x1="{x0}" x2="{x0}" y1="{y_lo}" y2="{y_hi}" stroke="{bc.INK}" stroke-width="1.2"/>')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
        f'<rect width="{W}" height="{H}" fill="{bc.CREAM}"/>{"".join(out)}</svg>'
    )


def main() -> None:
    from playwright.sync_api import sync_playwright

    page = (
        f"<!doctype html><html><head><meta charset='utf-8'>{bc.FONTS}<style>html,body{{margin:0}}"
        f"svg{{display:block}}</style></head><body>{cover()}</body></html>"
    )
    with tempfile.TemporaryDirectory() as tmp:
        src, shot = Path(tmp) / "cover.html", Path(tmp) / "hero.png"
        src.write_text(page)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            ctx = browser.new_context(viewport={"width": W, "height": H}, device_scale_factor=1536 / W)
            tab = ctx.new_page()
            tab.goto(src.as_uri(), wait_until="networkidle")
            tab.evaluate("document.fonts.ready")
            tab.screenshot(path=str(shot), clip={"x": 0, "y": 0, "width": W, "height": H})
            browser.close()
        img = Image.open(shot).convert("RGB")
    img.save(HERE / "report" / "figures" / "hero.jpg", quality=88, optimize=True)
    img.resize((W, H), Image.Resampling.LANCZOS).save(SHARED / "thumbs" / "cua-driver.jpg", quality=86, optimize=True)
    print("wrote report/figures/hero.jpg", img.size, "and ../shared/thumbs/cua-driver.jpg")


if __name__ == "__main__":
    main()
