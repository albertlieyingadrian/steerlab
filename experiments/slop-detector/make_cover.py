"""Draw the note's cover: EditLens test macro-F1 as vertical bars, ours marked with steerlab's logo.

    python make_cover.py

Four bars on a zoomed axis (the break is marked): our 4B + 1.7B blend and voi-slop-detect-4B, both with cut-offs fitted
on EditLens's validation split, with paired 95% bootstrap intervals; voi with its shipped cut-offs; and EditLens's 24B
cosine model, its card's number, drawn hollow because it is not paired with ours. Uses the FinanceMath covers' palette,
type and logo chips (experiments/shared/make_benchmark_cover.py; needs Playwright with Chromium). Writes
report/figures/hero.jpg (1536 x 1024) and ../shared/thumbs/slop-detector.jpg (720 x 480); build_meta.py crops og.jpg.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))
import make_benchmark_cover as mbc  # noqa: E402

RES = HERE / "results" / "eval"
W, H = mbc.W, mbc.H
CREAM, INK, EMBER, RULE, SOFT, MONO = mbc.CREAM, mbc.INK, mbc.EMBER, mbc.RULE, mbc.SOFT, mbc.MONO
T = mbc.text


def numbers() -> tuple[list, str]:
    ctl = json.loads((RES / "voi_editlens_val.json").read_text())["splits"]["test"]
    fixed = json.loads((RES / "v2-4b.json").read_text())["splits"]["test"]["paired"]["macro_f1"]
    ours, voi_cal = ctl["ours_calibrated_vs_voi_calibrated"]["macro_f1"], ctl["voi_calibrated_vs_voi_fixed"]["macro_f1"]
    systems = [
        (("Ours: 4B + 1.7B", "calibrated"), ours["a"], ours["a_ci"], "ours"),
        (("voi-slop-detect-4B", "calibrated"), voi_cal["a"], voi_cal["a_ci"], "rival"),
        (("voi-slop-detect-4B", "as shipped"), fixed["b"], fixed["b_ci"], "grey"),
        (("EditLens 24B", "published, unpaired"), 0.904, None, "hollow"),
    ]
    gap = f"ours +{ours['diff']:.3f} [{ours['diff_ci'][0]:+.3f}, {ours['diff_ci'][1]:+.3f}] vs voi, both calibrated"
    return systems, gap


def chips(kind: str, cx: float, cy: float, png: str, s: float = 28) -> str:
    if kind == "hollow":
        x, y = cx - s / 2, cy - s / 2
        return (
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{s}" height="{s}" rx="{s * 0.24:.1f}" fill="#FFFFFF" '
            f'stroke="#E4DDD2"/>' + T(cx - s * 0.3, cy + s * 0.15, "EL", s * 0.4, SOFT, 800, MONO)
        )
    if kind == "ours":
        return mbc.chip("qwen", cx - s * 0.6, cy, s) + mbc.steer_chip(png, cx + s * 0.6, cy, s)
    return mbc.chip("qwen", cx, cy, s)


def cover() -> str:
    systems, gap = numbers()
    png = mbc.steerlab_png()
    lo, hi, x0, x1, y0, y1 = 0.85, 0.925, 86, 700, 322, 118

    def Y(v: float) -> float:
        return y0 - (y0 - y1) * (v - lo) / (hi - lo)

    out = [T(36, 68, "EDITLENS TEST · 3-WAY MACRO-F1 · 6,115 TEXTS", 13, SOFT, 600, MONO),
           T(36, 88, gap, 13, EMBER, 800, MONO)]  # fmt: skip
    for t in (0.86, 0.88, 0.90, 0.92):
        out.append(f'<line x1="{x0}" x2="{x1}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="{RULE}"/>')
        out.append(T(x0 - 44, Y(t) + 4, f"{t:.2f}".rstrip("0"), 12, SOFT, 600, MONO))
    slot = (x1 - x0) / len(systems)
    for i, (name, v, ci, kind) in enumerate(systems):
        cx, bw = x0 + slot * (i + 0.5), 92 if kind == "ours" else 74
        fill = {"ours": EMBER, "rival": mbc.RIVAL_BAR, "grey": mbc.GREY_BAR, "hollow": "none"}[kind]
        dash = f'stroke="{mbc.RIVAL_BAR}" stroke-width="2" stroke-dasharray="6 4"' if kind == "hollow" else ""
        x, yt = cx - bw / 2, Y(v)
        out.append(f'<path d="M{x:.1f},{y0} V{yt + 5:.1f} q0,-5 5,-5 H{x + bw - 5:.1f} q5,0 5,5 V{y0} Z" '
                   f'fill="{fill}" {dash}/>')  # fmt: skip
        top = yt
        if ci:
            s = f'stroke="{INK}" stroke-width="1.6"'
            a, b = Y(ci[0]), Y(ci[1])
            out.append(f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{a:.1f}" y2="{b:.1f}" {s}/>')
            for yy in (a, b):
                out.append(f'<line x1="{cx - 6:.1f}" x2="{cx + 6:.1f}" y1="{yy:.1f}" y2="{yy:.1f}" {s}/>')
            top = b
        size, colour, weight = (30, EMBER, 800) if kind == "ours" else (18, SOFT if kind == "hollow" else INK, 600)
        out.append(T(cx - size * 1.2, top - 10, f"{v:.3f}", size, colour, weight))
        out.append(chips(kind, cx, y0 + 24, png))
        out.append(T(cx - len(name[0]) * 3.9, y0 + 58, name[0], 14, SOFT if kind == "hollow" else INK,
                     800 if kind == "ours" else 600))  # fmt: skip
        out.append(T(cx - len(name[1]) * 3.4, y0 + 75, name[1], 12, SOFT, 600, MONO))
    out.append(f'<line x1="{x0}" x2="{x0}" y1="{y1 - 12}" y2="{y0}" stroke="{INK}" stroke-width="1.2"/>')
    out.append(f'<line x1="{x0}" x2="{x1}" y1="{y0}" y2="{y0}" stroke="{INK}" stroke-width="1.2"/>')
    out.append(f'<path d="M{x0 - 8},{y0 - 10} l5,-6 l6,10 l5,-6" fill="none" stroke="{INK}" stroke-width="1.4"/>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
            f'<rect width="{W}" height="{H}" fill="{CREAM}"/>{"".join(out)}</svg>')  # fmt: skip


def main() -> None:
    from playwright.sync_api import sync_playwright

    page = (f"<!doctype html><html><head><meta charset='utf-8'>{mbc.FONTS}<style>html,body{{margin:0}}"
            f"svg{{display:block}}</style></head><body>{cover()}</body></html>")  # fmt: skip
    with tempfile.TemporaryDirectory() as tmp:
        src, shot = Path(tmp) / "c.html", Path(tmp) / "c.png"
        src.write_text(page)
        with sync_playwright() as p:
            b = p.chromium.launch()
            tab = b.new_context(viewport={"width": W, "height": H}, device_scale_factor=1536 / W).new_page()
            tab.goto(src.as_uri(), wait_until="networkidle")
            tab.evaluate("document.fonts.ready")
            tab.screenshot(path=str(shot), clip={"x": 0, "y": 0, "width": W, "height": H})
            b.close()
        img = Image.open(shot).convert("RGB")
    img.save(HERE / "report" / "figures" / "hero.jpg", quality=88, optimize=True)
    img.resize((W, H), Image.Resampling.LANCZOS).save(HERE.parent / "shared" / "thumbs" / "slop-detector.jpg",
                                                      quality=86, optimize=True)  # fmt: skip
    print("wrote report/figures/hero.jpg", img.size, "and shared/thumbs/slop-detector.jpg")


if __name__ == "__main__":
    main()
