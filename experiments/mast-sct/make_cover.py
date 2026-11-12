"""Draw the note's cover: every run's sct_score on the 174 open items as a horizontal bar from zero.

    python experiments/mast-sct/make_cover.py

The house benchmark cover (see ../shared/make_benchmark_cover.py, whose chips, type and colours this reuses): one row
per run, each bar from zero with its 95% bootstrap interval. DeepSeek-V4-Flash in thinking mode sits on the tinted row
in orange with its paired gap to GPT-5.5 beside the value; GPT-5.5 (MAST's own shipped run, re-scored here) is the
darker grey bar. Runs made for this note carry the steerlab chip. Numbers come from results/stats.json.
Writes report/figures/hero.jpg (1536 x 1024) and ../shared/thumbs/mast-sct.jpg (720 x 480).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))
import make_benchmark_cover as mbc

STATS = json.loads((HERE / "results" / "stats.json").read_text())
mbc.MARKS["meta"] = mbc._mark("simple-icons/meta.svg", "#0467DF")
if not mbc.STEERLAB_LOGO.exists():  # the logo ships in the public repo's docs/assets; point at a clone of it
    mbc.STEERLAB_LOGO = Path(os.environ.get("STEERLAB_LOGO", "../steerlab/docs/assets/logo.png")).resolve()
ROWS = [  # key in stats.json, logo, label, ours
    ("deepseek-v4-flash-0731-think", "deepseek", "DeepSeek-V4-Flash", True),
    ("gpt-5.5", "openai", "GPT-5.5 (MAST's run)", False),
    ("deepseek-v4-flash-0731-nothink", "deepseek", "V4-Flash, no thinking", True),
    ("llama-3.3-70b-instruct", "meta", "Llama-3.3-70B", True),
    ("qwen3-32b-nothink", "qwen", "Qwen3-32B", True),
]
WIN, RIVAL = "deepseek-v4-flash-0731-think", "gpt-5.5"


def cover() -> str:
    png = mbc.steerlab_png()
    x0, x1 = 344, 596  # sct_score 0 and 1

    def X(v: float) -> float:
        return x0 + (x1 - x0) * v

    big, top, bot = 74, 66, 426
    row = (bot - top - big) / (len(ROWS) - 1)
    out = [f'<rect x="20" y="{top}" width="680" height="{big}" rx="12" fill="{mbc.WIN_ROW}"/>']
    for g in (0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{X(g):.1f}" x2="{X(g):.1f}" y1="{top + big + 4}" y2="{bot - 2}" stroke="{mbc.RULE}"/>')
    gap = STATS["pairs"][f"{RIVAL} - {WIN}"]["diff"]
    y = float(top)
    for key, logo, name, ours in ROWS:
        s = STATS["runs"][key]["sct_score"]
        h = big if key == WIN else row
        cy = y + h / 2
        y += h
        if key == WIN:
            out.append(mbc.chip(logo, 50, cy, 36) + mbc.steer_chip(png, 90, cy, 36))
            out.append(mbc.text(120, cy - 4, name, 19, mbc.INK, 800))
            out.append(mbc.text(120, cy + 17, "thinking", 15, mbc.INK, 600))
        else:
            out.append(mbc.chip(logo, 50, cy, 28))
            if ours:
                out.append(mbc.steer_chip(png, 82, cy, 28))
            strong = key == RIVAL
            out.append(mbc.text(120, cy + 6, name, 16, mbc.INK if strong else mbc.SOFT, 600 if strong else 500))
        bh = 30 if key == WIN else 18
        fill = mbc.EMBER if key == WIN else mbc.RIVAL_BAR if key == RIVAL else mbc.GREY_BAR
        xe, yt = X(s["mean"]), cy - bh / 2
        out.append(
            f'<path d="M{x0},{yt:.1f} H{xe - 4:.1f} q4,0 4,4 V{yt + bh - 4:.1f} q0,4 -4,4 H{x0} Z" fill="{fill}"/>'
        )
        lo, hi = X(s["lo"]), X(s["hi"])
        whisker = f'stroke="{mbc.INK}" stroke-width="1.5"'
        out.append(f'<line x1="{lo:.1f}" x2="{hi:.1f}" y1="{cy:.1f}" y2="{cy:.1f}" {whisker}/>')
        for xx in (lo, hi):
            out.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{cy - 5:.1f}" y2="{cy + 5:.1f}" {whisker}/>')
        lx = hi + 10
        if key == WIN:
            out.append(mbc.text(lx, cy + 4, f"{s['mean']:.2f}", 30, mbc.EMBER, 800))
            call = f"≈ GPT-5.5 ({-gap['mean']:+.3f})".replace("-", "−")
            out.append(mbc.text(lx + 1, cy + 22, call, 12, mbc.INK, 600, mbc.MONO))
        else:
            colour = mbc.INK if key == RIVAL else mbc.SOFT
            out.append(mbc.text(lx, cy + 6, f"{s['mean']:.2f}", 17, colour, 600))
        print(f"cover: {key:<34} {s['mean']:.4f} [{s['lo']:.4f}, {s['hi']:.4f}]")
    out.append(f'<line x1="{x0}" x2="{x0}" y1="{top + 6}" y2="{bot - 2}" stroke="{mbc.INK}" stroke-width="1.2"/>')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{mbc.W}" height="{mbc.H}" viewBox="0 0 {mbc.W} {mbc.H}">'
        f'<rect width="{mbc.W}" height="{mbc.H}" fill="{mbc.CREAM}"/>{"".join(out)}</svg>'
    )


def main() -> None:
    from playwright.sync_api import sync_playwright

    page = (
        f"<!doctype html><html><head><meta charset='utf-8'>{mbc.FONTS}<style>html,body{{margin:0}}svg{{display:block}}"
        f"</style></head><body>{cover()}</body></html>"
    )
    with tempfile.TemporaryDirectory() as tmp:
        src, shot = Path(tmp) / "cover.html", Path(tmp) / "hero.png"
        src.write_text(page)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            ctx = browser.new_context(viewport={"width": mbc.W, "height": mbc.H}, device_scale_factor=1536 / mbc.W)
            tab = ctx.new_page()
            tab.goto(src.as_uri(), wait_until="networkidle")
            tab.evaluate("document.fonts.ready")
            tab.screenshot(path=str(shot), clip={"x": 0, "y": 0, "width": mbc.W, "height": mbc.H})
            browser.close()
        img = Image.open(shot).convert("RGB")
    hero = HERE / "report" / "figures" / "hero.jpg"
    img.save(hero, quality=88, optimize=True)
    thumb = HERE.parent / "shared" / "thumbs" / "mast-sct.jpg"
    img.resize((mbc.W, mbc.H), Image.Resampling.LANCZOS).save(thumb, quality=86, optimize=True)
    print(f"wrote {hero.relative_to(HERE.parent)} {img.size} and {thumb.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
