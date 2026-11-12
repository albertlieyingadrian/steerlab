"""Draw a FinanceMath note's cover: every system's test accuracy as a horizontal bar, ours marked with steerlab's logo.

    python experiments/shared/make_benchmark_cover.py fin-math       # episode 2
    python experiments/shared/make_benchmark_cover.py fin-math-v2

One row per system on the 1,000 FinanceMath test questions, each bar from zero, with the 95% bootstrap interval where
the note reports one. The note's own result sits on a tinted row in orange, with its paired gap to gpt-5.4 beside the
value; gpt-5.4 is the darker grey bar. A system this project ran carries the steerlab chip next to its model's logo.
Only systems scored on all 1,000 test questions appear. The numbers are the notes' results tables, typed in below.

The cover is an SVG rendered by headless Chromium (Playwright: `pip install playwright` then
`playwright install chromium`), so its type is the site's own Bricolage Grotesque and JetBrains Mono. Writes
`<note>/report/figures/hero.jpg` (1536 x 1024) and `shared/thumbs/<note>.jpg` (720 x 480); `build_meta.py` crops the
1200 x 630 `og.jpg` from the hero, and every row sits inside that crop.
"""

from __future__ import annotations

import argparse
import base64
import io
import re
import tempfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
LOGOS = HERE / "logos"
STEERLAB_LOGO = EXPERIMENTS.parent / "docs" / "assets" / "logo.png"
W, H = 720, 480

CREAM, INK, EMBER, RULE, SOFT = "#F3EDE3", "#2B2723", "#C9561C", "#D9D0C2", "#6E655C"
GREY_BAR, RIVAL_BAR, WIN_ROW = "#D3CCC1", "#8F867B", "#F8E2D4"
DISPLAY = "'Bricolage Grotesque', 'Avenir Next', sans-serif"
MONO = "'JetBrains Mono', ui-monospace, monospace"
FONTS = (
    '<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,600;'
    '12..96,800&family=JetBrains+Mono:wght@600&display=block" rel="stylesheet">'
)

# Test accuracy (%) and 95% bootstrap interval where the note reports one.
SYSTEMS: dict[str, dict] = {
    "think": dict(logo="deepseek", name="DeepSeek-V4-Flash", v=88.5, ci=(86.5, 90.4)),
    "plain": dict(logo="deepseek", name="V4-Flash + harness, plain", v=82.4, ci=(80.0, 84.7)),
    "gpt54": dict(logo="openai", name="gpt-5.4", v=81.8, ci=(79.4, 84.1)),
    "mini": dict(logo="openai", name="gpt-5.4-mini", v=73.0, ci=(70.2, 75.7)),
    "qpost": dict(logo="qwen", name="Qwen3-8B post-trained", v=62.2, ci=None),
    "gpt4o": dict(logo="openai", name="GPT-4o (paper)", v=60.9, ci=None),
    "qbase": dict(logo="qwen", name="Qwen3-8B base", v=55.4, ci=None),
}
OURS = {"think", "plain", "qpost"}  # systems this project ran: the harness runs and the post-trained 8B
RIVAL = "gpt54"
NOTES: dict[str, dict] = {
    "fin-math": dict(
        win="plain", line2="+ steerlab harness", call="≈ gpt-5.4",
        rows=["plain", "gpt54", "mini", "qpost", "gpt4o", "qbase"],
    ),
    "fin-math-v2": dict(
        win="think", line2="+ steerlab harness, thinking", call="+6.7 vs gpt-5.4",
        rows=["think", "plain", "gpt54", "mini", "qpost", "gpt4o", "qbase"],
    ),
}  # fmt: skip


def _mark(file: str, colour: str) -> tuple[str, str, str]:
    svg = (LOGOS / file).read_text()
    match = re.search(r'<path[^>]* d="([^"]+)"', svg)
    if match is None:
        raise ValueError(f"no path in {file}")
    return match.group(1), "evenodd" if 'fill-rule="evenodd"' in svg else "nonzero", colour


MARKS = {
    "openai": _mark("openai.svg", "#111111"),
    "deepseek": _mark("simple-icons/deepseek.svg", "#5786FE"),
    "qwen": _mark("simple-icons/qwen.svg", "#6950EF"),
}


def text(x: float, y: float, s: str, size: float, fill: str, weight: int, family: str = DISPLAY) -> str:
    s = s.replace("&", "&amp;")
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}">{s}</text>'
    )


def chip(key: str, cx: float, cy: float, s: float) -> str:
    """A white rounded tile with the model's mark centred in it."""
    d, rule, colour = MARKS[key]
    pad, x, y = s * 0.2, cx - s / 2, cy - s / 2
    stroke = 'stroke="#E4DDD2"'
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{s:.1f}" height="{s:.1f}" rx="{s * 0.24:.1f}" fill="#FFFFFF" {stroke}/>'
        f'<svg x="{x + pad:.1f}" y="{y + pad:.1f}" width="{s - 2 * pad:.1f}" height="{s - 2 * pad:.1f}" '
        f'viewBox="0 0 24 24"><path d="{d}" fill="{colour}" fill-rule="{rule}"/></svg>'
    )


def steerlab_png() -> str:
    """steerlab's logo cropped to its dark field, so the mark fills most of a small tile, as base64 PNG."""
    img = Image.open(STEERLAB_LOGO).convert("RGBA")
    side = img.width
    crop = img.crop((side * 3 // 16, side * 3 // 16, side * 13 // 16, side * 13 // 16))
    buf = io.BytesIO()
    crop.resize((128, 128), Image.Resampling.LANCZOS).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def steer_chip(png: str, cx: float, cy: float, s: float) -> str:
    cid, x, y = f"sc{int(cx)}x{int(cy)}", cx - s / 2, cy - s / 2
    return (
        f'<clipPath id="{cid}"><rect x="{x:.1f}" y="{y:.1f}" width="{s:.1f}" height="{s:.1f}" rx="{s * 0.24:.1f}"/>'
        f'</clipPath><image href="data:image/png;base64,{png}" x="{x:.1f}" y="{y:.1f}" width="{s:.1f}" '
        f'height="{s:.1f}" clip-path="url(#{cid})"/>'
    )


def cover(note: str) -> str:
    spec = NOTES[note]
    win, rows = spec["win"], spec["rows"]
    png = steerlab_png()
    x0, x1 = 344, 596  # 0% and 100%; the room to the right holds the winner's value and gap

    def X(v: float) -> float:
        return x0 + (x1 - x0) * v / 100

    big, top, bot = 74, 66, 426  # the winner's row is taller; the rest share what is left
    row = (bot - top - big) / (len(rows) - 1)
    out = [f'<rect x="20" y="{top}" width="680" height="{big}" rx="12" fill="{WIN_ROW}"/>']
    for g in (25, 50, 75, 100):
        out.append(f'<line x1="{X(g):.1f}" x2="{X(g):.1f}" y1="{top + big + 4}" y2="{bot - 2}" stroke="{RULE}"/>')
    y = float(top)
    for k in rows:
        s = SYSTEMS[k]
        h = big if k == win else row
        cy = y + h / 2
        y += h
        if k == win:
            out.append(chip(s["logo"], 50, cy, 36) + steer_chip(png, 90, cy, 36))
            out.append(text(120, cy - 4, s["name"] if k == "think" else "DeepSeek-V4-Flash", 19, INK, 800))
            out.append(text(120, cy + 17, spec["line2"], 15, INK, 600))
        else:
            out.append(chip(s["logo"], 50, cy, 28))
            if k in OURS:
                out.append(steer_chip(png, 82, cy, 28))
            strong = k == RIVAL or k in OURS
            out.append(text(120, cy + 6, s["name"], 16, INK if strong else SOFT, 600 if k == RIVAL else 500))
        bh = 30 if k == win else 18
        fill = EMBER if k == win else RIVAL_BAR if k == RIVAL else GREY_BAR
        xe, yt = X(s["v"]), cy - bh / 2
        out.append(
            f'<path d="M{x0},{yt:.1f} H{xe - 4:.1f} q4,0 4,4 V{yt + bh - 4:.1f} q0,4 -4,4 H{x0} Z" fill="{fill}"/>'
        )
        if s["ci"]:
            lo, hi = X(s["ci"][0]), X(s["ci"][1])
            whisker = f'stroke="{INK}" stroke-width="1.5"'
            out.append(f'<line x1="{lo:.1f}" x2="{hi:.1f}" y1="{cy:.1f}" y2="{cy:.1f}" {whisker}/>')
            for xx in (lo, hi):
                out.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{cy - 5:.1f}" y2="{cy + 5:.1f}" {whisker}/>')
        lx = (X(s["ci"][1]) if s["ci"] else xe) + 10
        if k == win:
            out.append(text(lx, cy + 4, f"{s['v']:.1f}", 30, EMBER, 800))
            out.append(text(lx + 1, cy + 22, spec["call"], 12, INK, 600, MONO))
        else:
            out.append(text(lx, cy + 6, f"{s['v']:.1f}", 17, INK if k == RIVAL or k in OURS else SOFT, 600))
    out.append(f'<line x1="{x0}" x2="{x0}" y1="{top + 6}" y2="{bot - 2}" stroke="{INK}" stroke-width="1.2"/>')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
        f'<rect width="{W}" height="{H}" fill="{CREAM}"/>{"".join(out)}</svg>'
    )


def main(note: str) -> None:
    from playwright.sync_api import sync_playwright

    page = (
        f"<!doctype html><html><head><meta charset='utf-8'>{FONTS}<style>html,body{{margin:0}}svg{{display:block}}"
        f"</style></head><body>{cover(note)}</body></html>"
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
    hero = EXPERIMENTS / note / "report" / "figures" / "hero.jpg"
    img.save(hero, quality=88, optimize=True)
    thumb = HERE / "thumbs" / f"{note}.jpg"
    img.resize((W, H), Image.Resampling.LANCZOS).save(thumb, quality=86, optimize=True)
    print(f"{note}: wrote {hero.relative_to(EXPERIMENTS)} {img.size} and {thumb.relative_to(EXPERIMENTS)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("note", choices=sorted(NOTES))
    main(parser.parse_args().note)
