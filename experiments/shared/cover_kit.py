"""Drawing helpers for the static SVG covers built by each note's make_cover.py.

A cover is an 880 x 495 inline SVG on the site's cream ground. `write_cover` puts it into a note's
report/index.html between `<!-- cover:start -->` and `<!-- cover:end -->`; `render_hero` screenshots it to
report/figures/hero.jpg (1600 x 900, which build_meta.py crops into og.jpg) and letterboxes it into the
720 x 480 research-page card in shared/thumbs/, never cropped.
"""

# ruff: noqa: E501  (SVG markup reads better on one line)

from __future__ import annotations

import html
import math
import tempfile
import textwrap
from pathlib import Path

F = "-apple-system,BlinkMacSystemFont,'Inter','Segoe UI',Helvetica,Arial,sans-serif"
MONO = "ui-monospace,'SF Mono',Menlo,monospace"
CREAM, EMBER, EMBER_D, EMBER_L = "#faf6ef", "#d26f2a", "#a85a22", "#f6dcc8"
TEAL, TEAL_L, PLUM, PLUM_L = "#0f766e", "#eaf6f4", "#8b3a62", "#f8eef3"
INK, INK2, MUTED, LINE = "#1d1a17", "#5a5048", "#7a7065", "#e3dbd0"
SLATE, SLATE_L, SLATE_M = "#3f5a73", "#eef2f6", "#c9d4df"
RED, GOOD = "#b23b2a", "#2f7d4f"
W, H = 880, 495


def t(x, y, s, size=13, weight=400, fill=INK, anchor="start", italic=False, ls=None, mono=False):
    st = f"font-family:{MONO if mono else F};font-size:{size}px;font-weight:{weight}"
    if italic:
        st += ";font-style:italic"
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    sp = f' letter-spacing="{ls}"' if ls else ""
    return f'<text x="{x}" y="{y}" fill="{fill}" style="{st}"{a}{sp}>{html.escape(s)}</text>'


def rect(x, y, w, h, fill, stroke="none", sw=1, rx=0):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def lines(x, y, s, n, size, fill, weight=400, italic=False, lh=None):
    """Text wrapped at n characters, one <text> per line."""
    lh = lh or size * 1.32
    return "".join(t(x, y + i * lh, ln, size, weight, fill, italic=italic) for i, ln in enumerate(textwrap.wrap(s, n)))


def arrow(x1, y1, x2, y2, color, sw=2.4):
    a = math.atan2(y2 - y1, x2 - x1)
    hx, hy = x2 - 9 * math.cos(a), y2 - 9 * math.sin(a)
    px, py = 5 * math.sin(a), -5 * math.cos(a)
    return (
        f'<path d="M{x1},{y1} L{hx:.1f},{hy:.1f}" stroke="{color}" stroke-width="{sw}" stroke-linecap="round"/>'
        f'<path d="M{hx + px:.1f},{hy + py:.1f} L{x2},{y2} L{hx - px:.1f},{hy - py:.1f}" fill="none" stroke="{color}" '
        f'stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"/>'
    )


def eq_badge(cx, cy, r=20):
    """No detectable difference."""
    k = r / 20
    return (
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#fff" stroke="#b8ad9f" stroke-width="2"/>'
        f'<path d="M{cx - 8 * k},{cy - 4.5 * k} H{cx + 8 * k} M{cx - 8 * k},{cy + 4.5 * k} H{cx + 8 * k}" stroke="{INK2}" stroke-width="{3 * k}" stroke-linecap="round"/>'
    )


def check_mark(cx, cy, r=16, fill=TEAL):
    k = r / 16
    return (
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}"/>'
        f'<path d="M{cx - 7 * k},{cy + 0.5 * k} L{cx - 2 * k},{cy + 5.5 * k} L{cx + 7.5 * k},{cy - 5 * k}" '
        f'fill="none" stroke="#fff" stroke-width="{3.2 * k}" stroke-linecap="round" stroke-linejoin="round"/>'
    )


def power_off(cx, cy, r=16, col=INK):
    """A power-button glyph: 'switched off'."""
    return (
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#fff" stroke="{col}" stroke-width="2.2"/>'
        f'<path d="M{cx - r * 0.42:.1f},{cy - r * 0.36:.1f} A{r * 0.55:.1f},{r * 0.55:.1f} 0 1 0 {cx + r * 0.42:.1f},{cy - r * 0.36:.1f}" '
        f'fill="none" stroke="{col}" stroke-width="2.4" stroke-linecap="round"/>'
        f'<path d="M{cx},{cy - r * 0.62:.1f} V{cy - r * 0.05:.1f}" stroke="{col}" stroke-width="2.4" stroke-linecap="round"/>'
    )


def letter_chip(cx, cy, letter, ok, r=20, word=None):
    """An answer letter in a green (right) or red (wrong) disc, with the option's text beside it."""
    col = GOOD if ok else RED
    out = f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{col}"/>' + t(
        cx, cy + r * 0.38, letter, r * 1.05, 800, "#fff", "middle"
    )
    if word:
        out += t(cx + r + 9, cy + r * 0.3, word, max(13, r * 0.75), 700, col)
    return out


def caps(s, y=484):
    return t(20, y, s, 9.5, 500, MUTED, ls="0.05em")


def kicker(series, right=None, y=24):
    out = t(20, y, series, 11.5, 800, EMBER_D, ls="0.09em")
    if right:
        out += t(860, y, right, 11.5, 600, MUTED, "end", ls="0.06em")
    return out


def headline(line, sub=None, y=420, size=27):
    out = t(20, y, line, size, 800, INK)
    if sub:
        out += t(20, y + 26, sub, 15, 400, INK2, italic=True)
    return out


def _dot(cx, cy, r, on):
    if on:
        return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{EMBER}" stroke="#fff" stroke-width="2"/>'
    k = r * 0.55
    return (
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="#fff" stroke="#9a8f82" stroke-width="2"/>'
        f'<path d="M{cx - k:.1f},{cy - k:.1f} l{2 * k:.1f},{2 * k:.1f} M{cx + k:.1f},{cy - k:.1f} l{-2 * k:.1f},{2 * k:.1f}" stroke="#9a8f82" stroke-width="1.8" stroke-linecap="round"/>'
    )


def layer_column(x, y_top, y_bot, w, n_layers, feats, edges, on=True, show_scale=True):
    """The network as a column of layers, layer 0 at the bottom. Layers below the first selected feature are
    squeezed into a short stub under a break mark so the selected layers get room. `feats` are (layer, index)
    pairs drawn as dots on their real layer; `edges` are pairs of feats. Horizontal position is only spacing."""
    lo = min(f[0] for f in feats) - 1
    stub = 22
    y_brk = y_bot - stub
    step = (y_brk - 10 - y_top) / (n_layers - 1 - lo)

    def ly(lay):
        return y_brk - 10 - (lay - lo) * step

    out = [rect(x, y_top - 10, w, y_bot - y_top + 18, "#fff", SLATE_M, 1.5, rx=10)]
    for k in range(lo, n_layers):
        out.append(
            f'<path d="M{x + 8},{ly(k):.1f} H{x + w - 8}" stroke="{SLATE_M if k % 5 == 0 else SLATE_L}" stroke-width="1"/>'
        )
    for yy in (y_bot - 4, y_bot - 9, y_bot - 14):
        out.append(f'<path d="M{x + 8},{yy} H{x + w - 8}" stroke="{SLATE_L}" stroke-width="1"/>')
    zz = " ".join(f"L{x + 4 + i * (w - 8) / 10:.1f},{y_brk - 2 + (4 if i % 2 else -2)}" for i in range(11))
    out.append(f'<path d="M{x + 4},{y_brk - 2} {zz}" fill="none" stroke="{SLATE_M}" stroke-width="1.5"/>')
    if show_scale:
        out.append(t(x - 6, y_bot, f"layers 0-{lo - 1}", 11, 600, MUTED, "end"))
        out.append(t(x - 6, ly(lo) + 4, f"layer {lo}", 11, 600, MUTED, "end"))
        out.append(t(x - 6, ly(n_layers - 1) + 4, f"layer {n_layers - 1}", 11, 600, MUTED, "end"))
    pos: dict = {}
    by_layer: dict = {}
    for f in feats:
        by_layer.setdefault(f[0], []).append(f)
    sp = min(28, (w - 24) / 4)
    for lay, fs in by_layer.items():
        for j, f in enumerate(fs):
            cx = x + w / 2 + (j - (len(fs) - 1) / 2) * sp * 1.6 + (sp * 0.5 if lay % 2 else -sp * 0.5)
            pos[f] = (cx, ly(lay))
    ec = EMBER if on else "#b8ad9f"
    dash = "" if on else ' stroke-dasharray="4 4"'
    for u, v in edges:
        (x1, y1), (x2, y2) = pos[u], pos[v]
        out.append(
            f'<path d="M{x1:.1f},{y1:.1f} L{x2:.1f},{y2:.1f}" stroke="{ec}" stroke-width="2" opacity="0.85"{dash}/>'
        )
    r = 7 if w >= 100 else 6
    out.extend(_dot(cx, cy, r, on) for cx, cy in pos.values())
    return "".join(out)


def svg(body, label, cls):
    return (
        f'<svg class="{cls}" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="{html.escape(label)}" '
        f'style="display:block;width:100%;height:auto" xmlns="http://www.w3.org/2000/svg">{rect(0, 0, W, H, CREAM)}{body}</svg>'
    )


def write_cover(index: Path, svg_text: str) -> None:
    text = index.read_text()
    start, end = "<!-- cover:start -->", "<!-- cover:end -->"
    i, j = text.index(start) + len(start), text.index(end)
    index.write_text(text[:i] + "\n    " + svg_text + "\n    " + text[j:])


def render_hero(svg_text: str, report: Path, thumb: Path) -> None:
    """Screenshot the SVG at 1600 x 900 into report/figures/hero.jpg, then letterbox it into the 720 x 480 card."""
    from PIL import Image
    from playwright.sync_api import sync_playwright

    fig = report / "figures"
    page = f'<!doctype html><meta charset="utf-8"><body style="margin:0;background:{CREAM}"><div style="width:880px">{svg_text}</div>'
    with tempfile.NamedTemporaryFile("w", suffix=".html", dir=report, delete=False) as f:
        f.write(page)
        tmp = Path(f.name)
    png = fig / "_hero.png"
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            pg = browser.new_page(viewport={"width": 880, "height": 495}, device_scale_factor=1600 / 880)
            pg.goto(tmp.as_uri())
            pg.locator("svg").screenshot(path=str(png))
            browser.close()
    finally:
        tmp.unlink()
    hero = Image.open(png).convert("RGB").resize((1600, 900), Image.Resampling.LANCZOS)
    png.unlink()
    hero.save(fig / "hero.jpg", quality=88, optimize=True)
    card = Image.new("RGB", (720, 480), CREAM)
    card.paste(hero.resize((720, 405), Image.Resampling.LANCZOS), (0, (480 - 405) // 2))
    card.save(thumb, quality=88, optimize=True)
