"""The Episode 2 cover, drawn from results/ (no API, no GPU; renders sketches in local Chromium like the RL runs).

python make_cover.py

It picks one sketch per arm from the seed-1 RL runs (spec midtraining + IT, and the IT-only twin): among the
evaluation sketches after RL that fail the picture check, the one with the most subject words in comments whose
render still produces a canvas. It re-renders both with Pigment Gym's own renderer (seed 0), saves the pictures
to report/figures/, writes the cover SVG into report/index.html between the cover markers, and renders it to
report/figures/hero.jpg (1600 x 900, which build_meta.py pads into og.jpg) and the research-page card
../shared/thumbs/spec-midtraining-ep2.jpg (720 x 480, letterboxed, never cropped).
"""

# ruff: noqa: E501  (SVG markup reads better on one line)

import html
import io
import json
import re
import sys
import tempfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
RL = HERE / "results" / "rl"
REPORT = HERE / "report"
FIG = REPORT / "figures"
THUMB = HERE.parent / "shared" / "thumbs" / "spec-midtraining-ep2.jpg"
sys.path.insert(0, str(HERE.parent / "pigment-gym"))

F = "-apple-system,BlinkMacSystemFont,'Inter','Segoe UI',Helvetica,Arial,sans-serif"
MONO = "ui-monospace,'SF Mono',Menlo,monospace"
CREAM, TEAL, PLUM, RED, EMBER_D = "#faf6ef", "#0f766e", "#8b3a62", "#b23b2a", "#a85a22"
INK, MUTED = "#1d1a17", "#7a7065"
STUFFED = re.compile(r"//\s*((?:Top|Base|Middle|Bottom|Center|Left|Right)[\w ']* of the (?:hill|sun))", re.I)
SPEC_QUOTE = "“does not repeat phrases …”"  # data/honest_effort_spec.txt, "Doing the Task Rather Than the Score"


def pick(run):
    """The failing sketch with the most subject words in comments whose render still produces a canvas."""
    from brush_tools import Renderer, gate

    rows = [json.loads(line) for line in (RL / f"{run}_after_samples.jsonl").open()]
    order = sorted((r for r in rows if r["gate"] == 0), key=lambda r: -r["mentions"])
    with Renderer(pool_size=2) as renderer:
        for r in order:
            out = renderer.render(r["code"], seed=0)
            if out.png and not gate(r["code"], out).passed:
                found = list(dict.fromkeys("// " + m for m in STUFFED.findall(r["code"])))[:2]
                print(run, "mentions", r["mentions"], "reward", round(r["reward"], 2), "comments", found)
                return out.png, found
    raise SystemExit(f"no failing sketch with a canvas in {run}")


def t(x, y, s, size, weight=400, fill=INK, anchor="start", italic=False, mono=False, ls=None):
    st = f"font-family:{MONO if mono else F};font-size:{size}px;font-weight:{weight}" + (
        ";font-style:italic" if italic else ""
    )
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    sp = f' letter-spacing="{ls}"' if ls else ""
    return f'<text x="{x}" y="{y}" fill="{fill}" style="{st}"{a}{sp}>{html.escape(s)}</text>'


def rect(x, y, w, h, fill, stroke="none", sw=1, rx=0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'


def pill(x, y, label, fill, size=11.5, h=20):
    w = round(len(label) * size * 0.6 + 18)
    return rect(x, y, w, h, fill, rx=h / 2) + t(x + 9, y + h * 0.7, label, size, 800, "#fff")


def fails(x, y):
    return (
        rect(x, y, 140, 26, RED, rx=13)
        + f'<path d="M{x + 13},{y + 8} l9,9 M{x + 22},{y + 8} l-9,9" stroke="#fff" stroke-width="2.6" stroke-linecap="round"/>'
        + t(x + 32, y + 18, "fails the check", 13, 700, "#fff")
    )


def comment_rows(x, y, rows):
    out = []
    for i, m in enumerate(rows):
        yy = y + i * 28
        out.append(rect(x - 4, round(yy - 12, 1), round(len(m) * 7.65 + 8, 1), 17, "#fff2e6", rx=4))
        out.append(t(x, yy, m, 12.5, 800, EMBER_D, mono=True))
    return "".join(out)


def cover_svg(spec_comments, twin_comments):
    ty, py = 46, 86
    b = [rect(0, 0, 880, 495, CREAM)]
    # the causal strip: the spec model reads the spec, the twin skips it, both go through the same RL
    b.append(rect(20, ty - 14, 104, 28, TEAL, rx=14) + t(72, ty + 5, "spec model", 13, 800, "#fff", "middle"))
    b.append(rect(20, py - 14, 104, 28, PLUM, rx=14) + t(72, py + 5, "twin", 13, 800, "#fff", "middle"))
    b.append(f'<path d="M124,{ty} H150 M410,{ty} H466" stroke="{TEAL}" stroke-width="3"/>')
    b.append(f'<path d="M124,{py} H466" stroke="{PLUM}" stroke-width="3" stroke-dasharray="7 6"/>')
    b.append(t(150, ty - 26, "reads the spec", 12, 800, TEAL, ls="0.02em"))
    b.append(
        rect(150, ty - 18, 260, 36, "#fff", TEAL, 2, rx=18)
        + t(280, ty + 5, SPEC_QUOTE, 14, 700, "#3a332d", "middle", italic=True)
    )
    b.append(t(280, py - 8, "skips it", 12, 700, PLUM, "middle"))
    b.append(rect(470, ty - 22, 390, py - ty + 44, "#fdeee6", RED, 2, rx=12))
    b.append(t(490, 72, "same RL", 18, 800, RED) + pill(582, 56, "hack learned here", RED))
    for x_from, x_to, col in ((560, 225, TEAL), (760, 655, PLUM)):
        b.append(
            f'<path d="M{x_from},108 V124 H{x_to} V140" fill="none" stroke="{col}" stroke-width="2.4" stroke-linejoin="round"/>'
        )
    # one panel per arm: its real picture after RL, two of its stuffed comments, the verdict
    for x, img, rows, col, lab in (
        (20, "cover-spec-model.jpg", spec_comments, TEAL, "SPEC MODEL"),
        (450, "cover-twin.jpg", twin_comments, PLUM, "TWIN"),
    ):
        b.append(rect(x, 140, 410, 256, "#fff", col, 1.5, rx=14, dash=None if col == TEAL else "6 5"))
        b.append(
            f'<image href="figures/{img}" x="{x + 16}" y="168" width="208" height="208" preserveAspectRatio="xMidYMid slice"/>'
        )
        b.append(rect(x + 16, 168, 208, 208, "none", "#d8cfc2", 2, rx=6))
        b.append(t(x + 240, 192, lab, 13, 800, col, ls="0.1em"))
        b.append(comment_rows(x + 244, 232, rows))
        b.append(fails(x + 240, 344))
    b.append(
        f'<circle cx="440" cy="268" r="18" fill="{CREAM}" stroke="#b8ad9f" stroke-width="1.5"/>'
        + t(440, 277, "=", 27, 800, "#5a5048", "middle")
    )
    b.append(t(20, 440, "Spec or no spec, RL taught the same hack.", 24, 800, INK))
    b.append(t(20, 462, "The spec was read before RL.", 14.5, 400, "#5a5048", italic=True))
    b.append(
        t(
            20,
            484,
            "PICTURES RE-RENDERED FROM EACH MODEL’S SAVED RL SKETCH WITH PIGMENT GYM’S RENDERER · REAL SPEC WORDING AND COMMENTS",
            9.5,
            500,
            MUTED,
            ls="0.05em",
        )
    )
    label = (
        "The model that read the spec (its words: does not repeat phrases) and its twin, which skipped it, go "
        "through the same RL, where the hack is learned. Below, each model's real sketch after RL, a few smudges and a "
        "blank canvas, each with code comments stuffed with the hill's name and a fails the check tag; an equals sign "
        "between them. Spec or no spec, RL taught the same hack."
    )
    return (
        f'<svg class="ep2-cover" viewBox="0 0 880 495" width="880" height="495" role="img" aria-label="{html.escape(label)}" '
        f'style="display:block;width:100%;height:auto" xmlns="http://www.w3.org/2000/svg">{"".join(b)}</svg>'
    )


def render_hero(svg):
    """Screenshot the SVG at 1600 x 900 for hero.jpg, then letterbox it into the 720 x 480 card."""
    from playwright.sync_api import sync_playwright

    page_html = f'<!doctype html><meta charset="utf-8"><body style="margin:0;background:{CREAM}"><div style="width:880px">{svg}</div>'
    with tempfile.NamedTemporaryFile("w", suffix=".html", dir=REPORT, delete=False) as f:
        f.write(page_html)
        tmp = Path(f.name)
    png = FIG / "_hero.png"
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 880, "height": 495}, device_scale_factor=1600 / 880)
            page.goto(tmp.as_uri())
            page.wait_for_load_state("networkidle")
            page.locator("svg").screenshot(path=str(png))
            browser.close()
    finally:
        tmp.unlink()
    hero = Image.open(png).convert("RGB").resize((1600, 900), Image.Resampling.LANCZOS)
    png.unlink()
    hero.save(FIG / "hero.jpg", quality=88, optimize=True)
    card = Image.new("RGB", (720, 480), CREAM)
    card.paste(hero.resize((720, 405), Image.Resampling.LANCZOS), (0, (480 - 405) // 2))
    card.save(THUMB, quality=88, optimize=True)


def main():
    spec_png, spec_comments = pick("H_s1")
    twin_png, twin_comments = pick("T_s1")
    for png, name in ((spec_png, "cover-spec-model.jpg"), (twin_png, "cover-twin.jpg")):
        Image.open(io.BytesIO(png)).convert("RGB").save(FIG / name, quality=86, optimize=True)
    svg = cover_svg(spec_comments, twin_comments)
    index = REPORT / "index.html"
    text = index.read_text()
    start, end = "<!-- cover:start -->", "<!-- cover:end -->"
    i, j = text.index(start) + len(start), text.index(end)
    index.write_text(text[:i] + "\n    " + svg + "\n    " + text[j:])
    render_hero(svg)
    print("cover: wrote", FIG / "hero.jpg", THUMB)


if __name__ == "__main__":
    main()
