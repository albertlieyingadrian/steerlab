"""Draw the cover from the data: one author, three openings, and the voice scale.

    python make_cover.py      (after export_pages.py)

Left: how Author A (a held-out Enron author) opens an email, how Qwen3-8B opens the same synthetic brief with no
help, and how it opens with the author's voice vector added (report/data.js, from the scored drafts). Right: the
voice scale from a different human (0) to the same human (1), with the mean strict position of a few contestants
(results/leaderboard.json). Writes the SVG into report/index.html between the cover markers, report/figures/hero.jpg
and ../shared/thumbs/sounds-like-me.jpg.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE / "report"
THUMB = HERE.parent / "shared" / "thumbs" / "sounds-like-me.jpg"
sys.path.insert(0, str(HERE.parent / "shared"))
from cover_kit import (
    EMBER,
    EMBER_D,
    INK,
    INK2,
    LINE,
    MUTED,
    SLATE,
    caps,
    headline,
    kicker,
    lines,
    render_hero,
    svg,
    t,
    write_cover,
)

POINTS = [
    ("Qwen3-8B, no samples", "Qwen3-8B", "k0", MUTED),
    ("Claude Sonnet 5.5, 8 samples", "claude-sonnet-5.5", "k8_rand", MUTED),
    ("Gemini 4 Carbon, 8 samples", "gemini-4-carbon", "k8_rand", INK2),
    ("DIPPER paraphrase (never saw the author)", "humanizer-dipper:gemini-4-carbon", "k0", SLATE),
    ("Qwen3-8B + own voice vector", "partc-Qwen3-8B", "own", EMBER),
]


def opening(text, n=22):
    body = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("Subject:")]
    return " ".join(" ".join(body).split()[:n]) + " …"


def cover_svg():
    js = (REPORT / "data.js").read_text()
    d = json.loads(js[js.index("=") + 1 :].rstrip().rstrip(";"))
    ex = d["steer"]["examples"][0]
    T = {(r["model"], r["arm"]): r for r in json.loads((HERE / "results" / "leaderboard.json").read_text())}
    b = [kicker("SOUNDS LIKE ME · EPISODE 1", "RL FOR UNVERIFIABLE WRITING")]
    b.append(headline("Can any model sound like you?", None, 64, 26))
    y = 102
    b.append(t(24, y, "AUTHOR A, THEIR OWN EMAIL", 9.5, 700, MUTED, ls="0.08em"))
    b.append(
        lines(
            24,
            y + 18,
            "Attached are the following drafts: 1. Amendment to Compressor Services Agreement "
            "2. Amendment to Operation and Maintenance Agreement … Please review.",
            58,
            12,
            INK,
        )
    )
    y += 78
    b.append(t(24, y, "QWEN3-8B, SAME BRIEF, NO HELP", 9.5, 700, MUTED, ls="0.08em"))
    b.append(lines(24, y + 18, opening(ex["none"]["text"]), 58, 12, INK2))
    y += 78
    b.append(t(24, y, "WITH THE AUTHOR'S VOICE VECTOR ADDED", 9.5, 700, EMBER_D, ls="0.08em"))
    b.append(lines(24, y + 18, opening(ex["own"]["text"]), 58, 12, INK, italic=True))
    rx, rw, ry = 500, 330, 140

    def X(v):
        return rx + (v + 1.0) / 2.1 * rw

    b.append(t(rx - 6, ry - 34, "Voice position, 30 held-out authors", 12.5, 600, INK2))
    b.append(f'<path d="M{rx},{ry} H{rx + rw}" stroke="{LINE}" stroke-width="2"/>')
    for v, lab in ((0, "a different human"), (1, "the same human")):
        b.append(
            f'<path d="M{X(v):.1f},{ry - 8} V{ry + 250}" stroke="{LINE}" stroke-width="1" stroke-dasharray="3 3"/>'
        )
        b.append(t(X(v) + (6 if v else 0), ry - 12, f"{lab} ({v})", 10, 600, MUTED, "end" if v else "middle"))
    for v in (-1.0, -0.5, 0.5):
        b.append(t(X(v), ry + 266, f"{v:+.1f}", 10, 500, MUTED, "middle"))
    yy = ry + 30
    for lab, m, arm, col in POINTS:
        v = T[(m, arm)]["voice_strict"][0]
        b.append(f'<path d="M{X(-1.0):.1f},{yy} H{X(v):.1f}" stroke="{col}" stroke-width="2" stroke-opacity="0.35"/>')
        b.append(f'<circle cx="{X(v):.1f}" cy="{yy}" r="6" fill="{col}"/>')
        b.append(t(X(-1.0), yy - 10, f"{lab}  {v:+.2f}", 10.5, 600, col if col != MUTED else INK2))
        print(f"cover {lab:42s} {v:+.3f}")
        yy += 46
    b.append(f'<path d="M20,468 H860" stroke="{LINE}" stroke-width="1"/>')
    b.append(
        caps("8 MODELS · 6 SIGNALS · 30 HELD-OUT ENRON AND BLOG AUTHORS · 30 SYNTHETIC SALES BRIEFS · LUAR VOICE RULER")
    )
    label = (
        "Left: how one held-out author opens an email, how Qwen3-8B opens a synthetic sales email with no help "
        "(\"I hope you're doing well. My name is…\"), and how it opens with the author's voice vector added. "
        "Right: mean voice position on a scale from a different human (0) to the same human (1): every contestant "
        "sits at or below a different human; a paraphraser that never saw the author scores as well as Gemini "
        "with 8 samples."
    )
    return svg("".join(b), label, "slm-cover"), label


def main():
    s, label = cover_svg()
    write_cover(REPORT / "index.html", s)
    render_hero(s, REPORT, THUMB)
    print(label)


if __name__ == "__main__":
    main()
