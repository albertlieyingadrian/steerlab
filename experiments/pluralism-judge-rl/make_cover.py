"""Draw the cover from the data: one house style across unrelated requests.

    python experiments/pluralism-judge-rl/make_cover.py

Left: the opening of the base model's reply and of the single-judge model's reply (seed 0, step 200) to two unrelated
held-out creative requests, read from results/samples/. Right: house style (mean similarity between replies to
different creative prompts) against drift for every run, from results/checkpoints.csv. Writes the SVG into
report/index.html between the cover markers, report/figures/hero.jpg and ../shared/thumbs/pluralism-judge-rl.jpg.
"""

import csv
import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE / "report"
THUMB = HERE.parent / "shared" / "thumbs" / "pluralism-judge-rl.jpg"
sys.path.insert(0, str(HERE.parent / "shared"))
from cover_kit import (  # noqa: E402
    EMBER_D,
    INK,
    INK2,
    LINE,
    MUTED,
    caps,
    headline,
    kicker,
    lines,
    rect,
    render_hero,
    svg,
    t,
    write_cover,
)

COLOR = {"single": "#d26f2a", "jury": "#7a4fa0", "bon": "#2f8f6b", "shuffled": "#4a86e8"}
NAME = {"single": "single judge", "jury": "jury", "bon": "best-of-8", "shuffled": "shuffled judge"}
PICKS = [
    ("ce001", "“Generate a joke about electric vehicles.”"),
    ("ce006", "“Write me a song about a guy named Jacob…”"),
]


def opening(tag, pid, n=17):
    with gzip.open(HERE / "results" / "samples" / f"{tag}.jsonl.gz", "rt") as f:
        rows = [json.loads(line) for line in f]
    text = next(r["text"] for r in rows if r["task"] == "creative" and r["id"] == pid and r["k"] == 0)
    return " ".join(text.replace("*", "").split()[:n]) + " …"


def cover_svg():
    with (HERE / "results" / "checkpoints.csv").open() as f:
        rows = list(csv.DictReader(f))
    base = next(r for r in rows if r["run"] == "base-s0")
    b = [kicker("PLURALISM · EPISODE 3", "DOES JUDGE-RL ERODE PLURALITY?")]
    b.append(headline("Trained against a judge, every reply finds one voice", None, 64, 26))
    y = 104
    for pid, label in PICKS:
        b.append(t(24, y, label, 12.5, 700, INK))
        b.append(t(24, y + 20, "BASE", 9.5, 700, MUTED, ls="0.08em"))
        b.append(lines(70, y + 20, opening("base-s0", pid), 50, 12, INK2))
        b.append(t(24, y + 64, "JUDGE", 9.5, 700, EMBER_D, ls="0.08em"))
        b.append(lines(70, y + 64, opening("single-creative-s0-step200", pid), 50, 12, INK, italic=True))
        y += 132
    rx, ry, rw, rh = 500, 110, 340, 260
    x_hi, y_hi = 2.5, 0.5

    def X(v):
        return rx + v / x_hi * rw

    def Y(v):
        return ry + rh - v / y_hi * rh

    b.append(t(rx - 30, 98, "Similarity between replies to different prompts", 12.5, 600, INK2))
    for v in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        b.append(f'<path d="M{rx},{Y(v):.1f} H{rx + rw}" stroke="{LINE}" stroke-width="1"/>')
        b.append(t(rx - 6, Y(v) + 4, f"{v:.1f}", 10, 500, MUTED, "end"))
    for v in (0, 1, 2):
        b.append(t(X(v), ry + rh + 16, str(v), 10, 500, MUTED, "middle"))
    b.append(t(rx + rw / 2, ry + rh + 34, "drift from the base model (nats per token)", 11, 600, INK2, "middle"))
    for run in sorted({r["run"] for r in rows if r["task"] == "creative"}):
        pts = sorted([r for r in rows if r["run"] == run], key=lambda r: int(r["step"]))
        arm = pts[0]["arm"]
        xy = [(0.0, float(base["house_creative"]))] + [(float(p["drift_kl"]), float(p["house_creative"])) for p in pts]
        path = " ".join(f"{'M' if i == 0 else 'L'}{X(x):.1f},{Y(yv):.1f}" for i, (x, yv) in enumerate(xy))
        b.append(f'<path d="{path}" fill="none" stroke="{COLOR[arm]}" stroke-width="2.4" stroke-linejoin="round"/>')
        b.append(f'<circle cx="{X(xy[-1][0]):.1f}" cy="{Y(xy[-1][1]):.1f}" r="4" fill="{COLOR[arm]}"/>')
    ly = ry + 8
    for arm in ["single", "jury", "bon", "shuffled"]:
        b.append(rect(rx + 12, ly - 8, 10, 10, COLOR[arm], rx=2))
        b.append(t(rx + 28, ly + 1, NAME[arm], 11, 600, INK2))
        ly += 17
    b.append(
        t(
            X(0) + 6,
            Y(float(base["house_creative"])) + 16,
            f"base {float(base['house_creative']):.2f}",
            10.5,
            600,
            MUTED,
        )
    )
    b.append(f'<path d="M20,468 H860" stroke="{LINE}" stroke-width="1"/>')
    b.append(caps("QWEN3-8B · 16 RUNS · 100 HELD-OUT CREATIVE PROMPTS × 8 REPLIES · TEXT-EMBEDDING-3-SMALL"))
    label = (
        "Left: the opening of the base model's reply and of the judge-trained model's reply to a request for a joke "
        "and "
        "a request for a song; the trained replies share one lyrical register. Right: similarity between replies to "
        "different creative prompts against drift from the base model, for every creative run: near the base model's "
        "level for the shuffled-judge arm, about three times higher for every arm trained on the judge's scores."
    )
    return svg("".join(b), label, "pjr-cover"), label


def main():
    s, label = cover_svg()
    write_cover(REPORT / "index.html", s)
    render_hero(s, REPORT, THUMB)
    print(label)


if __name__ == "__main__":
    main()
