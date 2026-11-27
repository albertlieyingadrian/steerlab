"""Draw the cover from the data: the two training curves (each arm's own reward) and the held-out judge's verdicts.

    python make_cover.py ["cover title"]

Reads results/summary.json and results/runs/{ral_s0,frozen_s0}/log.jsonl. Writes the SVG into report/index.html
between the cover markers, report/figures/hero.jpg and ../shared/thumbs/ral-technical-writer.jpg.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))
sys.path.insert(1, str(HERE))
from cover_kit import EMBER, INK, INK2, LINE, MUTED, caps, kicker, rect, render_hero, svg, t, write_cover

import common as C

GREY, BLUE = "#9a948c", "#3f6fb0"
NAMES = {"base": "Qwen3-4B, untrained", "base_fs": "Qwen3-4B, 3-shot", "llama_fs": "Llama-3.3-70B, 3-shot",
         "sft": "SFT on the references", "frozen": "RL vs a frozen judge", "ral": "RAL, seed 0", "ral1": "RAL, seed 1 (696 words)"}  # fmt: skip


def curve(points, x0, y0, w, h, colour, steps=200):
    pts = " ".join(f"{x0 + w * s / steps:.1f},{y0 + h - h * v:.1f}" for s, v in points)
    return f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="2.2" stroke-linejoin="round"/>'


def smooth(v, k=10):
    return [sum(v[max(0, i - k + 1) : i + 1]) / len(v[max(0, i - k + 1) : i + 1]) for i in range(len(v))]


def cover_svg(title: str) -> str:
    S = json.loads((C.RESULTS / "summary.json").read_text())["sources"]
    b = [kicker("RAL AT SMALL SCALE · NIH GRANT ABSTRACTS", "A REPLICATION")]
    b.append(t(20, 62, title, 21, 800, INK))
    x0, y0, w, h = 34, 120, 360, 250
    b.append(rect(20, 84, 400, 330, "#ffffff", LINE, 1, 6))
    b.append(t(34, 106, "TRAINING REWARD: EACH ARM'S OWN JUDGE", 9.5, 700, MUTED, ls="0.08em"))
    for frac in (0, 0.5, 1):
        b.append(f'<path d="M{x0},{y0 + h - h * frac:.1f} H{x0 + w}" stroke="{LINE}" stroke-width="1"/>')
        b.append(t(x0 + w + 4, y0 + h - h * frac + 3, f"{frac:.1f}", 8.5, 500, MUTED))
    for run, colour, label in (
        ("frozen_s0", BLUE, "frozen judge"),
        ("ral_s1", "#e89a6a", ""),
        ("ral_s0", EMBER, "RAL, 2 seeds"),
    ):
        p = C.RESULTS / "runs" / run / "log.jsonl"
        if not p.exists():
            continue
        L = C.read_jsonl(p)
        v = smooth([r["reward"] for r in L])
        b.append(curve(list(zip([r["step"] for r in L], v)), x0, y0, w, h, colour))
        if label:
            b.append(t(x0 + w - 4, y0 + h - h * v[-1] - 6, label, 10, 700, colour, "end"))
    b.append(t(x0, y0 + h + 18, "step 0", 8.5, 500, MUTED))
    b.append(t(x0 + w, y0 + h + 18, "200", 8.5, 500, MUTED, "end"))
    xr, y = 450, 106
    b.append(t(xr, y, "HELD-OUT JUDGES FOOLED (gpt-oss-120b + GLM-4.5-Air, 200 titles)", 9.5, 700, MUTED, ls="0.08em"))
    y += 14
    for s in ("base", "base_fs", "llama_fs", "sft", "frozen", "ral", "ral1"):
        if s not in S or "ensemble_fooling" not in S[s]:
            continue
        v = S[s]["ensemble_fooling"]
        col = EMBER if s == "ral" else "#e89a6a" if s == "ral1" else BLUE if s == "frozen" else GREY
        b.append(t(xr, y + 12, NAMES[s], 10.5, 500, INK2))
        b.append(rect(xr + 150, y + 2, max(240 * v / 0.5, 1.5), 13, col))
        b.append(t(xr + 156 + 240 * v / 0.5, y + 12.5, f"{v:.2f}", 10, 600, INK))
        y += 24
    b.append(f'<path d="M{xr + 150 + 240},{120} V{y}" stroke="{MUTED}" stroke-dasharray="3 3" stroke-width="1"/>')
    b.append(t(xr + 150 + 240, y + 12, "0.5 = cannot tell", 8.5, 500, MUTED, "end"))
    b.append(t(xr, y + 40, "At equal length, no source beats the untrained model:", 10.5, 700, INK2))
    b.append(t(xr, y + 58, "the judges are fooled 0.18 near the real length, 0.39 at over twice it.", 10.5, 400, INK2))
    b.append(f'<path d="M20,468 H860" stroke="{LINE}" stroke-width="1"/>')
    b.append(caps("QWEN3-4B GENERATOR · QWEN3-8B JUDGE · LORA · 200 STEPS x 128 ROLLOUTS · NIH FY2026 REFERENCES"))
    label = ("Left: the training reward of the RAL run and of the frozen-judge control over 200 steps, each scored by its "
             "own judge. Right: the share of verdicts in which the held-out judges named the real abstract as the AI one, "
             "per source (mean of gpt-oss-120b and GLM-4.5-Air), on 200 held-out titles; 0.5 means they cannot tell.")  # fmt: skip
    return svg("".join(b), label, "ral-cover")


def main() -> None:
    title = sys.argv[1] if len(sys.argv) > 1 else "A 4B RAL writer: starved by its judge, then longer"
    s = cover_svg(title)
    report = HERE / "report"
    write_cover(report / "index.html", s)
    render_hero(s, report, HERE.parent / "shared" / "thumbs" / "ral-technical-writer.jpg")
    print("cover written; hero.jpg and thumb rendered")


if __name__ == "__main__":
    main()
