"""Draw the note's cover from its own results, with no text: one strip per model over the 307 held-out agent tasks.

    .venv/bin/python experiments/agent-traces/make_cover.py [--thumb path/to/thumbs/agent-traces.jpg]

Columns are the 307 held-out tasks (144 eval + 163 test), sorted from the most to the fewest models solving them.
Rows, top to bottom: the teacher (Qwen3.5-397B-A17B), then Qwen3.5-2B trained on 2,238, 963, 473 (the guide's),
473 (ours), 241 and 120 traces, then the base model. A mark is a solved task: dark for the teacher and the base,
orange for the trained students. Writes report/figures/hero.jpg (1536 x 1024) and og.jpg (1200 x 630).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

HERE = Path(__file__).resolve().parent
CREAM, INK, EMBER = "#F3EDE3", "#2B2723", "#C9561C"
ROWS = ["released-teacher", "released-sft-full", "released-sft-1k", "released-sft-500", "ours-sft500s42",
        "ours-sft250s42", "ours-sft125s42", "ours-base"]  # fmt: skip


def solved() -> dict[str, dict[tuple[str, int], int]]:
    test = [int(x) for x in (HERE / "data" / "splits" / "test_indices.txt").read_text().strip().split(",")]
    keys = [("eval", i) for i in range(144)] + [("test", i) for i in test]
    out: dict = {}
    for f in (HERE / "results" / "per_task").glob("*.jsonl"):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            out.setdefault(r["run"], {})[(r["split"], r["index"])] = r["solved"]
    return {run: {k: out.get(run, {}).get(k, 0) for k in keys} for run in ROWS}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--thumb", default=None)
    a = p.parse_args()
    s = solved()
    keys = sorted(next(iter(s.values())), key=lambda k: -sum(s[r][k] for r in ROWS))
    fig = plt.figure(figsize=(15.36, 10.24), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(CREAM)
    x0, x1, top, h, gap = 0.07, 0.93, 0.84, 0.055, 0.035
    w = (x1 - x0) / len(keys)
    for i, run in enumerate(ROWS):
        y = top - i * (h + gap) - (0.04 if i >= 1 else 0) - (0.04 if run == "ours-base" else 0)
        colour = INK if run in ("released-teacher", "ours-base") else EMBER
        for j, k in enumerate(keys):
            if s[run][k]:
                ax.add_patch(Rectangle((x0 + j * w, y), w * 0.72, h, color=colour, lw=0))
    out = HERE / "report" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "hero.png", facecolor=CREAM)
    img = Image.open(out / "hero.png").convert("RGB")
    img.save(out / "hero.jpg", quality=90)
    (out / "hero.png").unlink()
    w0, h0 = img.size
    hh = int(w0 * 630 / 1200)
    img.crop((0, (h0 - hh) // 2, w0, (h0 + hh) // 2)).resize((1200, 630)).save(out / "og.jpg", quality=88)
    if a.thumb:
        img.resize((720, 480)).save(a.thumb, quality=85)
    print("rows (solved of 307):", {r: sum(s[r].values()) for r in ROWS})


if __name__ == "__main__":
    main()
