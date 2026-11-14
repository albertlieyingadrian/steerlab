"""Draw the cover and the research-page thumbnail from the monitor's own scores.

    python experiments/monitor-evasion/make_cover.py

Two ridgelines of the layer-17 monitor's score on the test answers: before fine-tuning (top) the sandbagged answers
(orange) sit far to the right of the honest ones (light); after the probe-only LoRA (bottom) both sit far to the left
of the alarm line (the dashed rule), although the model still sandbags. Each answer is also a small tick under its
ridge. Same inputs, same picture. Writes report/figures/hero.jpg (1536x1024) and ../shared/thumbs/monitor-evasion.jpg
(720x480).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
GROUND = "#15110e"
ORANGE, LIGHT, RULE = "#e07a3f", "#e8dfd4", "#8a8178"


def kde(x: np.ndarray, grid: np.ndarray, bw: float) -> np.ndarray:
    return np.exp(-0.5 * ((grid[:, None] - x[None, :]) / bw) ** 2).sum(1) / (len(x) * bw * np.sqrt(2 * np.pi))


def main() -> None:
    rows = list(csv.DictReader((HERE / "results" / "scores.csv").open()))
    thr = json.loads((HERE / "results" / "baseline.json").read_text())["monitors"]["17"]["threshold"]
    data = {}
    for arm in ("base", "P"):
        sub = [r for r in rows if r["arm"] == arm and r["capable"] == "1"]
        data[arm] = {k: np.array([float(r["s17"]) for r in sub if r["correct"] == v])
                     for k, v in (("wrong", "0"), ("right", "1"))}  # fmt: skip
    allv = np.concatenate([v for d in data.values() for v in d.values()])
    lo, hi = allv.min() - 3, allv.max() + 3
    grid = np.linspace(lo, hi, 800)
    fig = plt.figure(figsize=(15.36, 10.24), dpi=100, facecolor=GROUND)
    ax = fig.add_axes((0.06, 0.08, 0.88, 0.84), facecolor=GROUND)
    for i, arm in enumerate(("base", "P")):
        base_y = 1.0 - i * 1.0
        for key, colour in (("right", LIGHT), ("wrong", ORANGE)):
            dens = kde(data[arm][key], grid, bw=(hi - lo) / 90)
            dens = dens / dens.max() * 0.7
            ax.fill_between(grid, base_y, base_y + dens, color=colour, alpha=0.35, lw=0)
            ax.plot(grid, base_y + dens, color=colour, lw=2.2)
            ticks_y = base_y - 0.06 - (0.05 if key == "wrong" else 0)
            ax.scatter(data[arm][key], np.full(len(data[arm][key]), ticks_y), marker="|", s=90, color=colour,
                       alpha=0.6, lw=1)  # fmt: skip
    ax.axvline(thr, color=RULE, lw=1.6, ls=(0, (4, 4)))
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.2, 1.85)
    ax.axis("off")
    out = HERE / "report" / "figures" / "hero.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".png")
    fig.savefig(tmp, facecolor=GROUND)
    plt.close(fig)
    img = Image.open(tmp).convert("RGB").resize((1536, 1024), Image.LANCZOS)
    img.save(out, quality=90)
    img.resize((720, 480), Image.LANCZOS).save(HERE.parent / "shared" / "thumbs" / "monitor-evasion.jpg", quality=88)
    tmp.unlink()
    print(f"wrote {out.relative_to(HERE.parents[1])} and the thumbnail")


if __name__ == "__main__":
    main()
