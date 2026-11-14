"""Draw the note's cover from results/metrics.json: the reliability curve of base and tuned Glimmer on the full test.

    python experiments/glimmer-kalshi/make_cover.py

Each line is stated confidence (x) against how often the model was right (y), 15 equal bins over 33,712 test
questions; the diagonal is perfect calibration. Grey: base Muse Glimmer, which says "over 90% sure" on most questions
and is right 58% of the time. Orange: the replicated decision-tuned model after its temperature fit. Writes
report/figures/hero.jpg (1536 x 1024) and ../shared/thumbs/glimmer-kalshi.jpg (720 x 480).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parent
CREAM, INK, EMBER, GREY, RULE = "#F3EDE3", "#2B2723", "#C9561C", "#8F867B", "#D9D0C2"


def main() -> None:
    m = json.loads((HERE / "results" / "metrics.json").read_text())
    fig = plt.figure(figsize=(15.36, 10.24), dpi=100, facecolor=CREAM)
    ax = fig.add_axes([0.1, 0.13, 0.82, 0.62], facecolor=CREAM)
    ax.plot([0, 1], [0, 1], color=RULE, lw=3)
    for model, setting, colour, label in (
        ("base", "raw", GREY, "Base Glimmer"),
        ("tuned", "calibrated", EMBER, "Tuned + temperature"),
    ):
        bins = [b for b in m["models"][model][setting]["full"]["reliability"] if b["n"] >= 30]
        f = m["models"][model][setting]["full"]
        ax.plot(
            [b["conf"] for b in bins],
            [b["acc"] for b in bins],
            color=colour,
            lw=6,
            solid_capstyle="round",
        )
        x, y, ha = (0.33, 0.86, "left") if model == "tuned" else (0.99, 0.32, "right")
        ax.text(
            x,
            y,
            f"{label}\n{f['acc']:.1%} right, ECE {f['ece15']:.3f}",
            color=colour,
            fontsize=24,
            ha=ha,
            fontweight="bold",
        )
        print(model, [(round(b["conf"], 3), round(b["acc"], 3), b["n"]) for b in bins])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
    ax.tick_params(colors=INK, labelsize=18)
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0.3, 1.0)
    ax.set_xlabel("How sure the model says it is", fontsize=20, color=INK)
    ax.set_ylabel("How often it is right", fontsize=20, color=INK)
    fig.text(
        0.1,
        0.9,
        "Teaching Muse Glimmer to bet on Kalshi",
        fontsize=40,
        color=INK,
        fontweight="bold",
    )
    fig.text(
        0.1,
        0.84,
        "Settled markets, 33,712 held-out questions: confidence against accuracy",
        fontsize=24,
        color=GREY,
    )
    out = HERE / "report" / "figures" / "hero.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=CREAM, dpi=100, pil_kwargs={"quality": 90})
    thumb = HERE.parent / "shared" / "thumbs" / "glimmer-kalshi.jpg"
    Image.open(out).convert("RGB").resize((720, 480), Image.LANCZOS).save(thumb, quality=88)
    print("wrote", out.name, thumb.name)


if __name__ == "__main__":
    main()
