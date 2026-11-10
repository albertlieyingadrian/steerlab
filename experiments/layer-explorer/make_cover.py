"""Draw the cover and the research-page thumbnail from the explorer's data: the "Capital of France" layer stack.

    .venv/bin/python experiments/layer-explorer/make_cover.py

Each row is one layer of Qwen2.5-0.5B-Instruct, bottom (layer 0) to top (layer 23): attention's share of the
" Paris" logit in blue, the neurons' share in orange, and what the logit lens says after that layer on the right.
Writes report/figures/hero.jpg (1600x900) and experiments/shared/thumbs/layer-explorer.jpg (720x480). Prints the
numbers it drew.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE / "report" / "explorer" / "data.json"
HERO = HERE / "report" / "figures" / "hero.jpg"
THUMB = HERE.parent / "shared" / "thumbs" / "layer-explorer.jpg"
GROUND, FAINT = "#15110e", "#8a8178"
ATTN, MLP, MARK = "#4a86e8", "#e07034", "#f2b48e"


def show(text: str) -> str:
    text = text.replace("\n", "↵")
    return text.replace(" ", "␣") if not text.strip() else text


def draw(size: tuple[float, float], out: Path, *, words: bool) -> None:
    run = json.loads(DATA.read_text())["presets"][0]["run"]
    layers = [row for row in run["layers"] if row["layer"] >= 0]
    scale = max(max(abs(r["attn"]), abs(r["mlp"])) for r in layers)
    fig = plt.figure(figsize=size, dpi=100, facecolor=GROUND)
    ax = fig.add_axes((0.04, 0.05, 0.92, 0.9), facecolor=GROUND)
    ax.set_axis_off()
    for y, row in enumerate(layers):
        hit = row["target_rank"] == 1
        label_color = MARK if hit else FAINT
        ax.text(
            -0.08,
            y,
            f"L{row['layer']}",
            color=label_color,
            ha="right",
            va="center",
            fontsize=15 if words else 10,
            family="monospace",
        )
        for x0, value, color in ((0.0, row["attn"], ATTN), (1.15, row["mlp"], MLP)):
            ax.plot([x0 + 0.5, x0 + 0.5], [y - 0.45, y + 0.45], color="#3a322c", lw=1)
            ax.barh(y, 0.5 * value / scale, left=x0 + 0.5, height=0.62, color=color, alpha=0.9)
        if words:
            ax.text(2.42, y, show(row["says"]), color=label_color, va="center", fontsize=17, family="monospace")
    ax.set_xlim(-0.3, 3.1 if words else 2.3)
    ax.set_ylim(-1, len(layers))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=GROUND, pil_kwargs={"quality": 90})
    plt.close(fig)
    print("wrote", out.name)
    if words:
        for row in layers[-6:]:
            print(f"  L{row['layer']}: attention {row['attn']:+.2f}, neurons {row['mlp']:+.2f}, says {row['says']!r}")


def main() -> None:
    draw((16, 9), HERO, words=True)
    draw((7.2, 4.8), THUMB, words=False)


if __name__ == "__main__":
    main()
