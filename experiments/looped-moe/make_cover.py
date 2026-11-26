"""The note's cover, Open Graph image and research-index thumbnail, drawn from the lens results.

Each column is one layer visit of the toy model (prelude, three core passes, coda, output), each row one way of
reading it: the plain loop's single residual stream (bottom, blue), then the looped MoE's active block, its visit's
own MoE-input mix and the output head's mix (orange, top). A cell's brightness is the share of questions whose top
token is the answer. The gaps split the prelude, the three passes and the coda.

    .venv/bin/python experiments/looped-moe/make_cover.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
GROUND = "#15110e"
EMBER = LinearSegmentedColormap.from_list("ember", [GROUND, "#3a2a20", "#a0532a", "#e07a3f", "#f6c39c"])
BLUE = LinearSegmentedColormap.from_list("blue", [GROUND, "#1d2c3f", "#2f6fb0", "#8fb8e6"])


def lens():
    for name in ("lens_20k_s1337.json", "lens_6k_s1337.json"):
        p = HERE / "results" / name
        if p.exists():
            return json.loads(p.read_text())
    raise SystemExit("no lens results")


def draw(size, out):
    res = lens()
    lm, pl = res["loopmoe"], res["plainloop"]
    n = len(lm["visits"])
    gaps = {2, 6, 10, 14}  # columns where a new section starts: pass 1, pass 2, pass 3, coda
    xs, x = [], 0.0
    for i in range(n):
        if i in gaps:
            x += 0.45
        xs.append(x)
        x += 1.0
    w, h = size
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    fig.patch.set_facecolor(GROUND)
    ax = fig.add_axes([0.05, 0.12, 0.9, 0.76])
    ax.set_facecolor(GROUND)
    rows = [([None, None] + pl["act_in"], BLUE), (lm["act_in"], EMBER), (lm["mlp_in"], EMBER), (lm["mix"], EMBER)]
    for r, (vals, cmap) in enumerate(rows):
        for i, v in enumerate(vals):
            if v is None or v != v:  # visit not read, or NaN
                continue
            shade = min(1.0, max(0.0, (v - 0.05) / 0.95))
            ax.add_patch(plt.Rectangle((xs[i], r * 1.25), 0.86, 0.95, color=cmap(shade), lw=0))
    ax.set_xlim(-0.3, xs[-1] + 1.2)
    ax.set_ylim(-0.3, len(rows) * 1.25)
    ax.axis("off")
    fig.savefig(out, facecolor=GROUND)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    figs = HERE / "report" / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    draw((1600, 900), figs / "hero.jpg")
    draw((1200, 630), figs / "og.jpg")
    draw((720, 480), HERE.parent / "shared" / "thumbs" / "looped-moe.jpg")
