"""The note's cover and research-index thumbnail, drawn from GRAM's evaluation table.

One square per (deployed profile, test topic): rows are the five profiles (no module, then each topic's module
switched on), columns the four restricted topics. Brightness is how far that topic's test loss sits above the
baseline's (mean over seeds). Bright squares are topics whose module is off; the dark diagonal is each topic with
its own module on. build_meta.py crops the Open Graph image from the hero.

    .venv/bin/python experiments/repl-gram-modular-pretraining/make_cover.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
GROUND = "#15110e"
EMBER = LinearSegmentedColormap.from_list("ember", [GROUND, "#3a2a20", "#a0532a", "#e07a3f", "#f6c39c"])
AUX = [
    "a-deadline-or-time-limit",
    "alien-encounters",
    "bygone-eras",
    "cultural-traditions",
]
PROFILES = ["core"] + [f"core+{a}" for a in AUX]


def table():
    rows = []
    for d in sorted((HERE / "results" / "runs").glob("seed_*")):
        g, b = d / "gram_eval.json", d / "baseline.json"
        if not (g.exists() and b.exists()):
            continue
        g, b = json.loads(g.read_text()), json.loads(b.read_text())
        rows.append([[g["profiles"][p]["test"][a] - b["test"][a] for a in AUX] for p in PROFILES])
    return np.mean(np.array(rows), axis=0)


def draw(size, out, t):
    w, h = size
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    fig.patch.set_facecolor(GROUND)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_facecolor(GROUND)
    top = max(t.max(), 1e-6)
    n_r, n_c = t.shape
    cell = 0.82
    for r in range(n_r):
        for c in range(n_c):
            shade = float(np.clip(t[r, c] / top, 0, 1)) ** 0.7
            ax.add_patch(plt.Rectangle((c, n_r - 1 - r), cell, cell, color=EMBER(0.08 + 0.92 * shade), lw=0))
    span_y = n_r + 1.2
    span_x = span_y * w / h
    cx = (n_c - (1 - cell)) / 2
    ax.set_xlim(cx - span_x / 2, cx + span_x / 2)
    ax.set_ylim(-0.6, n_r + 0.6)
    ax.set_aspect("equal")
    ax.axis("off")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=GROUND, pil_kwargs={"quality": 90})
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    t = table()
    print("loss above baseline (nats), rows = profiles, cols = topics:")
    for p, row in zip(PROFILES, t, strict=True):
        print(f"  {p:34s} " + " ".join(f"{v:+.4f}" for v in row))
    draw((1600, 900), HERE / "report" / "figures" / "hero.jpg", t)
    draw(
        (720, 480),
        HERE.parent / "shared" / "thumbs" / "repl-gram-modular-pretraining.jpg",
        t,
    )
