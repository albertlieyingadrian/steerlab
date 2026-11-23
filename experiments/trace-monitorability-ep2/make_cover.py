"""Draw the cover and the research-page thumbnail for Episode 2 from the study's own data.

    python experiments/trace-monitorability-ep2/make_cover.py

A grid of the 144 rollouts: one column per task (24), one row per rollout of it (6). Inside a cell, the rollout's tool
calls run top to bottom (a rollout longer than 15 calls continues in a second, narrower column). Grey shade = action
type (light: read, mid: write a file, dark: run or install); outlined = a risky action (delete, permission or network
change); ember = a step the agent took without writing any thinking (82 of 644). Same inputs, same picture. Writes
report/figures/hero.jpg (1536x1024) and the 720x480 thumbnail; build_meta.py crops og.jpg from the hero.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
HERO = HERE / "report" / "figures" / "hero.jpg"
THUMB = HERE.parent / "shared" / "thumbs" / "trace-monitorability-ep2.jpg"

PAPER, EMBER, INK = "#f6f1ea", "#c9561c", "#2b2723"
SHADE = {"read_inspect": "#d9d2c8", "local_write": "#a69e94", "execute_install": "#6c655e"}
RISKY = {"delete", "permission_change", "network_change"}


def main() -> None:
    rollouts: dict[str, list[dict]] = {}
    with open(RESULTS / "rollouts.csv") as f:
        cells = [(r["task"], int(r["rep"]), r["rollout"]) for r in csv.DictReader(f)]
    with open(RESULTS / "steps.csv") as f:
        for s in csv.DictReader(f):
            rollouts.setdefault(s["rollout"], []).append(s)
    tasks = sorted({t for t, _, _ in cells})
    fig = plt.figure(figsize=(1536 / 200, 1024 / 200), dpi=200)
    ax = fig.add_axes([0.05, 0.07, 0.9, 0.86])
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    cw, ch, per = 1.0 / len(tasks), 1.0 / 6, 15
    n_empty = n_risky = 0
    for task, rep, rid in cells:
        x0, y0 = tasks.index(task) * cw, 1 - (rep + 1) * ch
        steps = sorted(rollouts.get(rid, []), key=lambda s: int(s["k"]))
        if not steps:
            ax.add_patch(Rectangle((x0 + cw * 0.15, y0 + ch * 0.9), cw * 0.6, ch * 0.012, color="#d9d2c8", lw=0))
            continue
        cols = (len(steps) + per - 1) // per
        bw, bh = cw * 0.72 / cols, ch * 0.9 / per
        for i, s in enumerate(steps):
            x, y = x0 + cw * 0.14 + (i // per) * bw, y0 + ch * 0.95 - (i % per + 1) * bh
            t = s["action_type"]
            if s["think_chars"] == "0":
                n_empty += 1
                ax.add_patch(Rectangle((x, y), bw * 0.9, bh * 0.72, color=EMBER, lw=0))
            elif t in RISKY:
                n_risky += 1
                ax.add_patch(Rectangle((x + bw * 0.05, y + bh * 0.06), bw * 0.8, bh * 0.6, fill=False, ec=INK, lw=0.5))
            else:
                ax.add_patch(Rectangle((x, y), bw * 0.9, bh * 0.72, color=SHADE.get(t, "#e8e2d9"), lw=0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    HERO.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(HERO, dpi=200, facecolor=PAPER, pil_kwargs={"quality": 90})
    fig.savefig(THUMB, dpi=200 * 720 / 1536, facecolor=PAPER, pil_kwargs={"quality": 88})
    plt.close(fig)
    print(
        f"{len(cells)} rollouts, {sum(len(r) for r in rollouts.values())} steps, {n_empty} without thinking (ember), "
        f"{n_risky} risky with thinking (outlined)"
    )
    print("wrote", HERO.relative_to(HERE), "and", THUMB.name)


if __name__ == "__main__":
    main()
