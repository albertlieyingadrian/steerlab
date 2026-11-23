"""Draw the cover and the research-page thumbnail for the trace-monitorability note from the study's own data.

    python experiments/trace-monitorability/make_cover.py

One row per trajectory (120), one short bar per agent step, in order. Grey shade = action type (light: read or
inspect, mid: write a file, dark: run code or install), outlined: delete. Ember = an action that no reasoning up to
that step mentions (M3 "never", results/unmentioned.csv). Rows are sorted by length. Same inputs, same picture.
Writes report/figures/hero.jpg (1536x1024) and the 720x480 thumbnail; build_meta.py crops og.jpg from the hero.
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
THUMB = HERE.parent / "shared" / "thumbs" / "trace-monitorability.jpg"

PAPER, EMBER, INK = "#f6f1ea", "#c9561c", "#2b2723"
SHADE = {"read_inspect": "#d9d2c8", "local_write": "#a69e94", "execute_install": "#6c655e", "submit": "#2b2723"}


def main() -> None:
    with open(RESULTS / "unmentioned.csv") as f:
        never = {(u["episode"], u["k"]) for u in csv.DictReader(f) if u["where"] == "never"}
    episodes: dict[str, list[dict]] = {}
    with open(RESULTS / "actions.csv") as f:
        for a in csv.DictReader(f):
            episodes.setdefault(a["episode"], []).append(a)
    rows = sorted(episodes.values(), key=len)
    longest = max(len(r) for r in rows)
    fig = plt.figure(figsize=(1536 / 200, 1024 / 200), dpi=200)
    ax = fig.add_axes([0.06, 0.08, 0.88, 0.84])
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    w, h = 1.0 / longest, 1.0 / len(rows)
    n_never = 0
    for i, row in enumerate(rows):
        for a in row:
            x, y = (int(a["k"]) - 1) * w, i * h
            t = a["action_type"]
            if (a["episode"], a["k"]) in never:
                n_never += 1
                ax.add_patch(Rectangle((x, y), w * 0.86, h * 0.72, color=EMBER, lw=0))
            elif t == "delete":
                ax.add_patch(Rectangle((x + w * 0.1, y + h * 0.1), w * 0.66, h * 0.52, fill=False, ec=INK, lw=0.5))
            else:
                ax.add_patch(Rectangle((x, y), w * 0.86, h * 0.72, color=SHADE.get(t, "#e8e2d9"), lw=0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    HERO.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(HERO, dpi=200, facecolor=PAPER, pil_kwargs={"quality": 90})
    fig.savefig(THUMB, dpi=200 * 720 / 1536, facecolor=PAPER, pil_kwargs={"quality": 88})
    plt.close(fig)
    print(f"{len(rows)} trajectories, {sum(len(r) for r in rows)} steps, {n_never} never mentioned (ember)")
    print("wrote", HERO.relative_to(HERE), "and", THUMB.name)


if __name__ == "__main__":
    main()
