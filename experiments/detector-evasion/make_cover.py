"""The note's cover, drawn from its data: the 9B's score on every training rewrite of three arms, step by step.

    python make_cover.py     # -> report/figures/hero.jpg (1600x900) and ../shared/thumbs/detector-evasion.jpg

Each dot is one rewrite sampled during GRPO (8 sources x 8 rewrites per step), placed at its step and at the 9B
detector's log-odds that the text is human, log((1 - s9) / s9), clipped to [-12, 6]. The dashed line is the flag
threshold (s9 = 0.372, flags 1% of 318 human passages); dots above it pass. Orange: the full reward; ink: detector
only; grey: the shuffled-reward control. A fixed seed thins the dots to 16 per step per arm and jitters them; the
plotted points are saved to results/cover_points.json, so the cover rebuilds without the (gitignored) rollouts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROLL = HERE / "data" / "rollouts"
POINTS = HERE / "results" / "cover_points.json"
ACCENT, INK, FAINT = "#c9561c", "#1d272a", "#9aa3a6"
ARMS = [("shuffled-s0", FAINT, "shuffled reward"), ("det-only-s0", INK, "detector only"),
        ("full-s0", ACCENT, "full reward")]  # fmt: skip
THRESH = 0.372
LO, HI = -12.0, 6.0


def logodds(s9: float) -> float:
    s9 = min(max(s9, 1e-12), 1 - 1e-12)
    return min(max(math.log((1 - s9) / s9), LO), HI)


def points() -> dict[str, list[list[float]]]:
    if all((ROLL / f"{a}.jsonl").exists() for a, _, _ in ARMS):
        rng = np.random.default_rng(7)
        out = {}
        for arm, _, _ in ARMS:
            by: dict[int, list[float]] = {}
            for line in (ROLL / f"{arm}.jsonl").open():
                r = json.loads(line)
                by.setdefault(r["step"], []).append(r["s9"])
            pts = []
            for step, vals in sorted(by.items()):
                for v in rng.choice(vals, size=min(16, len(vals)), replace=False):
                    pts.append([round(step + float(rng.uniform(-0.4, 0.4)), 2), round(logodds(float(v)), 3)])
            out[arm] = pts
        POINTS.write_text(json.dumps(out) + "\n")
        return out
    return json.loads(POINTS.read_text())


def main() -> None:
    P = points()
    fig = plt.figure(figsize=(8, 4.5), dpi=200)
    fig.patch.set_facecolor("#ffffff")
    ax = fig.add_axes([0.07, 0.1, 0.9, 0.7])
    for arm, col, lab in ARMS:
        xy = np.array(P[arm])
        ax.scatter(xy[:, 0], xy[:, 1], s=2.2, color=col, alpha=0.45 if arm != "full-s0" else 0.6, lw=0, label=lab)
    t = math.log((1 - THRESH) / THRESH)
    ax.axhline(t, color=INK, lw=0.8, ls=(0, (4, 3)))
    ax.text(2, t + 0.35, "passes the 9B detector", fontsize=7, color=INK)
    ax.set_ylim(LO - 0.3, HI + 0.3)
    ax.set_xlim(0, max(max(p[0] for p in pts) for pts in P.values()) + 2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(FAINT)
    ax.spines["bottom"].set_color(FAINT)
    ax.tick_params(labelsize=7, colors=FAINT)
    ax.set_xlabel("GRPO step (each dot one rewrite during training)", fontsize=7.5, color=FAINT)
    ax.set_ylabel("9B detector: log-odds the text is human", fontsize=7.5, color=FAINT)
    leg = ax.legend(loc="lower right", fontsize=7, frameon=False, markerscale=4)
    for txt in leg.get_texts():
        txt.set_color(INK)
    fig.text(0.03, 0.93, "MODEL VOICE · EPISODE 5", fontsize=8, color=ACCENT, weight="bold")
    fig.text(0.03, 0.865, "Rewrite until it passes the detector", fontsize=12.5, color=INK, weight="bold")
    out = HERE / "report" / "figures" / "hero.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".png")
    fig.savefig(tmp, facecolor="#ffffff")
    plt.close(fig)
    img = Image.open(tmp).convert("RGB").resize((1600, 900), Image.Resampling.LANCZOS)
    img.save(out, quality=88, optimize=True)
    tmp.unlink()
    thumb = img.resize((800, 450), Image.Resampling.LANCZOS).crop((40, 0, 760, 450)).resize((720, 480))
    thumb.save(HERE.parent / "shared" / "thumbs" / "detector-evasion.jpg", quality=84, optimize=True)
    for arm, _, _ in ARMS:
        ys = np.array(P[arm])
        late = ys[ys[:, 0] >= ys[:, 0].max() - 20, 1]
        print(arm, "last 20 steps: share above threshold", round(float((late > t).mean()), 3), "n", len(late))


if __name__ == "__main__":
    main()
