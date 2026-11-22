"""The note's cover, drawn from its data: every Track A reply's habit score, one row per arm.

    python make_cover.py     # -> report/figures/hero.jpg (1600x900) and ../shared/thumbs/deslop-h2h.jpg (720x480)

Each dot is one held-out reply (96 per arm, 48 prompts x 2 samples) at its habit score, clipped at 120 per 1,000 words;
the pale band is the human passages' range up to their 90th percentile (33.6). Orange rows are steering arms, the rest
are ink; the bar on each row is the arm's mean. Same inputs, same picture (the vertical jitter uses a fixed seed).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from make_figures import ACCENT, BAND, FAINT, HUMAN_P90, INK, LABEL, STEER

HERE = Path(__file__).resolve().parent
ORDER = ["original", "plain", "sampler", "plain+sampler", "register", "probe", "sae-dir", "sae-abl",
         "plain+register", "plain+sae", "sft", "dpo", "sft+sampler"]  # fmt: skip
CLIP = 120.0


def main() -> None:
    by: dict[str, list[float]] = defaultdict(list)
    for r in csv.DictReader((HERE / "results" / "per_reply.csv").open()):
        if r["group"].startswith("A/"):
            by[r["group"][2:]].append(float(r["habit"]))
    rng = np.random.default_rng(7)
    fig = plt.figure(figsize=(8, 4.5), dpi=200)
    fig.patch.set_facecolor("#ffffff")
    ax = fig.add_axes([0.2, 0.08, 0.76, 0.74])
    ax.axvspan(0, HUMAN_P90, color=BAND, zorder=0)
    for i, arm in enumerate(reversed(ORDER)):
        v = np.minimum(np.array(by[arm]), CLIP)
        c = ACCENT if arm in STEER else INK
        ax.scatter(v, i + rng.uniform(-0.22, 0.22, len(v)), s=4, color=c, alpha=0.55, lw=0, zorder=2)
        m = min(float(np.mean(by[arm])), CLIP)
        ax.plot([m, m], [i - 0.33, i + 0.33], color=c, lw=2.2, zorder=3)
    ax.set_yticks(range(len(ORDER)), [LABEL[a] for a in reversed(ORDER)], fontsize=7.5, color=INK)
    ax.set_xlim(0, CLIP)
    ax.set_ylim(-0.7, len(ORDER) - 0.3)
    ax.tick_params(axis="x", labelsize=7, colors=FAINT)
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(FAINT)
    ax.set_xlabel("chat-assistant habits per 1,000 words (each dot one reply, Qwen3-8B)", fontsize=7.5, color=FAINT)
    fig.text(0.03, 0.93, "MODEL VOICE · EPISODE 3", fontsize=8, color=ACCENT, weight="bold")
    fig.text(
        0.03, 0.875, "A de-slop benchmark: every method on the same 48 prompts", fontsize=12.5, color=INK, weight="bold"
    )
    fig.text(0.965, 0.93, "human band ← 33.6", fontsize=7.5, color=FAINT, ha="right")
    out = HERE / "report" / "figures" / "hero.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".png")
    fig.savefig(tmp, facecolor="#ffffff")
    plt.close(fig)
    img = Image.open(tmp).convert("RGB").resize((1600, 900), Image.Resampling.LANCZOS)
    img.save(out, quality=88, optimize=True)
    tmp.unlink()
    thumb = img.resize((800, 450), Image.Resampling.LANCZOS).crop((40, 0, 760, 450)).resize((720, 480))
    thumb.save(HERE.parent / "shared" / "thumbs" / "deslop-h2h.jpg", quality=84, optimize=True)
    print({a: round(float(np.mean(by[a])), 1) for a in ORDER})


if __name__ == "__main__":
    main()
