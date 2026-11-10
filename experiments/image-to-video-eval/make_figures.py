"""Draw the note's figures from results/, printing every number plotted.

    python experiments/image-to-video-eval/make_figures.py

agreement.png  per question (photorealism and integrity: 800 pairs; camera: 400, judges 200), how often each scorer
               prefers the clip most of the remaining voters chose (each voter left out in turn, a tie counting
               half), with 95% bootstrap intervals; the grey band is how often the left-out voter agrees with them.
traps.png      for 25 real generations and four broken copies of each, the share of clips where each scorer prefers
               the broken copy (ties count half): orange at 0.6 or more, dark grey between 0.4 and 0.6, light below.
speed.png      seconds per clip against peak GPU memory for each open model on one H100 80 GB.
leaderboard.png  our four models on the 25 realism stills by the scorers that agreed best with people, with 95%
               intervals over stills (drawn only when results/ours_leaderboard.csv exists).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import RESULTS

FIGURES = HERE / "report" / "figures"
EMBER, GREY, LIGHT, INK, MUTED = "#C9561C", "#6F6A63", "#C8C2B8", "#2B2723", "#8A847C"
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "savefig.dpi": 150,
    }
)
NAMES = {
    "clip": "CLIP similarity",
    "dinov2": "DINOv2 similarity",
    "subject_consistency": "subject consistency",
    "background_consistency": "background consistency",
    "motion_smoothness": "motion smoothness",
    "temporal_flickering": "temporal flickering",
    "dynamic_degree": "dynamic degree (yes/no)",
    "flow": "optical-flow size",
    "aesthetic_quality": "aesthetic quality",
    "imaging_quality": "imaging quality",
    "i2v_subject": "I2V subject",
    "i2v_background": "I2V background",
    "camera_hit": "camera move, VBench yes/no",
    "camera_shift": "camera move, graded shift",
    "gpt-5.4-mini": "gpt-5.4-mini judge",
    "gemini-3.6-flash": "gemini-3.6-flash judge",
}
JUDGES = {"gpt-5.4-mini", "gemini-3.6-flash"}
MODELS = {"wan22_ti2v_5b": "Wan 2.2 TI2V-5B", "ltx_video_097_distilled": "LTX-Video 0.9.7 distilled",
          "cogvideox_5b_i2v": "CogVideoX-5B-I2V", "svd_xt": "SVD-XT"}  # fmt: skip


def agreement() -> None:
    t = pd.read_csv(RESULTS / "agreement_human.csv")
    t = t[t.subset == "all"]
    panels = [("photorealism", "Photorealism (800 pairs)"), ("integrity", "Fewer glitches (800 pairs)"),
              ("camera", "Camera move (400 pairs, judges 200)")]  # fmt: skip
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.8))
    for ax, (q, title) in zip(axes, panels, strict=True):
        sub = t[t.question == q].set_index("scorer")
        keys = [k for k in NAMES if k in sub.index]
        voter = sub.loc["one more voter"]
        ax.axvspan(voter.lo, voter.hi, color=LIGHT, alpha=0.6, lw=0)
        ax.axvline(voter.agree, color=GREY, lw=1, ls="--")
        ax.axvline(0.5, color=MUTED, lw=0.8, ls=":")
        for k, key in enumerate(keys):
            r = sub.loc[key]
            colour = EMBER if key in JUDGES else GREY
            ax.errorbar(r.agree, k, xerr=[[r.agree - r.lo], [r.hi - r.agree]], fmt="o", color=colour, capsize=3)
            print(f"agreement {q:12s} {NAMES[key]:28s} {r.agree:.3f} [{r.lo:.3f}, {r.hi:.3f}] n={r.n}")
        print(f"agreement {q:12s} {'one more voter':28s} {voter.agree:.3f} [{voter.lo:.3f}, {voter.hi:.3f}]")
        ax.set_yticks(range(len(keys)), [NAMES[k] for k in keys])
        ax.set_ylim(len(keys) - 0.4, -1.2)
        ax.set_xlim(0.3, 0.82)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("agreement with the other voters' majority")
        ax.text(0.495, -0.8, "chance", fontsize=8, color=MUTED, va="center", ha="right")
        ax.text(voter.hi + 0.006, -0.8, "one more\nvoter", fontsize=8, color=GREY, va="center")
    fig.tight_layout()
    fig.savefig(FIGURES / "agreement.png")
    plt.close(fig)


def traps() -> None:
    t = pd.read_csv(RESULTS / "traps_result.csv")
    piv = t.pivot(index="scorer", columns="trap", values="prefers_trap")
    n = t.pivot(index="scorer", columns="trap", values="n")
    cols = ["still", "reversed", "shuffled", "blurred"]
    rows = [k for k in NAMES if k in piv.index]
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for i, key in enumerate(rows):
        for j, c in enumerate(cols):
            v = piv.loc[key, c]
            colour = EMBER if v >= 0.6 else LIGHT if v <= 0.4 else GREY  # 0.4 to 0.6: cannot tell them apart
            ax.scatter(j, i, s=60 + 900 * v, color=colour, zorder=3)
            ax.text(j + 0.22, i, f"{v:.2f}", va="center", fontsize=8, color=INK)
            print(f"trap {c:9s} {NAMES[key]:28s} prefers the broken copy {v:.2f} (n={int(n.loc[key, c])})")
    ax.set_xticks(range(len(cols)), ["still\n(input repeated)", "reversed", "frames\nshuffled", "blurred"])
    ax.set_yticks(range(len(rows)), [NAMES[k] for k in rows])
    ax.set_xlim(-0.5, len(cols) - 0.2)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_title("Share of 25 clips where the scorer prefers the broken copy", fontsize=10, loc="left")
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(FIGURES / "traps.png")
    plt.close(fig)


def speed() -> None:
    timing = pd.read_csv(RESULTS / "gen_timing.csv")
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for m, name in MODELS.items():
        sub = timing[timing.model == m]
        x, y = sub.seconds.median(), sub.peak_reserved_gb.max()
        ax.scatter(x, y, color=GREY, s=40, zorder=3)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
        print(f"speed {name:26s} {x:.1f} s median over {len(sub)} clips, {y:.1f} GB, ~${x / 3600 * 3.95:.3f} a clip")
    ax.set_xscale("log")
    ax.set_xlim(3, 400)
    ax.set_ylim(0, 60)
    ax.set_xlabel("seconds per clip (median, one H100 80 GB, model loaded; log scale)")
    ax.set_ylabel("peak GPU memory held (GB)")
    fig.tight_layout()
    fig.savefig(FIGURES / "speed.png")
    plt.close(fig)


def leaderboard() -> None:
    b = pd.read_csv(RESULTS / "ours_leaderboard.csv").set_index("model")
    b = b.loc[[m for m in MODELS if m in b.index]]
    panels = [(c, NAMES.get(c, c)) for c in ("imaging_quality", "aesthetic_quality", "flow", "elo") if c in b]
    fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 2.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (col, title) in zip(axes, panels, strict=True):
        for k, m in enumerate(b.index):
            v, lo, hi = b.loc[m, col], b.loc[m, f"{col}_lo"], b.loc[m, f"{col}_hi"]
            ax.errorbar(v, k, xerr=[[v - lo], [hi - v]], fmt="o", color=GREY, capsize=3)
            print(f"leaderboard {col:18s} {MODELS[m]:26s} {v:.3f} [{lo:.3f}, {hi:.3f}]")
        ax.set_title("judges' Elo" if col == "elo" else title, fontsize=9, loc="left")
    axes[0].set_yticks(range(len(b)), [MODELS[m] for m in b.index])
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIGURES / "leaderboard.png")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    agreement()
    traps()
    speed()
    if (RESULTS / "ours_leaderboard.csv").exists():
        leaderboard()
    print(json.dumps({"figures": sorted(p.name for p in FIGURES.glob("*.png"))}))


if __name__ == "__main__":
    main()
