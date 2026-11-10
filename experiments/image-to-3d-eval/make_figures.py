"""Draw the note's figures from results/, printing every number plotted.

    python experiments/image-to-3d-eval/make_figures.py

agreement.png  for 400 geometry and 400 appearance pairs from benchmark.ai's votes, how often each scorer prefers the
               side most of four voters chose (each of the five left out in turn, a 2-2 split counting half), with 95%
               bootstrap intervals; the grey band is how often the left-out voter agrees with the same four.
ranks.png      Spearman's rho between each scorer's ranking of the 13 models (from the 400 sampled pairs) and the
               ranking from all 262,668 human votes, per question, with 95% intervals from resampling models.
leaderboard.png  the GSO results: Chamfer distance, LPIPS on new views, CLIP similarity and the judges' Elo for each
               system, with 95% intervals over the 40 objects (Elo: over battles).
speed.png      seconds per object against peak GPU memory for each model on one L40S, with Idol's published claim.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
SCORERS = {
    "clip": "CLIP similarity",
    "dino": "DINOv2 similarity",
    "gpt-5.4-mini:final": "gpt-5.4-mini judge",
    "gemini-3.6-flash:final": "gemini-3.6-flash judge",
    "both judges:final": "both judges, averaged",
}
SYSTEMS = {"idol": "Idol", "trellis_mesh": "TRELLIS mesh", "trellis_splats": "TRELLIS splats", "sf3d": "Stable Fast 3D",
           "triposr": "TripoSR"}  # fmt: skip


def agreement() -> None:
    t = pd.read_csv(RESULTS / "agreement_human.csv")
    t = t[t.subset == "all"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, q in zip(axes, ("geometry", "appearance"), strict=True):
        sub = t[t.question == q].set_index("scorer")
        sub = sub.rename(columns={"agree4": "a", "lo4": "l", "hi4": "h"})
        ceil = sub.loc["one human (ceiling)"]
        ax.axvspan(ceil.l, ceil.h, color=LIGHT, alpha=0.6, lw=0)
        ax.axvline(ceil.a, color=GREY, lw=1, ls="--")
        ax.axvline(0.5, color=MUTED, lw=0.8, ls=":")
        for k, (key, label) in enumerate(SCORERS.items()):
            r = sub.loc[key]
            colour = EMBER if "judge" in label and "both" not in label else GREY
            ax.errorbar(r.a, k, xerr=[[r.a - r.l], [r.h - r.a]], fmt="o", color=colour, capsize=3)
            print(f"agreement {q:10s} {label:24s} {r.a:.3f} [{r.l:.3f}, {r.h:.3f}]")
        print(f"agreement {q:10s} one voter vs other four  {ceil.a:.3f} [{ceil.l:.3f}, {ceil.h:.3f}]")
        ax.set_yticks(range(len(SCORERS)), list(SCORERS.values()))
        ax.set_ylim(len(SCORERS) - 0.4, -1.1)  # first scorer on top; room above for the band's label
        ax.set_xlim(0.45, 0.8)
        ax.set_title(f"{q.capitalize()} question (400 pairs)", fontsize=10, loc="left")
        ax.set_xlabel("agreement with the majority of four voters")
        ax.text(0.503, -0.75, "chance", fontsize=8, color=MUTED, va="center")
        ax.text(ceil.h + 0.004, -0.75, "one voter vs the other four", fontsize=8, color=GREY, va="center")
    fig.tight_layout()
    fig.savefig(FIGURES / "agreement.png")
    plt.close(fig)


def ranks() -> None:
    r = pd.read_csv(RESULTS / "rank_human.csv")
    show = [("clip", "CLIP", GREY), ("dino", "DINOv2", GREY), ("gpt-5.4-mini:final", "gpt-5.4\nmini judge", EMBER),
            ("gemini-3.6-flash:final", "gemini-3.6\nflash judge", EMBER)]  # fmt: skip
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, q in zip(axes, ("geometry", "appearance"), strict=True):
        sub = r[r.question == q].set_index("scorer")
        for k, (key, label, colour) in enumerate(show):
            rho, lo, hi = sub.loc[key, ["spearman", "lo", "hi"]]
            ax.bar(k, rho, color=colour, width=0.6)
            ax.errorbar(k, rho, yerr=[[rho - lo], [hi - rho]], color=INK, capsize=3)
            print(f"rank {q:10s} {label.replace(chr(10), ' '):22s} rho {rho:.3f} [{lo:.3f}, {hi:.3f}]")
        own = sub.loc["humans, these 400 pairs only", "spearman"]
        ax.axhline(own, color=GREY, ls="--", lw=1)
        ax.text(-0.4, own + 0.02, "rho of the 400 sampled human votes", fontsize=8, color=GREY)
        print(f"rank {q:10s} humans on the sample rho {own:.3f}")
        ax.set_xticks(range(len(show)), [s[1] for s in show])
        ax.set_ylim(0, 1.08)
        ax.set_title(f"{q.capitalize()} question, 13 models", fontsize=10, loc="left")
    axes[0].set_ylabel("Spearman's rho with the ranking\nfrom all human votes")
    fig.tight_layout()
    fig.savefig(FIGURES / "ranks.png")
    plt.close(fig)


def leaderboard() -> None:
    b = pd.read_csv(RESULTS / "gso_leaderboard.csv").set_index("system")
    b = b.loc[[s for s in SYSTEMS if s in b.index]]
    panels = [("chamfer", "Chamfer distance\n(lower is better)"), ("lpips", "LPIPS on 7 new views\n(lower is better)"),
              ("clip", "CLIP similarity\nto the input"), ("elo", "Judges' Elo\n(both judges)")]  # fmt: skip
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.9), sharey=True)
    for ax, (col, title) in zip(axes, panels, strict=True):
        if col not in b:
            continue
        for k, s in enumerate(b.index):
            v, lo, hi = b.loc[s, col], b.loc[s, f"{col}_lo"], b.loc[s, f"{col}_hi"]
            ax.errorbar(v, k, xerr=[[v - lo], [hi - v]], fmt="o", color=EMBER if s == "idol" else GREY, capsize=3)
            print(f"leaderboard {col:8s} {SYSTEMS[s]:16s} {v:.4f} [{lo:.4f}, {hi:.4f}]")
        ax.set_title(title, fontsize=9, loc="left")
    axes[0].set_yticks(range(len(b)), [SYSTEMS[s] for s in b.index])
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIGURES / "leaderboard.png")
    plt.close(fig)


def speed() -> None:
    b = pd.read_csv(RESULTS / "gso_leaderboard.csv").set_index("system")
    b = b.loc[[s for s in SYSTEMS if s in b.index and s != "trellis_splats"]]
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    for s in b.index:
        x, y = b.loc[s, "seconds_median"], b.loc[s, "peak_reserved_gb"]
        name = "TRELLIS (mesh and splats)" if s == "trellis_mesh" else SYSTEMS[s]
        ax.scatter(x, y, color=EMBER if s == "idol" else GREY, s=40, zorder=3)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
        print(f"speed {name:26s} {x:.2f} s median, {b.loc[s, 'seconds_p90']:.2f} s p90, {y:.2f} GB reserved")
    ax.set_xlim(left=0, right=max(b.seconds_median) * 1.35)
    ax.set_ylim(bottom=0, top=max(max(b.peak_reserved_gb), 7.2) * 1.2)
    ax.axhline(7.2, color=EMBER, ls=":", lw=1)
    ax.text(ax.get_xlim()[1] * 0.98, 7.4, "Idol's stated peak, 7.2 GB", ha="right", fontsize=8, color=EMBER)
    ax.set_xlabel("seconds per object (median, one L40S, model loaded)")
    ax.set_ylabel("peak GPU memory held (GB)")
    fig.tight_layout()
    fig.savefig(FIGURES / "speed.png")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    agreement()
    ranks()
    if (RESULTS / "gso_leaderboard.csv").exists():
        leaderboard()
        speed()
    print(json.dumps({"figures": sorted(p.name for p in FIGURES.glob("*.png"))}))


if __name__ == "__main__":
    main()
