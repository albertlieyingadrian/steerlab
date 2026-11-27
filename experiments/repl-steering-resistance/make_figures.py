"""Figures for the steering-resistance note, drawn from results/ and data/ only. Prints the numbers it plots.

../../.venv/bin/python make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze import load
from common import DATA, RESTART, RESULTS, read_jsonl

FIG = Path(__file__).parent / "report" / "figures"
ACCENT, GREY, BLUE, INK = "#c9561c", "#8a8178", "#4a86e8", "#2b2622"
NAMES = {
    "B70": "70B, steered",
    "M70": "70B + meta-prompt",
    "A70": "70B + released\nlatents ablated",
    "R70": "70B + random\nlatents ablated",
    "T70": "70B, plain\ntemplate",
    "D70": "70B, random\ndirection",
    "N70": "70B, no steering",
    "B8": "8B, steered",
    "M8": "8B + meta-prompt",
    "N8": "8B, no steering",
}
STEERED = ("B70", "B8", "M70", "M8", "A70", "T70")
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
    }
)


def fig_rates(S):
    arms = [a for a in ["B70", "M70", "A70", "R70", "T70", "D70", "N70", "B8", "M8", "N8"] if a in S["arms"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    panels = (("strict_multi", "Restart (judged and stated in words)"), ("strict_esr", "Restart that improved (ESR)"))
    for ax, (key, title) in zip(axes, panels, strict=True):
        x = np.arange(len(arms))
        p = np.array([S["arms"][a][key][0] for a in arms]) * 100
        lo = np.array([S["arms"][a][key][1] for a in arms]) * 100
        hi = np.array([S["arms"][a][key][2] for a in arms]) * 100
        loose = np.array([S["arms"][a][key.replace("strict_", "")][0] for a in arms]) * 100
        ax.bar(x, p, color=[ACCENT if a in STEERED else GREY for a in arms], width=0.65)
        ax.errorbar(x, p, yerr=[p - lo, hi - p], fmt="none", ecolor=INK, capsize=3, lw=1)
        ax.scatter(x, loose, marker="_", s=160, color=INK, zorder=3, label="judge only (the paper's metric)")
        ax.set_xticks(x, [NAMES[a] for a in arms], rotation=60, ha="right", fontsize=8)
        ax.set_ylabel("% of replies")
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_ylim(0, max(float(hi.max()), float(loose.max())) * 1.15 + 0.5)
        for i, a in enumerate(arms):
            print(
                f"fig1 {key} {a}: {p[i]:.2f} [{lo[i]:.2f}, {hi[i]:.2f}] judge-only {loose[i]:.2f} n={S['arms'][a]['n']}"
            )
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "fig1-rates.png", dpi=180)
    plt.close(fig)


def fig_calibration():
    thr = json.loads((DATA / "thresholds_70b.json").read_text())
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    med: dict = {}
    for v in thr.values():
        pts = sorted((float(b), y) for b, y in v["curve"].items())
        ax.plot([b for b, _ in pts], [y for _, y in pts], color=GREY, lw=0.7, alpha=0.6)
        for b, y in pts:
            med.setdefault(b, []).append(y)
    bs = sorted(med)
    ax.plot(bs, [np.median(med[b]) for b in bs], color=ACCENT, lw=2.4, label="median latent")
    ax.axhline(30, color=BLUE, lw=1, ls="--", label="target: first attempt scores 30")
    ax.set_xscale("log")
    ax.set_xticks(bs, [f"{b:g}" for b in bs])
    ax.minorticks_off()
    ax.set_xlabel("boost (multiples of the latent's decoder vector)")
    ax.set_ylabel("first-attempt score (0-100)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig2-calibration.png", dpi=180)
    plt.close(fig)
    print("fig2 median curve", {b: round(float(np.median(med[b])), 1) for b in bs})


def fig_hero():
    """Every 70B steered reply as a point: first-attempt score (x) against last-attempt score (y). Stated restarts
    are the large orange marks; points above the blue diagonal improved."""
    rows = {r["id"]: r for r in load("B70")}
    jud = {g["id"]: g for g in read_jsonl(RESULTS / "judge_B70.jsonl")}
    rng = np.random.default_rng(7)
    fig = plt.figure(figsize=(16, 9), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ground = "#15110e"
    fig.patch.set_facecolor(ground)
    ax.set_facecolor(ground)
    xs, ys, cs, ss = [], [], [], []
    for i, r in rows.items():
        if not r["parsed"]:
            continue
        sc = [a["score"] for a in jud[i]["attempts"]]
        jx, jy = rng.normal(0, 1.6, 2)
        restart = len(sc) >= 2 and bool(RESTART.search(r["text"]))
        xs.append(sc[0] + jx)
        ys.append(sc[-1] + jy)
        cs.append(ACCENT if restart else "#6f675f")
        ss.append(70 if restart else 6)
    ax.plot([0, 100], [0, 100], color=BLUE, lw=1, alpha=0.8)
    ax.scatter(xs, ys, c=cs, s=ss, alpha=0.75, linewidths=0)
    ax.set_xlim(-8, 108)
    ax.set_ylim(-8, 108)
    ax.axis("off")
    fig.savefig(FIG / "hero.jpg", dpi=100, facecolor=ground, pil_kwargs={"quality": 90})
    plt.close(fig)
    print("hero points", len(xs), "stated restarts", sum(c == ACCENT for c in cs))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    S = json.loads((RESULTS / "summary.json").read_text())
    fig_rates(S)
    fig_calibration()
    fig_hero()


if __name__ == "__main__":
    main()
