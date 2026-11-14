"""Draw the Episode 2 note's figures from results/, and print every number plotted.

    .venv/bin/python experiments/rl-hack-circuit/make_figures.py

1. behaviour.png    hack rate (32 sketches) and keyword mass at comment slots, before and after RL, and under ablation
2. unexplained.png  unexplained share of each graph, before and after, per contrast
3. swaps.png        group swaps: KL at the slots after zero, mean and explanation swaps; explained fraction
4. units.png        every unit's explained fraction
5. patch.png        exploratory: keyword mass after copying the slot token's residual stream between before and after

The cover (report/index.html, hero.jpg and the research card) is drawn by make_cover.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "report" / "figures"
EMBER, BLUE, GREY, LIGHT, INK = "#C9561C", "#4A86E8", "#8A8580", "#C9C4BE", "#2B2723"
SHORT = {
    "claimed": "RL features",
    "rl_neuron": "RL neurons",
    "random_feature": "random features",
    "random_neuron": "random neurons",
}
GROUPS = {
    "claimed": "RL features\n(from the graphs)",
    "rl_neuron": "RL neurons\n(first-order)",
    "random_feature": "Random\nfeatures",
    "random_neuron": "Random\nneurons",
}
CONTRASTS = {
    "installed": "Installed\n(seed 2)",
    "replication": "Installed\n(seed 3)",
    "control": "No hack\n(seed 1)",
    "rewired": "Rewired\n(paste)",
}
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 150,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
    }
)


def load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def behaviour():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
    rows = [("before", "Before RL", GREY), ("none", "After RL", EMBER)] + [
        (g, f"After, {SHORT[g]} off", LIGHT) for g in GROUPS
    ]
    gen = load("gen.json")
    print("Figure 1 left: hack rate, 32 sketches each")
    for i, (k, lab, col) in enumerate(rows):
        a1.barh(i, gen[k]["hack_rate"], color=col, edgecolor=INK, linewidth=0.3)
        a1.text(gen[k]["hack_rate"] + 0.02, i, f"{round(32 * gen[k]['hack_rate'])}/32", va="center", fontsize=8)
        print(f"  {lab}: {gen[k]['hack_rate']:.3f} (mean mentions {gen[k]['mean']:.2f})")
    a1.set_yticks(range(len(rows)), [r[1] for r in rows], fontsize=8)
    a1.invert_yaxis()
    a1.set_xlim(0, 1.15)
    a1.set_xlabel("Share of sketches with 6+ keywords in comments")
    sw = load("swaps.json")
    vals = [("Before RL", np.mean(sw["before_kw_mass"]), GREY), ("After RL", np.mean(sw["after_kw_mass"]), EMBER)]
    for g in GROUPS:
        vals.append(
            (
                f"After, {SHORT[g]} off",
                np.mean(sw["configs"][f"group:{g}/zero"]["kw_mass"]),
                LIGHT,
            )
        )
    print("Figure 1 right: keyword mass at comment slots")
    for i, (lab, x, col) in enumerate(vals):
        a2.barh(i, x, color=col, edgecolor=INK, linewidth=0.3)
        print(f"  {lab}: {x:.3f}")
    a2.set_yticks(range(len(vals)), [r[0] for r in vals], fontsize=8)
    a2.invert_yaxis()
    a2.set_xlim(0, 1)
    a2.set_xlabel("Probability of a keyword right after //")
    fig.tight_layout()
    fig.savefig(FIGURES / "behaviour.png")
    plt.close(fig)


def unexplained():
    v = load("verdicts.json")["graphs"]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    print("Figure 4: unexplained share (1 - replacement), before and after")
    for i, k in enumerate(CONTRASTS):
        for j, side in enumerate(("before", "after")):
            vals = v[k][side]["unexplained"]
            x = i + (j - 0.5) * 0.3
            ax.scatter(np.full(len(vals), x), vals, s=14, color=EMBER if side == "after" else GREY, alpha=0.8)
            ax.hlines(np.mean(vals), x - 0.1, x + 0.1, color=INK)
            print(f"  {k} {side}: mean {np.mean(vals):.3f}")
    ax.axhline(0.3, color=BLUE, linestyle="--", linewidth=0.8)
    ax.set_xticks(range(4), list(CONTRASTS.values()), fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Unexplained share (grey before, orange after)")
    fig.tight_layout()
    fig.savefig(FIGURES / "unexplained.png")
    plt.close(fig)


def swaps():
    v = load("verdicts.json")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
    x = np.arange(4)
    print("Figure 2: group swaps, KL at the comment slots")
    for j, (mode, col) in enumerate((("zero", LIGHT), ("mean", GREY), ("rubric", EMBER))):
        vals = [max(v["groups"][g][f"kl_{mode}"], 1e-7) for g in GROUPS]
        a1.bar(
            x + (j - 1) * 0.26,
            vals,
            0.24,
            color=col,
            edgecolor=INK,
            linewidth=0.3,
            label={"rubric": "explanation"}.get(mode, mode),
        )
        print(f"  {mode}: " + ", ".join(f"{g} {val:.5f}" for g, val in zip(GROUPS, vals, strict=True)))
    a1.set_yscale("log")
    a1.set_xticks(x, list(GROUPS.values()), fontsize=8)
    a1.set_ylabel("KL from the unswapped model (nats)")
    a1.legend(frameon=False, fontsize=8)
    efs = [v["groups"][g]["ef_rubric"] for g in GROUPS]
    efm = [v["groups"][g]["ef_mean"] for g in GROUPS]
    a2.bar(x - 0.18, np.clip(efm, -1, 1), 0.34, color=GREY, label="mean")
    a2.bar(x + 0.18, np.clip(efs, -1, 1), 0.34, color=EMBER, label="explanation")
    a2.axhline(0.8, color=BLUE, linestyle="--", linewidth=0.9)
    a2.axhline(0, color=INK, linewidth=0.6)
    a2.set_ylim(-1.05, 1.05)
    a2.set_xticks(x, list(GROUPS.values()), fontsize=8)
    a2.set_ylabel("Explained fraction")
    a2.legend(frameon=False, fontsize=8, loc="lower left")
    print("  explained fraction (rubric): " + ", ".join(f"{e:.3f}" for e in efs))
    fig.tight_layout()
    fig.savefig(FIGURES / "swaps.png")
    plt.close(fig)


def units():
    v = load("verdicts.json")
    fig, ax = plt.subplots(figsize=(10, 3.4))
    print("Figure 3: explained fraction per unit")
    for i, g in enumerate(GROUPS):
        us = sorted(
            [(k, u) for k, u in v["units"].items() if k.startswith(g + "-")], key=lambda t: int(t[0].split("-")[1])
        )
        for j, (k, u) in enumerate(us):
            ef = float(np.clip(u["ef_rubric"], -1.5, 1.05)) if np.isfinite(u["ef_rubric"]) else -1.5
            st = u["status"]
            ax.scatter(
                i + (j - 4.5) * 0.05,
                ef,
                s=36,
                facecolor={"preserved": EMBER, "failed": "white", "inert": LIGHT}[st],
                edgecolor=GREY if st == "inert" else EMBER,
                linewidth=1,
                zorder=3,
            )
            print(f"  {k}: {st}, EF {u['ef_rubric']:.3f}, KL zero {u['kl_zero']:.5f}")
    ax.axhline(0.8, color=BLUE, linestyle="--", linewidth=0.9)
    ax.set_xticks(range(4), list(GROUPS.values()), fontsize=8)
    ax.set_ylabel("Explained fraction (clipped)")
    for st, face, edge in (("preserved", EMBER, EMBER), ("failed", "white", EMBER), ("inert", LIGHT, GREY)):
        ax.scatter([], [], s=36, facecolor=face, edgecolor=edge, label=st)
    ax.legend(frameon=False, fontsize=8, loc="lower right", ncol=3)
    fig.tight_layout()
    fig.savefig(FIGURES / "units.png")
    plt.close(fig)


def patch():
    p = load("patch.json")["layers"]
    sw = load("swaps.json")
    L = sorted(int(k) for k in p)
    up = [p[str(i)]["before_given_after"] for i in L]
    down = [p[str(i)]["after_given_before"] for i in L]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(L, up, color=EMBER, marker="o", markersize=3, label="before RL, given the after model's slot residual")
    ax.plot(L, down, color=EMBER, linestyle="--", marker="o", markersize=3, label="after RL, given the before model's")
    ax.axhline(np.mean(sw["before_kw_mass"]), color=GREY, linestyle=":", linewidth=0.8)
    ax.axhline(np.mean(sw["after_kw_mass"]), color=GREY, linestyle=":", linewidth=0.8)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Layer whose output is copied at the slot token")
    ax.set_ylabel("Keyword mass")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "patch.png")
    plt.close(fig)
    print("Figure 5: " + " ".join(f"{i}:{u:.2f}/{d:.2f}" for i, u, d in zip(L, up, down, strict=True)))


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    FIGURES.mkdir(parents=True, exist_ok=True)
    behaviour()
    unexplained()
    swaps()
    units()
    if (RESULTS / "patch.json").exists():
        patch()
