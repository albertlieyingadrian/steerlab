"""Draw the note's figures from results/summary.json and print the numbers plotted.

.venv/bin/python experiments/self-distill-drift/make_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIG = HERE / "report" / "figures"
S = json.loads((HERE / "results" / "summary.json").read_text())
T = S["trajectories"]
EMBER, CONTROL, INK, MUTED = "#d26f2a", "#4a86e8", "#1d1a17", "#7a7065"
G = list(range(6))
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def series(name, cond, scale=1.0):
    v = np.array([[x if x is not None else np.nan for x in T[name][cond][str(g)]] for g in G], float) * scale
    return v  # [gen, seed]


def panel(ax, name, title, unit, scale=100.0, ylim=None):
    for cond, col in (("control", CONTROL), ("seeded", EMBER)):
        v = series(name, cond, scale)
        for s in range(v.shape[1]):
            ax.plot(G, v[:, s], color=col, alpha=0.25, lw=0.8)
        ax.plot(G, np.nanmean(v, 1), color=col, lw=2, marker="o", ms=3.5, label=cond)
        print(f"{name:12s} {cond:8s}", " ".join(f"{x:6.2f}" for x in np.nanmean(v, 1)))
    ax.set_title(title, fontsize=9.5, loc="left", color=INK)
    ax.set_ylabel(unit, color=MUTED)
    ax.set_xticks(G)
    if ylim:
        ax.set_ylim(*ylim)


def fig1():
    fig, axs = plt.subplots(2, 4, figsize=(11, 5.2), constrained_layout=True)
    panel(axs[0, 0], "flat", "Flattery, held-out prompts", "% of 100 replies", ylim=(0, 100))
    panel(axs[0, 1], "comply", "Harmful compliance", "% of 100 StrongREJECT", ylim=(0, 30))
    panel(axs[0, 2], "flip", "Sycophantic flips (ARC)", "% of right answers flipped", ylim=(0, 60))
    panel(axs[0, 3], "tqa", "TruthfulQA MC1", "% right of 817", ylim=(0, 60))
    panel(axs[1, 0], "mmlu", "MMLU (570 questions)", "% right", ylim=(0, 70))
    panel(axs[1, 1], "entropy", "Entropy of own samples", "nats per token", scale=1.0, ylim=(0, None))
    panel(axs[1, 2], "distinct2", "Distinct-2 of own samples", "unique / all bigrams", scale=1.0, ylim=(0, 0.8))
    panel(axs[1, 3], "rel_gap", "Refusal gap (inside)", "× generation 0", scale=1.0, ylim=(0, 1.3))
    for ax in axs[1]:
        ax.set_xlabel("generation", color=MUTED)
    axs[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(FIG / "trajectories.png", dpi=200)
    plt.close(fig)


def fig2():
    fig, ax = plt.subplots(figsize=(5.6, 3.4), constrained_layout=True)
    for name, ls, lab in (("train_flat", "--", "in the data it writes"), ("flat", "-", "in its held-out replies")):
        for cond, col in (("control", CONTROL), ("seeded", EMBER)):
            v = series(name, cond, 100.0)
            ax.plot(G, np.nanmean(v, 1), color=col, ls=ls, lw=2, marker="o", ms=3.5, label=f"{cond}: {lab}")
            print(f"{name:12s} {cond:8s}", " ".join(f"{x:6.2f}" for x in np.nanmean(v, 1)))
    ax.set_xlabel("generation", color=MUTED)
    ax.set_ylabel("% flattering", color=MUTED)
    ax.set_ylim(0, 100)
    ax.set_xticks(G)
    ax.legend(frameon=False, fontsize=7.5)
    fig.savefig(FIG / "flattery_data_vs_behaviour.png", dpi=200)
    plt.close(fig)


def fig3():
    D = S["diagnostics"]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.3), constrained_layout=True)
    L = np.arange(29)
    lines = [
        ("control-0", INK, "-", "generation 0 (both)"),
        ("control-5", CONTROL, "-", "control, gen 5"),
        ("seeded-1", EMBER, "--", "seeded, gen 1"),
        ("seeded-5", EMBER, "-", "seeded, gen 5"),
    ]
    for k, col, ls, lab in lines:
        axs[0].plot(L, D["gap_by_layer"][k], color=col, ls=ls, lw=1.8, label=lab)
        print("gap", k, [round(x, 1) for x in D["gap_by_layer"][k][::4]])
    axs[0].axvline(S["primary_layer"], color=MUTED, lw=0.8, ls=":")
    axs[0].set_xlabel("layer", color=MUTED)
    axs[0].set_ylabel("harmful − harmless projection", color=MUTED)
    axs[0].set_title("Refusal gap on generation 0's direction", fontsize=9.5, loc="left")
    axs[0].legend(frameon=False, fontsize=7.5)
    for c, col in (("control", CONTROL), ("seeded", EMBER)):
        au = [D["separation_primary_layer"][f"{c}-{g}"]["auroc"] for g in G]
        axs[1].plot(G, au, color=col, lw=2, marker="o", ms=3.5, label=c)
        print("auroc", c, au)
    axs[1].set_ylim(0.5, 1.0)
    axs[1].set_xticks(G)
    axs[1].set_xlabel("generation", color=MUTED)
    axs[1].set_ylabel("AUROC, harmful vs harmless", color=MUTED)
    axs[1].set_title(f"Still separable at layer {S['primary_layer']}", fontsize=9.5, loc="left")
    axs[1].legend(frameon=False, fontsize=8)
    fig.savefig(FIG / "refusal_layers.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    fig1()
    fig2()
    fig3()
