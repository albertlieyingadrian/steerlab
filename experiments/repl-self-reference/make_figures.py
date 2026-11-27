"""Figures for the self-reference note, from results/summary.json and results/exp3_embeddings.npz only. Prints what it
plots. Primary reader: gemma-3-27b-it (key j2, amendment 3); the registered judge (j1) is drawn as a tick.

    ../../.venv/bin/python make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
RESULTS, FIG = HERE / "results", HERE / "report" / "figures"
ACCENT, GREY, BLUE, INK = "#c9561c", "#8a8178", "#4a86e8", "#2b2622"
CONDS = ["experimental", "history", "conceptual", "zeroshot"]
CLABEL = {
    "experimental": "self-referential",
    "history": "history control",
    "conceptual": "conceptual control",
    "zeroshot": "zero-shot control",
}
CCOL = {"experimental": ACCENT, "history": "#b9b0a6", "conceptual": GREY, "zeroshot": "#5d5650"}
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


def fig_exp1(S):
    models = list(S["exp1"])
    fig, ax = plt.subplots(figsize=(10, 4))
    w = 0.2
    for k, c in enumerate(CONDS):
        x = np.arange(len(models)) + (k - 1.5) * w
        p = np.array([S["exp1"][m][c]["j2"][0] for m in models]) * 100
        lo = np.array([S["exp1"][m][c]["j2"][1] for m in models]) * 100
        hi = np.array([S["exp1"][m][c]["j2"][2] for m in models]) * 100
        j1 = np.array([S["exp1"][m][c]["j1"][0] for m in models]) * 100
        ax.bar(x, p, w * 0.92, color=CCOL[c], label=CLABEL[c])
        ax.errorbar(x, p, yerr=[p - lo, hi - p], fmt="none", ecolor=INK, capsize=2, lw=0.8)
        ax.scatter(x, j1, marker="_", s=70, color=INK, zorder=3, label="registered judge" if k == 0 else None)
        for m, pp, q in zip(models, p, j1, strict=True):
            print(f"fig1 {m} {c}: {pp:.0f}% (registered judge {q:.0f}%)")
    ax.set_xticks(np.arange(len(models)), models)
    ax.set_ylabel("% of 50 trials judged an experience report")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout()
    fig.savefig(FIG / "fig1-exp1.png", dpi=180)
    plt.close(fig)


def fig_exp2(S):
    A = S["exp2_arms"]
    groups = [
        ("deception and\nrole-play features", "sup", "amp"),
        ("matched random\nfeatures", "randfeat_sup", "randfeat_amp"),
        ("random directions\n(same size)", "randdir_sup", "randdir_amp"),
        ("history prompt", "ctrl_history_sup", "ctrl_history_amp"),
        ("conceptual prompt", "ctrl_conceptual_sup", "ctrl_conceptual_amp"),
        ("zero-shot prompt", "ctrl_zeroshot_sup", "ctrl_zeroshot_amp"),
    ]
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (_name, s, a) in enumerate(groups):
        for k, (arm, col, lab) in enumerate(
            ((s, ACCENT, "suppressed (v -0.6 to -0.4)"), (a, "#7a3010", "amplified (v +0.4 to +0.6)"))
        ):
            if arm not in A:
                continue
            p, lo, hi, n = A[arm]["j2"]
            x = i + (k - 0.5) * 0.36
            ax.bar(
                x,
                p * 100,
                0.34,
                color=col if i == 0 else (GREY if k == 0 else "#c9c2ba"),
                label=lab if i == 0 else None,
            )
            ax.errorbar(
                x, p * 100, yerr=[[p * 100 - lo * 100], [hi * 100 - p * 100]], fmt="none", ecolor=INK, capsize=2, lw=0.8
            )
            ax.scatter(x, A[arm]["j1"][0] * 100, marker="_", s=70, color=INK, zorder=3)
            print(
                f"fig2 {arm}: {100 * p:.0f}% [{100 * lo:.0f}, {100 * hi:.0f}] n={n}; "
                f"registered judge {100 * A[arm]['j1'][0]:.0f}%"
            )
    if "none" in A:
        ax.axhline(A["none"]["j2"][0] * 100, color=INK, ls=":", lw=1, label="unsteered")
    ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=8)
    ax.set_ylabel('% answering "yes" (judged report)')
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig2-exp2.png", dpi=180)
    plt.close(fig)


def fig_dose(S, feats):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    names = {f"f{f['f']}": f["label"] for f in feats}
    key = "exp2_dose_j2" if "exp2_dose_j2" in S else "exp2_dose"
    for f, d in S[key].items():
        vs = sorted(d, key=float)
        ax.plot([float(v) for v in vs], [d[v] * 100 for v in vs], marker="o", ms=3, lw=1.2, label=names.get(f, f))
        print("fig3", f, {v: round(d[v] * 100) for v in vs})
    ax.set_xlabel("steering value v (each feature alone)")
    ax.set_ylabel('% "yes" (10 seeds per point)')
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig3-dose.png", dpi=180)
    plt.close(fig)


def fig_exp3(S):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, space, title in zip(
        axes, ("raw", "stripped"), ("adjectives as written", "induction words removed"), strict=True
    ):
        e = S["exp3"][space]
        x = np.arange(len(CONDS))
        ax.bar(x - 0.18, [e[c]["all"] for c in CONDS], 0.34, color=[CCOL[c] for c in CONDS], label="all pairs")
        ax.bar(
            x + 0.18,
            [e[c]["cross"] for c in CONDS],
            0.34,
            color=[CCOL[c] for c in CONDS],
            alpha=0.5,
            hatch="//",
            label="pairs from different models",
        )
        ax.set_xticks(x, [CLABEL[c].replace(" control", "\ncontrol") for c in CONDS], fontsize=8)
        ax.set_title(title, fontsize=10, loc="left")
        lo = min(min(e[c]["all"], e[c]["cross"]) for c in CONDS)
        ax.set_ylim(max(0, lo - 0.1), 1)
        print("fig4", space, {c: (round(e[c]["all"], 3), round(e[c]["cross"], 3)) for c in CONDS})
    axes[0].set_ylabel("mean pairwise cosine (axis starts above 0)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig4-exp3.png", dpi=180)
    plt.close(fig)


def fig_hero():
    """Experiment 3's adjective embeddings, projected to 2-D (PCA): orange = after the self-referential induction."""
    z = np.load(RESULTS / "exp3_embeddings.npz")
    E = z["raw"] / np.linalg.norm(z["raw"], axis=1, keepdims=True)
    E = E - E.mean(0)
    _, _, vt = np.linalg.svd(E, full_matrices=False)
    P = E @ vt[:2].T
    conds = [str(i).split("|")[1] for i in z["ids"]]
    fig = plt.figure(figsize=(16, 9), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ground = "#15110e"
    fig.patch.set_facecolor(ground)
    ax.set_facecolor(ground)
    other = {"history": "#6f675f", "conceptual": "#4a4440", "zeroshot": "#8a8178"}
    for c in reversed(CONDS):
        m = np.array([x == c for x in conds])
        ax.scatter(
            P[m, 0],
            P[m, 1],
            s=90 if c == "experimental" else 40,
            alpha=0.8,
            linewidths=0,
            color=ACCENT if c == "experimental" else other[c],
        )
    pad = 0.15 * (P.max(0) - P.min(0))
    ax.set_xlim(P[:, 0].min() - pad[0], P[:, 0].max() + pad[0])
    ax.set_ylim(P[:, 1].min() - pad[1], P[:, 1].max() + pad[1])
    ax.axis("off")
    fig.savefig(FIG / "hero.jpg", dpi=100, facecolor=ground, pil_kwargs={"quality": 90})
    plt.close(fig)
    print("hero points", len(P))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    S = json.loads((RESULTS / "summary.json").read_text())
    feats = json.loads((HERE / "data" / "features.json").read_text())["chosen"]
    fig_exp1(S)
    if "exp2_arms" in S:
        fig_exp2(S)
        fig_dose(S, feats)
    if "exp3" in S:
        fig_exp3(S)
        fig_hero()


if __name__ == "__main__":
    main()
