"""Draw the note's figures from both models' held-out analyses (results/qwen3-8b/ and results/), printing every number
plotted. Qwen3-8B is the main model; Gemma-2-2B-it is the cross-model check under the same judge.

    python make_figures.py
"""

from __future__ import annotations

import json
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import va

FIG = va.HERE / "report" / "figures"
EMBER, EMBER_L, GREY, GREY_L, INK = "#d26f2a", "#f2b48e", "#6f675f", "#c9c0b5", "#1d1a17"
plt.rcParams.update(
    {"font.family": "sans-serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
)
TRAITS = list(va.TRAITS)
TYPES = ["prompt", "direction", "sae", "neuron", "community", "circuit"]
NAMES = {
    "prompt": "Prompt\n(baseline)",
    "direction": "Direction\n(baseline)",
    "sae": "SAE\nfeature",
    "neuron": "Single\nneuron",
    "community": "Neuron\ncommunity",
    "circuit": "Circuit\n(transcoders)",
}
LABELS = ["additive", "dominates", "interferes", "emergent", "breaks"]
LCOL = {"additive": EMBER, "dominates": EMBER_L, "interferes": "#8f8a84", "emergent": "#8c6bb1", "breaks": "#3b3632"}
COLS = {"direction": EMBER, "sae": "#8c6bb1", "neuron": "#8f8a84", "community": INK, "circuit": "#59a14f"}
MODELS = [
    ("Qwen3-8B", va.SHARED / "qwen3-8b" / "analysis_heldout.json"),
    ("Gemma-2-2B-it", va.SHARED / "analysis_heldout.json"),
]


def fig_steer(runs) -> None:
    fig, axs = plt.subplots(2, 1, figsize=(8.4, 6.4), sharey=True)
    print("Figure 1: target gain at operating strength (unit / control)")
    for ax, (name, a) in zip(axs, runs):
        for i, typ in enumerate(TYPES):
            us = sorted(
                (u for u in a["units"].values() if u["type"] == typ and "target_op" in u),
                key=lambda u: TRAITS.index(u["trait"]),
            )
            for k, u in enumerate(us):
                x = i + (k - 3.5) * 0.08
                ctrl = u.get("control_by_s", {}).get(str(u["s_op"]))
                if ctrl:
                    ax.plot([x], [ctrl["d"]], marker="x", color=GREY, ms=5, ls="none")
                ax.plot([x], [u["target_op"]], marker="o", color=EMBER, ms=6, ls="none")
            vals = [u["target_op"] for u in us]
            print(
                f"  {name:13s} {typ:10s} {[round(v, 2) for v in vals]} moving {sum(v >= va.MIN_SINGLE for v in vals)}"
            )
        ax.axhline(0, color=GREY_L, lw=1)
        ax.axhline(va.MIN_SINGLE, color=GREY_L, lw=1, ls="--")
        ax.set_xticks(range(len(TYPES)), [NAMES[t] for t in TYPES])
        ax.set_title(name, loc="left", fontsize=10, fontweight="bold")
        ax.set_ylabel("target voice vs unsteered\n(judge points)")
    axs[0].plot([], [], "o", color=EMBER, label="unit (one per voice)")
    axs[0].plot([], [], "x", color=GREY, label="its random twin")
    axs[0].legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_steer.png", dpi=200)
    plt.close(fig)


def fig_compose(runs) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    print("Figure 2: pair labels (scorable) and pairs not scorable")
    for ax, (name, a) in zip(axs, runs):
        for i, typ in enumerate(TYPES):
            ps = [p for p in a["pairs"] if p["type"] == typ and p["kind"] == "pair"]
            left, counts = 0, {}
            for lab in LABELS:
                n = sum(p["eligible"] and p["label"] == lab for p in ps)
                counts[lab] = n
                ax.barh(i, n, left=left, color=LCOL[lab], edgecolor="white")
                left += n
            ne = sum(not p["eligible"] for p in ps)
            ax.barh(i, ne, left=left, color="white", edgecolor=GREY_L, hatch="///")
            print(f"  {name:13s} {typ:10s} {counts} not scorable {ne} of {len(ps)}")
        ax.set_yticks(range(len(TYPES)), [NAMES[t].replace("\n", " ") for t in TYPES])
        ax.invert_yaxis()
        ax.set_xlabel("pairs (28 per kind; circuits 4)")
        ax.set_title(name, loc="left", fontsize=10, fontweight="bold")
    handles = [Patch(facecolor=LCOL[lab], label=lab) for lab in LABELS]
    handles.append(
        Patch(facecolor="white", edgecolor=GREY_L, hatch="///", label="a single unit did not move its voice")
    )
    fig.legend(handles=handles, frameon=False, ncol=6, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "fig2_compose.png", dpi=200)
    plt.close(fig)


def fig_geometry(runs) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    for (name, a), marker in zip(runs, ("o", "^")):
        geo = [p for p in a["pairs"] if "cos" in p and p["kind"] == "pair" and p["eligible"]]
        print(
            f"Figure 3 {name}: {len(geo)} scorable pairs; Spearman all pairs {a['geometry']['spearman_abs_cos_E_all']}"
        )
        for typ, c in COLS.items():
            g = [p for p in geo if p["type"] == typ]
            if g:
                ax.scatter([abs(p["cos"]) for p in g], [p["E"] for p in g], s=22, color=c, marker=marker, alpha=0.85)
    for typ, c in COLS.items():
        ax.scatter([], [], color=c, label=typ, s=22)
    ax.scatter([], [], color=GREY, marker="o", label="Qwen3-8B")
    ax.scatter([], [], color=GREY, marker="^", label="Gemma-2-2B-it")
    ax.set_xlabel("|cosine| between the two pushes")
    ax.set_ylabel("additivity error E\n(0 = the pair is the sum)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_geometry.png", dpi=200)
    plt.close(fig)


def fig_dose(runs) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(8.4, 5.6), sharex=True)
    grid = [0.3, 0.45, 0.6, 0.9, 1.8]
    print("Figure 4: mean target gain / coherence by strength")
    for row, (name, a) in zip(axs, runs):
        for typ, c in COLS.items():
            us = [u for u in a["units"].values() if u["type"] == typ]
            g = [st.mean(u["by_s"][str(s)]["d"][u["trait"]] for u in us if str(s) in u["by_s"]) for s in grid]
            h = [st.mean(u["by_s"][str(s)]["coherence"] for u in us if str(s) in u["by_s"]) for s in grid]
            row[0].plot(grid, g, marker="o", color=c, label=typ)
            row[1].plot(grid, h, marker="o", color=c)
            print(f"  {name:13s} {typ:10s} gain {[round(x, 2) for x in g]} coh {[round(x, 1) for x in h]}")
        row[0].set_ylabel(f"{name}\ntarget voice vs unsteered")
        row[1].set_ylabel("coherence (0-10)")
        row[1].set_ylim(0, 10)
    for ax in axs.flat:
        ax.set_xscale("log")
        ax.set_xticks(grid, [str(s) for s in grid])
        ax.minorticks_off()
    for ax in axs[1]:
        ax.set_xlabel("push strength s")
    axs[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_dose.png", dpi=200)
    plt.close(fig)


def main() -> None:
    runs = [(name, json.loads(path.read_text())) for name, path in MODELS]
    FIG.mkdir(parents=True, exist_ok=True)
    fig_steer(runs)
    fig_compose(runs)
    fig_geometry(runs)
    fig_dose(runs)


if __name__ == "__main__":
    main()
