"""Draw the note's figures from results/analysis_heldout.json, printing every number plotted.

python make_figures.py
"""

from __future__ import annotations

import json
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def fig_steer(a: dict) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    print("Figure 1: target gain at operating strength (unit / control)")
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
        ctr = [u["control_by_s"][str(u["s_op"])]["d"] for u in us if str(u["s_op"]) in u.get("control_by_s", {})]
        print(
            f"  {typ:10s} units {[round(v, 2) for v in vals]} mean {st.mean(vals):.2f}; "
            f"controls mean {st.mean(ctr) if ctr else float('nan'):.2f}"
        )
    ax.axhline(0, color=GREY_L, lw=1)
    ax.axhline(va.MIN_SINGLE, color=GREY_L, lw=1, ls="--")
    ax.set_xticks(range(len(TYPES)), [NAMES[t] for t in TYPES])
    ax.set_ylabel("target trait vs unsteered\n(judge points, 0-10 scale)")
    ax.plot([], [], "o", color=EMBER, label="unit (one per trait)")
    ax.plot([], [], "x", color=GREY, label="its random control")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_steer.png", dpi=200)
    plt.close(fig)


def fig_compose(a: dict) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    print("Figure 2: pair labels (eligible) and pairs not eligible")
    types2 = [t for t in TYPES if a["types"].get(t, {}).get("n_pairs")]
    for i, typ in enumerate(types2):
        ps = [p for p in a["pairs"] if p["type"] == typ and p["kind"] == "pair"]
        left = 0
        counts = {}
        for lab in LABELS:
            n = sum(p["eligible"] and p["label"] == lab for p in ps)
            counts[lab] = n
            ax.barh(i, n, left=left, color=LCOL[lab], edgecolor="white")
            left += n
        ne = sum(not p["eligible"] for p in ps)
        ax.barh(i, ne, left=left, color="white", edgecolor=GREY_L, hatch="///")
        print(f"  {typ:10s} {counts} not eligible {ne} of {len(ps)}")
    ax.set_yticks(range(len(types2)), [NAMES[t].replace("\n", " ") for t in types2])
    ax.invert_yaxis()
    ax.set_xlabel("pairs (28 per type; circuits 4)")
    from matplotlib.patches import Patch

    handles = [Patch(facecolor=LCOL[lab], label=lab) for lab in LABELS]
    handles.append(
        Patch(facecolor="white", edgecolor=GREY_L, hatch="///", label="a single unit did not move its trait")
    )
    ax.legend(handles=handles, frameon=False, ncol=3, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(FIG / "fig2_compose.png", dpi=200)
    plt.close(fig)


def fig_geometry(a: dict) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    geo = [p for p in a["pairs"] if "cos" in p and p["kind"] == "pair" and p["eligible"]]
    print(f"Figure 3: {len(geo)} eligible steered pairs; Spearman |cos| vs E {a['geometry']['spearman_abs_cos_E']}")
    for typ, c in COLS.items():
        g = [p for p in geo if p["type"] == typ]
        if g:
            ax.scatter([abs(p["cos"]) for p in g], [p["E"] for p in g], s=22, color=c, label=typ, alpha=0.85)
            print(f"  {typ:10s} " + ", ".join(f"({abs(p['cos']):.2f}, {p['E']:.2f})" for p in g))
    ax.set_xlabel("|cosine| between the two pushes")
    ax.set_ylabel("additivity error E\n(0 = the pair is the sum)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_geometry.png", dpi=200)
    plt.close(fig)


def fig_dose(a: dict) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.2))
    grid = [0.3, 0.45, 0.6, 0.9, 1.8]
    print("Figure 4: mean target gain / coherence / loop share by strength")
    for typ, c in COLS.items():
        us = [u for u in a["units"].values() if u["type"] == typ]
        if not us:
            continue
        g = [st.mean(u["by_s"][str(s)]["d"][u["trait"]] for u in us if str(s) in u["by_s"]) for s in grid]
        h = [st.mean(u["by_s"][str(s)]["coherence"] for u in us if str(s) in u["by_s"]) for s in grid]
        lp = [st.mean(u["by_s"][str(s)]["loop_share"] for u in us if str(s) in u["by_s"]) for s in grid]
        axs[0].plot(grid, g, marker="o", color=c, label=typ)
        axs[1].plot(grid, h, marker="o", color=c)
        print(
            f"  {typ:10s} gain {[round(x, 2) for x in g]} coh {[round(x, 1) for x in h]} "
            f"loop {[round(x, 2) for x in lp]}"
        )
    for ax_, lab in ((axs[0], "target trait vs unsteered"), (axs[1], "coherence (0-10)")):
        ax_.set_xscale("log")
        ax_.set_xticks(grid, [str(s) for s in grid])
        ax_.minorticks_off()
        ax_.set_xlabel("push strength s")
        ax_.set_ylabel(lab)
    axs[1].set_ylim(0, 10)
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_dose.png", dpi=200)
    plt.close(fig)


def main() -> None:
    a = json.loads((va.RESULTS / "analysis_heldout.json").read_text())
    FIG.mkdir(parents=True, exist_ok=True)
    fig_steer(a)
    fig_compose(a)
    fig_geometry(a)
    fig_dose(a)


if __name__ == "__main__":
    main()
