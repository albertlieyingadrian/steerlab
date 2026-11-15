"""Draw every figure in the looped-transformer note from results/summary.json, and print the numbers each one plots.

    python experiments/looped-transformer/make_figures.py

Figures (report/figures/): doubling.png (a schematic of the two ways to follow a chain, drawn from the rule, no data),
accuracy.png (accuracy by hop count), loops.png (accuracy by loop count, with the ablations), inside.png (digit
frontier by depth), convergence.png (how much the state still changes per loop).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIG = HERE / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
INK, MUTED, EMBER, EMBER_D, BLUE, GREY, LINE = (
    "#1d1a17",
    "#7a7065",
    "#d26f2a",
    "#a85a22",
    "#3f6fb5",
    "#b9b1a7",
    "#e3dbd0",
)
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 200,
    }
)
SEEDS = ["1", "2", "3"]
KS_SHOWN = ["1", "2", "4", "8", "16"]
K_SHADES = {"1": "#f3c9a6", "2": "#eaa46f", "4": EMBER, "8": EMBER_D, "16": "#6e3a15"}


def S():
    return json.loads((HERE / "results" / "summary.json").read_text())


def curve(s, model, k):
    hops = [str(h) for h in range(1, 25)]
    seeds = [sd for sd in SEEDS if sd in s["by_hop"][model][k]["1"]["seeds"]]
    rows = np.array([[s["by_hop"][model][k][h]["seeds"][sd] for h in hops] for sd in seeds])
    return np.arange(1, 25), rows


def fig_doubling():
    """Schematic: 8 assignments in a chain (a=b, b=c, ... h=7). Left: one link per step. Right: pointer doubling,
    each assignment reads the assignment its pointer names and jumps twice as far."""
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.2))
    names = list("abcdefgh")
    for ax, (title, reach) in zip(axs, [("One link per step", lambda t: t + 1), ("Pointer doubling", lambda t: 2**t)]):
        for t in range(4):
            y = 3 - t
            r = reach(t)
            for i, n in enumerate(names):
                known = min(i + r, 8)
                done = known >= 8
                ax.add_patch(
                    plt.Rectangle(
                        (i - 0.38, y - 0.3),
                        0.76,
                        0.6,
                        fc=EMBER if done else "white",
                        ec=EMBER if done else GREY,
                        lw=1.2,
                    )
                )
                label = "7" if done else names[known]
                ax.text(i, y, f"{n}→{label}", ha="center", va="center", fontsize=7.5, color="white" if done else INK)
            ax.text(-0.75, y, f"step {t + 1}", ha="right", va="center", fontsize=8, color=MUTED)
        ax.set_xlim(-1.6, 7.6)
        ax.set_ylim(-0.6, 3.6)
        ax.axis("off")
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    fig.text(
        0.01,
        0.01,
        "Chain a=b, b=c, …, h=7. Each box shows what an assignment points at after a step; filled = it already knows the digit.",
        fontsize=7.5,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / "doubling.png")
    plt.close(fig)
    print("doubling.png: schematic, no data")


def band(ax, x, rows, color, label, ls="-", lw=1.8):
    ax.plot(x, rows.mean(0), color=color, lw=lw, ls=ls, label=label)
    if len(rows) > 1:
        ax.fill_between(x, rows.min(0), rows.max(0), color=color, alpha=0.15, lw=0)


def fig_accuracy(s):
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.6), sharey=True, sharex=True)
    axs = axs.ravel()
    base = np.array([s["shortcut_baseline"][str(h)] for h in range(1, 25)])
    for ax in axs:
        ax.axvspan(0.5, 8.5, color=LINE, alpha=0.5, lw=0)
        ax.text(4.5, 1.03, "training range", ha="center", fontsize=8, color=MUTED)
        ax.plot(np.arange(1, 25), base, color=MUTED, ls=":", lw=1.2, label="shortcut baseline")
        ax.set_xlim(0.5, 24.5)
        ax.set_ylim(0, 1.08)
    for ax in axs[2:]:
        ax.set_xlabel("hops in the query's chain")
    for ax in (axs[0], axs[2]):
        ax.set_ylabel("accuracy (500 items per hop count)")
    ax = axs[0]
    for m, k, color, label, ls in [
        ("standard4", "None", INK, "standard, 4 layers (same parameters)", "--"),
        ("standard10", "None", GREY, "standard, 10 layers (compute of k = 4)", "-"),
        ("looped", "8", EMBER, "looped, random k in training, run at k = 8", "-"),
        ("looped_fixed4", "8", MUTED, "looped, k = 4 in training, run at k = 8", "-."),
        ("looped_noinject", "8", BLUE, "looped, no input re-added, run at k = 8", "-"),
    ]:
        x, r = curve(s, m, k)
        ax.plot(x, r.mean(0), color=color, lw=1.8, ls=ls, label=label)
    ax.set_title("A. Every model, mean of 3 seeds", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    for i, sd in enumerate(SEEDS):
        ax = axs[i + 1]
        for k in KS_SHOWN:
            row = [s["by_hop"]["looped"][k][str(h)]["seeds"][sd] for h in range(1, 25)]
            ax.plot(np.arange(1, 25), row, color=K_SHADES[k], lw=1.8, label=f"k = {k}")
        row = [s["by_hop"]["standard4"]["None"][str(h)]["seeds"][sd] for h in range(1, 25)]
        ax.plot(np.arange(1, 25), row, color=INK, lw=1, ls="--", label="standard, 4 layers")
        ax.set_title(f"{'BCD'[i]}. Looped (random k), seed {sd}, run with k loops", loc="left", fontsize=10)
        if i == 0:
            ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "accuracy.png")
    plt.close(fig)
    for m, k in [("standard4", "None"), ("standard10", "None"), ("looped_fixed4", "8"), ("looped_noinject", "8")] + [
        ("looped", k) for k in KS_SHOWN
    ]:
        b = s["bins"][m][k]
        print(
            "accuracy.png",
            m,
            k,
            {n: round(b[n]["mean"], 3) for n in ("in", "near", "far")},
            "seeds near:",
            {sd: round(v, 3) for sd, v in b["near"]["seeds"].items()},
        )


def fig_loops(s):
    ks = ["1", "2", "3", "4", "5", "6", "8", "10", "12", "16"]
    kx = [int(k) for k in ks]
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=True)
    for ax, (b, title) in zip(axs, [("in", "hops 1-8 (trained)"), ("near", "hops 9-16"), ("far", "hops 17-24")]):
        ax.axvspan(0.8, 6.2, color=LINE, alpha=0.5, lw=0)
        for m, color, label, ls in [
            ("looped", EMBER, "looped, random k in training", "-"),
            ("looped_fixed4", MUTED, "looped, k = 4 in training", "--"),
            ("looped_noinject", BLUE, "looped, no input re-added", "-."),
        ]:
            seeds = [sd for sd in SEEDS if sd in s["bins"][m]["1"][b]["seeds"]]
            rows = np.array([[s["bins"][m][k][b]["seeds"][sd] for k in ks] for sd in seeds])
            ax.plot(kx, rows.mean(0), color=color, ls=ls, marker="o", ms=3, lw=1.6, label=label)
            ax.fill_between(kx, rows.min(0), rows.max(0), color=color, alpha=0.12, lw=0)
        ax.axhline(s["bins"]["standard4"]["None"][b]["mean"], color=INK, lw=1, ls="--", label="standard, 4 layers")
        ax.axhline(s["bins"]["standard10"]["None"][b]["mean"], color=GREY, lw=1.4, label="standard, 10 layers")
        ax.axhline(s["shortcut_bins"][b], color=MUTED, lw=1, ls=":", label="shortcut baseline")
        ax.set_xscale("log", base=2)
        ax.set_xticks(kx)
        ax.set_xticklabels(ks, fontsize=7)
        ax.minorticks_off()
        ax.set_xlabel("loops at test time (k)")
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylim(0, 1.05)
    axs[0].set_ylabel("accuracy")
    axs[1].legend(frameon=False, fontsize=6.8, loc="upper left", ncol=1)
    for ax in axs:
        ax.text(1.05, 1.0, "k used in training", fontsize=6.5, color=MUTED, va="bottom")
    fig.tight_layout()
    fig.savefig(FIG / "loops.png")
    plt.close(fig)
    for m in ("looped", "looped_fixed4", "looped_noinject"):
        print("loops.png", m, {b: [round(s["bins"][m][k][b]["mean"], 3) for k in ks] for b in ("in", "near", "far")})


def fig_inside(s):
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.7), gridspec_kw={"width_ratios": [1.25, 1]})
    ins = s["inside"]
    ax = axs[0]
    for m, color, ls, label in [
        ("standard4", INK, "--", "standard, 4 layers"),
        ("standard10", GREY, "-", "standard, 10 layers"),
        ("looped", EMBER, "-", "looped, random k (seeds thin)"),
        ("looped_fixed4", MUTED, "-.", "looped, k = 4 in training"),
        ("looped_noinject", BLUE, "-", "looped, no input re-added"),
    ]:
        rows = []
        for sd in SEEDS:
            j = ins.get(f"{m}_s{sd}")
            if j:
                fr = j["query_frontier"]
                rows.append([fr[str(i)] for i in range(len(fr))])
        rows = np.array(rows)
        depth = (
            np.array([1] + [1 + 2 * i for i in range(1, rows.shape[1])])
            if m.startswith("looped")
            else np.arange(rows.shape[1])
        )
        if m == "looped":
            for r in rows:
                ax.plot(depth, r, color=color, lw=0.9, alpha=0.7)
        ax.plot(depth, rows.mean(0), color=color, ls=ls, lw=1.9, marker="o", ms=2.5, label=label)
        print("inside.png A", m, "query frontier per step (seeds):", rows.tolist())
    ax.axhline(8, color=MUTED, lw=0.8, ls=":")
    ax.text(33, 8.2, "longest chain in training", fontsize=7, color=MUTED, ha="right")
    ax.set_xlim(-0.3, 34)
    ax.set_ylim(0, 16)
    ax.set_xlabel("blocks run so far (looped: prelude + 2 per loop)")
    ax.set_ylabel("longest chain answered at the query")
    ax.set_title("A. How far the question has been followed", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax = axs[1]
    j = ins["looped_fixed4_s1"]["query_hop_probe"]
    steps = [str(i) for i in range(11)]
    M = np.array([[j[st][str(h)] if j[st][str(h)] is not None else np.nan for h in range(1, 16)] for st in steps])
    im = ax.imshow(
        M, origin="lower", aspect="auto", cmap="Oranges", vmin=0, vmax=1, extent=(0.5, 15.5, -0.5, len(steps) - 0.5)
    )
    ax.set_xlabel("variable j hops down the question's chain")
    ax.set_ylabel("loop (0 = after the prelude)")
    ax.set_title("B. Walking the chain (k = 4 model, seed 1)", loc="left", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.045, label="probe accuracy")
    print("inside.png B: fixed4 s1 query probe, rows loops 0-10, cols j 1-15:", np.round(M, 2).tolist())
    fig.tight_layout()
    fig.savefig(FIG / "inside.png")
    plt.close(fig)


def fig_convergence(s):
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    shades = {"2": "#f3c9a6", "4": "#eaa46f", "8": EMBER, "12": EMBER_D, "16": "#6e3a15"}
    for h, color in shades.items():
        rows = np.array(
            [s["inside"][f"looped_s{sd}"]["convergence"][h] for sd in SEEDS if f"looped_s{sd}" in s["inside"]]
        )
        if not len(rows):
            continue
        x = np.arange(2, 2 + rows.shape[1])
        ax.plot(x, rows.mean(0), color=color, lw=1.6, label=f"{h} hops")
        print("convergence.png", h, [round(v, 4) for v in rows.mean(0)])
    ax.axvline(6, color=MUTED, lw=0.8, ls=":")
    ax.text(6.2, ax.get_ylim()[1] * 0.9, "most loops in training", fontsize=7, color=MUTED)
    ax.set_xlabel("loop i")
    ax.set_ylabel("relative change |s_i − s_(i−1)| / |s_i|")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=7.5, title="query chain", title_fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "convergence.png")
    plt.close(fig)


def fig_control(s):
    """Standard test against the twin-distractor control set, for the models that answer long chains."""
    if not s.get("control"):
        return
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.7), sharey=True)
    picks = [
        ("standard4", "None", INK, "--", "standard, 4 layers"),
        ("looped", "16", EMBER, "-", "looped, random k, k = 16"),
        ("looped_fixed4", "8", MUTED, "-.", "looped, k = 4 in training, k = 8"),
        ("looped_noinject", "8", BLUE, "-", "looped, no re-injection, k = 8"),
        ("looped_h16", "8", EMBER_D, ":", "looped trained to 16 hops, k = 8"),
        ("standard4_h16", "None", GREY, ":", "standard 4 trained to 16 hops"),
    ]
    x = np.arange(1, 25)
    for ax, title, src in [
        (axs[0], "A. Standard test (distractors 1-8 long)", "test"),
        (axs[1], "B. Control: one distractor as long as the question", "control"),
    ]:
        ax.axvspan(0.5, 8.5, color=LINE, alpha=0.5, lw=0)
        ax.axvspan(8.5, 16.5, color=LINE, alpha=0.2, lw=0)
        ax.text(4.5, 1.02, "trained (all)", ha="center", fontsize=7, color=MUTED)
        ax.text(12.5, 1.02, "trained (\u201c16 hops\u201d models)", ha="center", fontsize=7, color=MUTED)
        base = s["shortcut_baseline"] if src == "test" else s["control_shortcut"]
        ax.plot(x, [base[str(h)] for h in x], color=MUTED, ls=":", lw=1.1, label="shortcut baseline")
        for m, k, color, ls, label in picks:
            if src == "test":
                if m not in s["by_hop"]:
                    continue
                y = [s["by_hop"][m][k][str(h)]["mean"] for h in x]
            else:
                if m not in s["control"]:
                    continue
                y = np.mean(list(s["control"][m][k]["by_hop"].values()), axis=0)
            ax.plot(x, y, color=color, ls=ls, lw=1.8, label=label)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("hops in the question's chain")
        ax.set_xlim(0.5, 24.5)
        ax.set_ylim(0, 1.08)
    axs[0].set_ylabel("accuracy (mean of 3 seeds)")
    axs[0].legend(frameon=False, fontsize=7, loc="center right", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(FIG / "control.png")
    plt.close(fig)
    for m, k, *_ in picks:
        if m in s["control"]:
            c = s["control"][m][k]
            print(
                "control.png",
                m,
                k,
                "control near/far",
                round(np.mean(list(c["near"].values())), 3),
                round(np.mean(list(c["far"].values())), 3),
                "solved",
                c["solved"],
            )


if __name__ == "__main__":
    fig_doubling()
    s = S()
    fig_accuracy(s)
    fig_loops(s)
    if s["inside"]:
        fig_inside(s)
        fig_convergence(s)
    fig_control(s)
