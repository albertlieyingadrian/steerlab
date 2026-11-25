"""Draw the note's figures from results/ and print every number plotted.

.venv/bin/python experiments/looped-transformer-from-scratch/make_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "report" / "figures"
ACCENT, CONTROL, INK, FAINT = "#d9622b", "#2f6fb0", "#222222", "#9a9a9a"
SEEDS = (0, 1, 2)


def load(seed, name):
    return json.loads((RES / f"seed{seed}" / name).read_text())


def reach(row):
    longest = 0
    for h in sorted(row, key=int):
        if row[h] < 0.9:
            break
        longest = int(h)
    return longest


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def fig_grid():
    """Figure 1: accuracy by loop count and hop count, seed 0, plus the two standard models."""
    grid = load(0, "looped_s0.json")["grid"]
    cmp_ = load(0, "compare_s0.json")["grid"]
    rows = [(f"looped K={k}", grid[k]) for k in grid] + [
        ("standard, 3 blocks", cmp_["standard3"]["None"]),
        ("standard, 10 blocks", cmp_["standard10"]["None"]),
    ]
    hops = sorted(rows[0][1], key=int)
    m = np.array([[r[h] for h in hops] for _, r in rows])
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.imshow(m, cmap="Oranges", vmin=0, vmax=1, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(
                j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if m[i, j] > 0.7 else INK
            )
    ax.set_xticks(range(len(hops)), hops)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.axvline(5.5, color=INK, lw=1)
    ax.axhline(len(grid) - 0.5, color=INK, lw=1)
    ax.set_xlabel("hops in the question's chain (training saw 1-6; right of the line: longer)")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig1-grid.png", dpi=150)
    print("fig1 rows:", [(name, [round(r[h], 2) for h in hops]) for name, r in rows])


def fig_reach():
    """Figure 2: longest chain solved against loop count, every seed, with the standard models."""
    fig, ax = plt.subplots(figsize=(6, 3.4))
    out = {}
    for s in SEEDS:
        grid = load(s, f"looped_s{s}.json")["grid"]
        ks = sorted(grid, key=int)
        r = [reach(grid[k]) for k in ks]
        out[f"looped s{s}"] = dict(zip(ks, r, strict=True))
        ax.plot(
            [int(k) for k in ks],
            r,
            "-o",
            color=ACCENT,
            ms=3,
            alpha=0.5 + 0.25 * (s == 0),
            label="looped (3 seeds)" if s == 0 else None,
        )
        c = load(s, f"compare_s{s}.json")["grid"]
        for kind, ls in (("standard3", "--"), ("standard10", ":")):
            v = reach(c[kind]["None"])
            out.setdefault(kind, []).append(v)
            ax.axhline(
                v + 0.04 * s,
                color=CONTROL if kind == "standard3" else FAINT,
                ls=ls,
                lw=1,
                label=(
                    {
                        "standard3": "standard, 3 blocks (same weights)",
                        "standard10": "standard, 10 blocks (compute of K=8)",
                    }[kind]
                    if s == 0
                    else None
                ),
            )
    ax.axhline(6, color=INK, lw=0.6)
    ax.text(16, 6.15, "longest chain in training", ha="right", fontsize=7)
    ax.set_xlabel("loops at test time (K)")
    ax.set_ylabel("longest chain solved (hops)")
    ax.set_ylim(0, 12)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig2-reach.png", dpi=150)
    print("fig2:", out)


def fig_lenses():
    """Figure 3: every lens, loop by loop (chains of 1-6 hops). Left: the looped model that re-adds its input (seed 0).
    Right: the Inkling-style model whose passes restart from the input (seed 1; its seed 0 never used its loops)."""
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    spec = [
        ("tuned_per_loop", "tuned lens, one per loop", ACCENT, "-"),
        ("tuned_shared", "tuned lens, one for all loops", ACCENT, ":"),
        ("jacobian_per_loop", "Jacobian lens, one per loop", CONTROL, "-"),
        ("jacobian_averaged", "Jacobian lens, averaged over loops", CONTROL, ":"),
        ("logit_lens", "logit lens", FAINT, "--"),
    ]
    for ax, (seed, name, title) in zip(
        axes,
        (
            (0, "tuned_lens.json", "re-add the input (looped), seed 0"),
            (1, "looped_message_tuned_lens.json", "restart + gated message, seed 1"),
        ),
        strict=True,
    ):
        rows = load(seed, name)
        loops = [r["loop"] for r in rows]
        for key, label, col, ls in spec:
            ax.plot(loops, [r[key] for r in rows], ls, marker="o", ms=3, color=col, label=label)
            print("fig3", name, key, [round(r[key], 2) for r in rows])
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("after loop (0 = after the prelude)")
        ax.set_ylim(0, 1.05)
        style(ax)
    axes[0].set_ylabel("lens names the right answer")
    axes[1].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig3-lenses.png", dpi=150)


def fig_probe():
    """Figure 4: probes after each loop on 6-hop chains, seed 0. Rows 1-6: the chain's digit, read at the statement d
    links from the end of the chain; last row: the answer, read at the question."""
    rows = load(0, "probe.json")
    keys = [f"digit_at_dist{d}" for d in range(1, 7)] + ["answer"]
    labels = [f"statement {d} from the end" for d in range(1, 7)] + ["question (answer)"]
    m = np.array([[r[k] for k in keys] for r in rows]).T
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.imshow(m, cmap="Oranges", vmin=0, vmax=1, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(
                j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if m[i, j] > 0.7 else INK
            )
    ax.set_yticks(range(len(keys)), labels)
    ax.set_xticks(range(len(rows)), [r["loop"] for r in rows])
    ax.set_xlabel("after loop")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig4-probe.png", dpi=150)
    extra = ["hop2", "hop3", "answer_shuffled_labels", "answer_untrained_model"]
    print("fig4:", {k: [round(r[k], 2) for r in rows] for k in keys + extra})


def fig_patch():
    """Figure 5: activation patching by loop, seed 0."""
    res = load(0, "patch.json")
    rows = res["rows"]
    loops = [r["loop"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(loops, [r["all_positions"] for r in rows], "-o", ms=3, color=ACCENT, label="patch every position")
    ax.plot(loops, [r["question_only"] for r in rows], "--o", ms=3, color=ACCENT, label="patch the question only")
    ax.plot(loops, [r["random_control"] for r in rows], ":o", ms=3, color=CONTROL, label="random vector (control)")
    ax.set_xlabel("patched right after loop (0 = the prelude's output)")
    ax.set_ylabel("corrupted run gives the clean answer")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig5-patch.png", dpi=150)
    print(
        "fig5:",
        res["base"],
        [
            (r["loop"], round(r["all_positions"], 2), round(r["question_only"], 2), round(r["random_control"], 2))
            for r in rows
        ],
    )


def fig_sae():
    """Figure 6: the answer-digit latents of one shared SAE, loop by loop, by hop count, seed 0."""
    rep = load(0, "sae.json")
    track = rep["track"]
    loops = [r["loop"] for r in track]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    cmap = plt.get_cmap("Oranges")
    rnd = rep["random_direction_track"]
    for h in range(1, 7):
        ax.plot(loops, [r[f"hops{h}"] for r in track], "-o", ms=3, color=cmap(0.3 + 0.12 * h), label=f"{h} hops")
        ax.plot(
            loops,
            [r[f"hops{h}"] for r in rnd],
            ":",
            color=CONTROL,
            alpha=0.3 + 0.1 * h,
            label="random direction (control)" if h == 6 else None,
        )
    ax.set_xlabel("after loop")
    ax.set_ylabel("the true digit's latent fires")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig6-sae.png", dpi=150)
    print(
        "fig6: fvu",
        round(rep["fvu"], 3),
        "alive",
        rep["alive_latents"],
        "best",
        rep["best_latents"],
        "random",
        [round(x, 2) for x in rep["random_direction_selectivity"]],
    )
    print("fig6 track:", [[round(r[f"hops{h}"], 2) for h in range(1, 7)] for r in track])
    print("fig6 random track:", [[round(r[f"hops{h}"], 2) for h in range(1, 7)] for r in rnd])


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    for f in (fig_grid, fig_reach, fig_lenses, fig_probe, fig_patch, fig_sae):
        f()
