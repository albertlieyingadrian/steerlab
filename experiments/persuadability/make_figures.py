"""Draw the note's figures from results/analysis.json and print every number they plot.

    python make_figures.py

Writes to report/figures/:
- map.png       flip rate (wrong pushback) against update rate (correct pushback + new information), pooled over the
                three test domains, judge readout: untrained, the four LoRA arms (mean of two seeds), steering at the
                pre-registered strength and at half of it, and the random directions at both strengths
- transfer.png  flip rate in domain (general knowledge) and on unseen domains (arithmetic + memo), judge readout
- dose.png      steering at layer 12 by signed strength (-4, -2, 0, +2, +4 x ||d||): flip, update and new-information
                rates, against the range of five random directions at |alpha| = 2 and 4
- costs.png     side effects: MMLU letter accuracy, benign follow-ups done, coherent replies
Colours: steering orange, LoRA ink, random directions blue (controls), the untrained model grey.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C

FIG = C.HERE / "report" / "figures"
INK, SOFT, FAINT, RULE = "#1d272a", "#586468", "#8a9497", "#e3e7e8"
ACCENT, ACCENT2, CONTROL, CONTROL_L = "#c9561c", "#e0a07c", "#4a86e8", "#c9daf8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelcolor": SOFT, "text.color": INK})
LORA = ["lora_easy", "lora_hard", "lora_resist", "lora_twin"]
NAMES = {
    "base": "untrained",
    "lora_easy": "LoRA easy",
    "lora_hard": "LoRA hard",
    "lora_resist": "LoRA resist",
    "lora_twin": "LoRA twin",
    "steer_easy": "steer easy (α=4)",
    "steer_hard": "steer hard (α=4)",
    "dose_0.5x_easy": "steer easy (α=2)",
    "dose_0.5x_hard": "steer hard (α=2)",
}


def style(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.tick_params(colors=SOFT, length=0)
    if grid:
        ax.grid(axis=grid, color=RULE, lw=0.6, zorder=0)


def colour(org):
    if org == "base":
        return FAINT
    if org.startswith("lora"):
        return INK
    if org.startswith("dose"):
        return ACCENT2
    return ACCENT


def fig_map(t):
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=200)
    style(ax, grid="both")
    for k in range(5):
        for name, c in ((f"rand{k}", CONTROL), (f"randhalf{k}", CONTROL_L)):
            r = t[name]
            ax.scatter(r["update_pooled"][0], r["flip_pooled"][0], s=22, color=c, zorder=2, edgecolor="none")
    print("map (update_pooled, flip_pooled), judge:")
    for org in ["base", *LORA, "steer_easy", "steer_hard", "dose_0.5x_easy", "dose_0.5x_hard"]:
        r = t[org]
        x, y = r["update_pooled"], r["flip_pooled"]
        ax.errorbar(
            x[0],
            y[0],
            xerr=[[x[0] - x[1]], [x[2] - x[0]]],
            yerr=[[y[0] - y[1]], [y[2] - y[0]]],
            fmt="o",
            ms=6,
            color=colour(org),
            ecolor=colour(org),
            elinewidth=0.8,
            capsize=0,
            zorder=3,
        )
        dx, dy = 0.012, 0.018
        if org == "lora_twin":
            dy = 0.02
        elif org == "lora_hard":
            dy = -0.045
        ax.text(x[0] + dx, y[0] + dy, NAMES[org], fontsize=7.5, color=colour(org) if org != "base" else SOFT)
        print(f"  {org:16s} update {x}  flip {y}")
    ax.scatter([], [], s=22, color=CONTROL, label="random direction, α=4 (5)")
    ax.scatter([], [], s=22, color=CONTROL_L, label="random direction, α=2 (5)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.annotate(
        "wanted: hard to persuade,\nstill takes fair corrections",
        xy=(0.99, 0.02),
        xytext=(0.72, 0.30),
        fontsize=7.5,
        color=SOFT,
        arrowprops={"arrowstyle": "->", "color": FAINT, "lw": 0.7},
    )
    ax.set_xlim(0.5, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("update rate: takes a correct pushback or new information (pooled, 3 domains)")
    ax.set_ylabel("flip rate: gives in to a wrong pushback (pooled)")
    fig.tight_layout()
    fig.savefig(FIG / "map.png", facecolor="white")
    plt.close(fig)


def fig_transfer(t):
    orgs = ["base", *LORA, "steer_easy", "steer_hard", "dose_0.5x_easy", "dose_0.5x_hard"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=200)
    style(ax)
    w = 0.38
    print("transfer (flip_know, flip_unseen), judge:")
    for i, org in enumerate(orgs):
        a, b = t[org]["flip_know"], t[org]["flip_unseen"]
        c = colour(org)
        ax.bar(i - w / 2, a[0], w, color=c, alpha=0.45, zorder=2)
        ax.bar(i + w / 2, b[0], w, color=c, zorder=2)
        ax.errorbar(
            [i - w / 2, i + w / 2],
            [a[0], b[0]],
            yerr=[[a[0] - a[1], b[0] - b[1]], [a[2] - a[0], b[2] - b[0]]],
            fmt="none",
            ecolor=SOFT,
            elinewidth=0.7,
            zorder=3,
        )
        print(f"  {org:16s} in domain {a}  unseen {b}")
    ax.set_xticks(range(len(orgs)))
    ax.set_xticklabels([NAMES[o].replace(" (", "\n(") for o in orgs], fontsize=7, color=SOFT)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("flip rate (wrong pushback)")
    ax.bar([], [], color=FAINT, alpha=0.45, label="in domain: general knowledge (120)")
    ax.bar([], [], color=FAINT, label="unseen: arithmetic + memo (200)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "transfer.png", facecolor="white")
    plt.close(fig)


def fig_dose(t):
    xs = [-4, -2, 0, 2, 4]
    org_at = {-4: "steer_hard", -2: "dose_0.5x_hard", 0: "base", 2: "dose_0.5x_easy", 4: "steer_easy"}
    rand_at = {2: [f"randhalf{k}" for k in range(5)], 4: [f"rand{k}" for k in range(5)]}
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.9), dpi=200, sharey=True)
    print("dose (judge):")
    panels = (
        ("flip_pooled", "flip rate (wrong pushback)"),
        ("update_pooled", "update rate (correct + new info)"),
        ("newinfo_memo", "new-information rate (memo)"),
    )
    for ax, (m, label) in zip(axes, panels, strict=False):
        style(ax)
        for s in (2, 4):
            vals = [t[r][m][0] for r in rand_at[s]]
            for x in (-s, s):
                ax.plot([x, x], [min(vals), max(vals)], color=CONTROL_L, lw=6, solid_capstyle="butt", zorder=1)
        ys = [t[org_at[x]][m] for x in xs]
        ax.plot(xs, [y[0] for y in ys], color=ACCENT, lw=1.4, zorder=3)
        ax.errorbar(
            xs,
            [y[0] for y in ys],
            yerr=[[y[0] - y[1] for y in ys], [y[2] - y[0] for y in ys]],
            fmt="o",
            ms=4,
            color=ACCENT,
            ecolor=ACCENT,
            elinewidth=0.8,
            zorder=4,
        )
        ax.set_title(label, fontsize=8.5, color=INK)
        ax.set_xticks(xs)
        ax.set_xticklabels(["−4\nhold", "−2", "0", "+2", "+4\nyield"], fontsize=7.5)
        ax.set_ylim(0, 1.05)
        band = {s: (min(t[r][m][0] for r in rand_at[s]), max(t[r][m][0] for r in rand_at[s])) for s in (2, 4)}
        print(
            f"  {m}: "
            + ", ".join(f"{x:+d} {t[org_at[x]][m][0]:.3f}" for x in xs)
            + f" | random a=2 [{band[2][0]:.3f}, {band[2][1]:.3f}] a=4 [{band[4][0]:.3f}, {band[4][1]:.3f}]"
        )
    axes[1].set_xlabel("strength along the yield−hold direction at layer 12 (× its length)")
    fig.tight_layout()
    fig.savefig(FIG / "dose.png", facecolor="white")
    plt.close(fig)


def val(x):
    return x[0] if isinstance(x, list) else x


def fig_costs(t):
    orgs = ["base", *LORA, "steer_easy", "steer_hard", "dose_0.5x_easy", "dose_0.5x_hard"]
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.2), dpi=200, sharey=True)
    print("side effects:")
    panels = (
        ("mmlu", "MMLU letter accuracy (300)"),
        ("benign_complied", "benign requests done (60)"),
        ("coherent", "coherent replies (judge)"),
    )
    for ax, (m, label) in zip(axes, panels, strict=False):
        style(ax, grid="x")
        for k, rand in ((4, [f"rand{j}" for j in range(5)]), (2, [f"randhalf{j}" for j in range(5)])):
            vals = [val(t[r][m]) for r in rand]
            ax.axvspan(min(vals), max(vals), color=CONTROL_L if k == 2 else CONTROL, alpha=0.18, lw=0)
        for i, org in enumerate(orgs):
            v = val(t[org][m])
            ax.barh(i, v, color=colour(org), height=0.6, zorder=2)
            ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=7, color=SOFT)
        ax.set_title(label, fontsize=8.5, color=INK)
        ax.set_xlim(0, 1.12)
        print(f"  {m}: " + ", ".join(f"{o} {val(t[o][m]):.3f}" for o in orgs))
    axes[0].set_yticks(range(len(orgs)))
    axes[0].set_yticklabels([NAMES[o] for o in orgs], fontsize=7.5, color=SOFT)
    axes[0].invert_yaxis()
    fig.text(
        0.99, 0.01, "shaded: range of 5 random directions (dark α=4, light α=2)", ha="right", fontsize=7, color=SOFT
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "costs.png", facecolor="white")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    t = json.loads((C.RESULTS / "analysis.json").read_text())["judge"]["table"]
    fig_map(t)
    fig_transfer(t)
    fig_dose(t)
    fig_costs(t)


if __name__ == "__main__":
    main()
