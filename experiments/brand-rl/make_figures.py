"""Figures for the note, all drawn from results/summary.json (laptop). Prints every number it plots.

    .venv/bin/python experiments/brand-rl/make_figures.py configs/main.json

fig1_curves    dev curves per run: rewarded-rule strict pass, held-out-rule strict pass, strict fidelity, vacuous
               share, KL per token
fig2_heldout   held-out change from the base model per arm (final checkpoint, seeds pooled), 95% intervals
fig3_families  per-family strict pass rate on held-out pages: base vs each arm's final checkpoint
fig4_diversity cross-brief similarity on unseen features (block tree, screenshots) and TEAM's own feature
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
sys.path.insert(0, str(HERE))
import ep6

FIG = HERE / "report" / "figures"
INK, SOFT, FAINT, ACCENT, CONTROL = "#1f1a17", "#6b625b", "#c9c0b8", "#d4572a", "#3f6fb0"
ARM_COL = {"check": ACCENT, "env": "#8a3b1c", "team": "#e3a33b", "shuf": CONTROL, "base": SOFT}
ARM_NAME = {"check": "CHECK", "env": "ENV", "team": "TEAM", "shuf": "SHUF"}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": SOFT,
        "axes.labelcolor": INK,
        "xtick.color": SOFT,
        "ytick.color": SOFT,
        "figure.dpi": 150,
    }
)


def run_label(name: str) -> str:
    """check_s0 -> CHECK s0 (the prose's names)."""
    if name == "base":
        return "untrained"
    arm, seed = name.split("_")
    return f"{ARM_NAME[arm]} {seed}"


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def fig1(S, runs):
    panels = [
        ("reward_strict", "Rewarded rules\n(strict pass)"),
        ("heldout_strict", "Rules hidden from\nthe reward (strict pass)"),
        ("fidelity_strict", "Brief texts\n(strict fidelity)"),
        ("vacuous", "Rewarded rules with\nnothing to check"),
        ("kl", "KL to base\n(per token)"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(12, 2.6))
    for ax, (k, title) in zip(axes, panels):
        for r in runs:
            c = S["dev_curves"][r["name"]]
            ys = c["kl"] if k == "kl" else [m["mean"] for m in c["metrics"][k]]
            ax.plot(
                c["steps"],
                ys,
                color=ARM_COL[r["arm"]],
                lw=1.4 if r["seed"] == 0 else 0.9,
                ls="-" if r["seed"] == 0 else "--",
                label=f"{ARM_NAME[r['arm']]} s{r['seed']}",
            )
            print("fig1", k, r["name"], [round(y, 3) if y is not None else None for y in ys])
        ax.set_title(title, fontsize=8.5, color=INK)
        ax.set_xlabel("step")
    axes[0].legend(fontsize=6.5, frameon=False, ncol=1)
    save(fig, "fig1_curves")


def fig2(S):
    rows = [
        ("d_reward_strict", "Rewarded rules, strict pass"),
        ("d_heldout_strict", "Rules hidden from the reward, strict pass"),
        ("d_vacuous", "Rewarded rules with nothing to check"),
        ("d_fidelity_strict", "Brief texts, strict"),
        ("d_slop_p", "Markup slop probe P(AI)"),
        ("d_shot_p", "Screenshot slop probe P(AI)"),
        ("d_copy_structure_ex", "Structure similarity to the exemplar"),
        ("d_valid_ok", "Valid, untruncated"),
    ]
    H = S["heldout_pooled"]
    arms = [a for a in ("check", "env", "team", "shuf") if f"{a}:final" in H]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for j, arm in enumerate(arms):
        for i, (k, _) in enumerate(rows):
            d = H[f"{arm}:final"].get(k)
            if not d or d["mean"] is None:
                continue
            y = i + (j - (len(arms) - 1) / 2) * 0.17
            ax.errorbar(
                d["mean"],
                y,
                xerr=[[d["mean"] - d["lo"]], [d["hi"] - d["mean"]]],
                fmt="o",
                ms=3.5,
                color=ARM_COL[arm],
                lw=1,
                label=ARM_NAME[arm] if i == 0 else None,
            )
            print("fig2", arm, k, round(d["mean"], 3), [round(d["lo"], 3), round(d["hi"], 3)])
    ax.axvline(0, color=FAINT, lw=1)
    ax.set_yticks(range(len(rows)), [t for _, t in rows])
    ax.invert_yaxis()
    ax.set_xlabel("change from the base model (held-out briefs, final checkpoint, 95% interval)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save(fig, "fig2_heldout")


def fig3(S, runs):
    """Per run (final checkpoint): seeds learned different things, so they are not pooled here."""
    split = ep6.family_split()
    H = S["heldout"]
    fams = split["reward"] + split["heldout"]
    cols = ["base"] + [f"{r['name']}:final" for r in runs]
    M = np.array(
        [
            [H[c][f"fam_strict_{f}"]["mean"] if H[c][f"fam_strict_{f}"]["mean"] is not None else np.nan for c in cols]
            for f in fams
        ]
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.imshow(1 - M, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    for i in range(len(fams)):
        for j in range(len(cols)):
            txt = f"{M[i, j]:.2f}" if np.isfinite(M[i, j]) else "n/a"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if np.isfinite(M[i, j]) and M[i, j] < 0.45 else INK,
            )
    ax.set_xticks(
        range(len(cols)), [run_label(c.replace(":final", "")) for c in cols], fontsize=7, rotation=45, ha="right"
    )
    ax.set_yticks(range(len(fams)), [f + (" (hidden)" if f in split["heldout"] else "") for f in fams], fontsize=7)
    ax.axhline(len(split["reward"]) - 0.5, color=ACCENT, lw=1.5)
    ax.set_title(
        "Strict pass rate per rule family, held-out pages (darker = more failures; n/a = never applies)", fontsize=8
    )
    print("fig3", dict(zip(fams, M.round(3).tolist())))
    save(fig, "fig3_families")


def fig4(S, runs):
    """Per run (seeds are separate policies, so their pages are not pooled here), final checkpoint."""
    D = S["diversity"]
    names = ["base"] + [f"{r['name']}:final" for r in runs if f"{r['name']}:final" in D]
    feats = [
        ("wl2", "Block tree (unseen by every reward)"),
        ("sig", "Screenshot (unseen)"),
        ("wl5", "Element tree (TEAM's novelty feature)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.0))
    for ax, (f, title) in zip(axes, feats):
        for i, n in enumerate(names):
            d = D[n][f]
            arm = n.split("_")[0] if n != "base" else "base"
            ax.errorbar(
                i,
                d["cross"],
                yerr=[[d["cross"] - d["cross_lo"]], [d["cross_hi"] - d["cross"]]],
                fmt="o",
                color=ARM_COL.get(arm, SOFT),
                ms=4,
            )
            ax.plot(i, d["within"], marker="_", color=ARM_COL.get(arm, SOFT), ms=12)
            print("fig4", f, n, round(d["cross"], 3), round(d["within"], 3))
        ax.set_xticks(
            range(len(names)), [run_label(n.replace(":final", "")) for n in names], fontsize=7, rotation=45, ha="right"
        )
        ax.set_title(title, fontsize=8.5)
    axes[0].set_ylabel("mean cosine similarity\n(dot: across briefs; dash: within a brief)")
    save(fig, "fig4_diversity")


def fig5(S, runs):
    """Hack share (the outside check) per run, and the monitors' flag steps against random directions."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.8))
    ax = axes[0]
    for r in runs:
        c = S["dev_curves"][r["name"]]
        ax.plot(
            c["steps"],
            c["hack_share"],
            color=ARM_COL[r["arm"]],
            lw=1.4 if r["seed"] == 0 else 0.9,
            ls="-" if r["seed"] == 0 else "--",
            label=f"{ARM_NAME[r['arm']]} s{r['seed']}",
        )
        print("fig5 hack share", r["name"], [round(h, 3) for h in c["hack_share"]])
    ax.axhline(0.10, color=FAINT, lw=1, ls=":")
    ax.set_title("Hack share on dev pages (outside check)", fontsize=8.5)
    ax.set_xlabel("step")
    ax.legend(fontsize=6.5, frameon=False)
    ax = axes[1]
    names = [r["name"] for r in runs]
    for i, n in enumerate(names):
        m = S["monitors"][n]
        rf = [f for f in m["rand_flags"] if f is not None]
        ax.scatter(
            rf, [i] * len(rf), s=10, color=CONTROL, alpha=0.5, label="random direction flags" if i == 0 else None
        )
        for k, mk, col in (("dir_flag", "D", ACCENT), ("probe_flag", "s", "#8a3b1c")):
            if m.get(k) is not None:
                dy = 0.18 if k == "dir_flag" else -0.18  # offset so a diamond is not hidden under a square
                ax.scatter([m[k]], [i + dy], marker=mk, s=28, color=col)
        if m.get("outside_flag") is not None:
            ax.scatter(
                [m["outside_flag"]], [i], marker="|", s=120, color=INK, label="outside check" if i == 0 else None
            )
        print(
            "fig5 flags",
            n,
            {k: m.get(k) for k in ("outside_flag", "dir_flag", "probe_flag", "drift_flag")},
            "random flagged",
            len(rf),
            "of",
            len(m["rand_flags"]),
        )
    ax.set_yticks(range(len(names)), [run_label(n) for n in names], fontsize=7)
    ax.set_xlabel("first dev step flagged")
    ax.set_xlim(0, max(r["steps"] for r in runs) + 10)
    ax.set_title("When each monitor first flags a run", fontsize=8.5)
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], marker="o", ls="", color=CONTROL, alpha=0.6, label="random directions (20)"),
        Line2D([], [], marker="D", ls="", color=ACCENT, label="hack direction"),
        Line2D([], [], marker="s", ls="", color="#8a3b1c", label="probe"),
    ]
    ax.legend(handles=handles, fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save(fig, "fig5_monitors")


def fig6(S):
    st = S.get("steering", {})
    a = st.get("chosen_scale")
    tags = [("steer_dev_a0.1", 0.1), ("steer_dev_a0.25", 0.25), ("steer_dev_a0.5", 0.5)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.8))
    for ax, key, title in (
        (axes[0], "hack", "Hack share (48 dev pages)"),
        (axes[1], "bad", "Broken pages (invalid, empty or loop)"),
    ):
        for tag, sc in tags:
            if tag not in st:
                continue
            v = st[tag]
            ax.scatter(
                [sc],
                [v[f"hack@{sc}"][key]],
                color=ACCENT,
                s=36,
                zorder=3,
                label="hack direction" if sc == 0.1 else None,
            )
            rs = [v[f"rand{i}@{sc}"][key] for i in range(5)]
            ax.scatter([sc] * 5, rs, color=CONTROL, s=14, alpha=0.7, label="random, same norm" if sc == 0.1 else None)
            print("fig6", key, sc, "hack", v[f"hack@{sc}"][key], "random", rs)
        none = st["steer_dev_a0.5"]["none@0.0"][key]
        ax.axhline(none, color=FAINT, lw=1, ls=":")
        ax.set_xticks([0.1, 0.25, 0.5])
        ax.set_xlabel("added norm (× median activation norm)" + (f"; chosen {a}" if a else ""))
        ax.set_title(title, fontsize=8.5)
        ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=7, frameon=False)
    save(fig, "fig6_steering")


def main() -> None:
    cfg = json.loads((HERE / sys.argv[1]).read_text())
    S = json.loads((HERE / "results" / "summary.json").read_text())
    runs = [{**cfg["common"], **r} for r in cfg["runs"]]
    fig1(S, runs)
    fig2(S)
    fig3(S, runs)
    fig4(S, runs)
    fig5(S, runs)
    fig6(S)


if __name__ == "__main__":
    main()
