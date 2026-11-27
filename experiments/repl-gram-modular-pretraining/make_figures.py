"""Figures 1-4 of the note from results/summary.json and results/runs/. Prints every number it plots.

.venv/bin/python experiments/repl-gram-modular-pretraining/analyze.py
.venv/bin/python experiments/repl-gram-modular-pretraining/make_figures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
RUNS = RES / "runs"
FIG = HERE / "report" / "figures"
EMBER, GREY, BLUE, INK = "#d9622b", "#8a8a8a", "#3b6fb6", "#222222"
AUX = [
    "a-deadline-or-time-limit",
    "alien-encounters",
    "bygone-eras",
    "cultural-traditions",
]
SHORT = {
    "a-deadline-or-time-limit": "deadline",
    "alien-encounters": "aliens",
    "bygone-eras": "bygone eras",
    "cultural-traditions": "traditions",
}
COLOR = {"GRAM": EMBER, "Filtering": GREY, "MaxEnt": BLUE}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)


def ci95(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return float(v.mean()), 0.0
    return float(v.mean()), float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v)))


def seeds():
    return sorted(d for d in RUNS.glob("seed_*") if (d / "baseline.json").exists())


def load(d, name):
    p = d / name
    return json.loads(p.read_text()) if p.exists() else None


def fig1(S):
    """Theirs vs ours: compute ratio per metric and method (bars: ours, 90% CI over seeds; diamonds: the paper)."""
    metrics = [
        ("core", "Core"),
        ("retain", "Retain\n(module on)"),
        ("forget", "Forget\n(module off)"),
        ("elicited_forget", "Forget after\nrelearning"),
    ]
    meths = ["GRAM", "Filtering", "MaxEnt"]
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    w = 0.26
    print("Figure 1: compute ratio, ours (mean, 90% CI) vs paper")
    for j, m in enumerate(meths):
        for i, (k, _) in enumerate(metrics):
            v = S["summary"].get(m, {}).get(k)
            if not v:
                continue
            x = i + (j - 1) * w
            ax.bar(
                x,
                v["mean"],
                w * 0.92,
                color=COLOR[m],
                label=m if i == 0 else None,
                zorder=2,
            )
            ax.errorbar(x, v["mean"], yerr=v["ci90"], color=INK, lw=1, capsize=2, zorder=3)
            pp = S["paper"].get(m, {}).get(k)
            if pp:
                ax.plot(
                    x,
                    pp[0],
                    marker="D",
                    ms=5,
                    mfc="white",
                    mec=INK,
                    mew=1.1,
                    zorder=4,
                    label="paper (26M, 3 seeds)" if (i == 0 and j == 0) else None,
                )
            print(
                f"  {m:9s} {k:16s} ours {v['mean']:.3f} ± {v['ci90']:.3f}   paper {pp[0] if pp else float('nan'):.3f}"
            )
    ax.axhline(1.0, color=INK, lw=0.6, ls=":", zorder=1)
    ax.set_xticks(range(len(metrics)), [m[1] for m in metrics], fontsize=8)
    ax.set_ylabel("compute ratio (baseline = 1)", fontsize=8)
    ax.set_ylim(0, 1.3)
    ax.legend(
        fontsize=7.5,
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
    )
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_theirs_vs_ours.png", dpi=200)
    plt.close(fig)


def fig2():
    """Module off minus module on (GRAM) next to never-seen minus seen (filtering), per topic, in nats."""
    rows = {}
    for d in seeds():
        ge = load(d, "gram_eval.json")
        f0 = load(d, "filter_core.json")
        for a in AUX:
            fi = load(d, f"filter_core+{a}.json")
            if ge:
                rows.setdefault(("GRAM", a), []).append(
                    ge["profiles"]["core"]["test"][a] - ge["profiles"][f"core+{a}"]["test"][a]
                )
            if f0 and fi:
                rows.setdefault(("Filtering", a), []).append(f0["test"][a] - fi["test"][a])
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    print("Figure 2: loss gap, topic absent minus present (nats), mean ± 95% CI over seeds")
    for j, m in enumerate(["GRAM", "Filtering"]):
        for i, a in enumerate(AUX):
            v = rows.get((m, a))
            if not v:
                continue
            mu, h = ci95(v)
            y = i + (0.15 if j == 0 else -0.15)
            ax.errorbar(
                mu,
                y,
                xerr=h,
                fmt="o",
                color=COLOR[m],
                ms=5,
                capsize=2,
                label=m if i == 0 else None,
            )
            print(f"  {m:9s} {a:26s} {mu:.4f} ± {h:.4f}  per seed {[round(x, 4) for x in v]}")
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_yticks(range(4), [SHORT[a] for a in AUX], fontsize=8)
    ax.set_xlabel("loss without the topic minus loss with it (nats per token)", fontsize=8)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_module_gap.png", dpi=200)
    plt.close(fig)


def fig3():
    """Relearning attack on the core-only profile: val loss per attack step, minus the baseline's loss on that topic."""
    curves = {}
    for d in seeds():
        b = load(d, "baseline.json")
        srcs = {
            "GRAM": [e for e in (load(d, "gram_elicit.json") or []) if e["profile"] == "core"],
            "Filtering": load(d, "filter_core_elicit.json") or [],
            "MaxEnt": (load(d, "maxent_core.json") or {}).get("elicit", []),
        }
        for m, es in srcs.items():
            for e in es:
                c = np.array(e["val_curve"]) - b["val"][e["label"]]
                curves.setdefault((m, e["num_seq"]), []).append(c)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.0), sharey=True)
    print("Figure 3: relearning, val loss minus baseline (nats), mean over topics and seeds")
    for ax, n in zip(axes, (512, 4096), strict=True):
        for m in ("GRAM", "Filtering", "MaxEnt"):
            cs = curves.get((m, n))
            if not cs:
                continue
            L = max(len(c) for c in cs)
            pad = np.array([np.pad(c, (0, L - len(c)), mode="edge") for c in cs])  # hold the last value
            mu = pad.mean(0)
            ax.plot(np.arange(L), mu, color=COLOR[m], lw=1.6, label=m)
            best = np.array([c.min() for c in cs])
            print(f"  n={n:4d} {m:9s} start {mu[0]:.4f}  best-step mean {best.mean():.4f}  (k={len(cs)})")
        ax.axhline(0, color=INK, lw=0.6, ls=":")
        ax.set_yscale("symlog", linthresh=0.05, linscale=2.0)
        ax.set_yticks([-0.02, 0, 0.02, 0.04, 0.06, 0.1, 0.5, 1.0])
        ax.set_yticklabels(["-0.02", "0", "0.02", "0.04", "0.06", "0.1", "0.5", "1"])
        ax.set_xlim(0, 40)
        ax.set_title(
            f"{n} attack sequences" + (" (the paper's budget)" if n == 512 else " (8x)"),
            fontsize=8.5,
        )
        ax.set_xlabel("attack step (one full pass over the sequences)", fontsize=8)
        style(ax)
    axes[0].set_ylabel("loss above baseline (nats; linear < 0.05, log above)", fontsize=8)
    axes[0].legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_relearning.png", dpi=200)
    plt.close(fig)


def fig4(S):
    """The yardstick: the baseline's loss curves with the fitted power laws."""
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    print("Figure 4: power-law fits (A, alpha, x0, rmse nats)")
    for lab, col in (("core", INK), ("alien-encounters", EMBER)):
        f = S["curve_fit"][lab]
        xs = []
        for d in seeds():
            b = load(d, "baseline.json")
            xs = [p["step"] for p in b["curve"]]
            ax.plot(xs, [p[lab] for p in b["curve"]], ".", color=col, ms=2, alpha=0.5)
        xx = np.linspace(0.05 * max(xs), max(xs), 200)
        ax.plot(
            xx,
            f["A"] * (xx + f["x0"]) ** -f["alpha"],
            color=col,
            lw=1.2,
            label=f"{lab if lab == 'core' else SHORT[lab]} (fit)",
        )
        print(f"  {lab:20s} A {f['A']:.3f} alpha {f['alpha']:.4f} x0 {f['x0']:.1f} rmse {f['rmse_nats']:.4f}")
    ax.axvline(0.9 * max(xs), color=GREY, lw=0.8, ls=":")
    ax.text(
        0.9 * max(xs),
        2.13,
        " learning-rate\n decay starts",
        fontsize=7,
        color=GREY,
        va="top",
    )
    ax.set_xlim(1500, max(xs) * 1.02)
    ax.set_ylim(1.55, 2.15)
    ax.set_xlabel("baseline training step", fontsize=8)
    ax.set_ylabel("test loss (nats per token)", fontsize=8)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_yardstick.png", dpi=200)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    S = json.loads((RES / "summary.json").read_text())
    fig1(S)
    fig2()
    fig3()
    fig4(S)


if __name__ == "__main__":
    main()
