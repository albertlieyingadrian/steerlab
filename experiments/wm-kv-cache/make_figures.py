"""Draw the note's figures from results/summary.json (run analyse.py first) and print every plotted number.

python make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "report" / "figures"
EMBER, SOFT, GREY, LIGHT, INK = "#C9561C", "#E9A27A", "#8A8580", "#C8C2B8", "#2B2723"
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "savefig.dpi": 160,
        "legend.frameon": False,
    }
)
BUDGETS = [2, 4, 8, 16]
# window and sink are the policies video models use (orange); H2O and pool are alternatives (ink); random is the control
STYLE = {
    "window": {"color": EMBER, "ls": "-", "marker": "o", "label": "window (last B frames)"},
    "sink": {"color": SOFT, "ls": "--", "marker": "s", "label": "sink (frame 0 + last B-1)"},
    "h2o": {"color": INK, "ls": "-", "marker": "^", "label": "H2O (recent + most attended)"},
    "pool": {"color": INK, "ls": ":", "marker": "D", "label": "pool (recent + 1 mean entry per old frame)"},
    "random": {"color": GREY, "ls": "-", "marker": "x", "label": "random entries (control)"},
}
GAPS = [(2, 3), (4, 7), (8, 15), (16, 31), (32, 47)]
CHANCE = 1 / 8


def untrained_floor():
    """Dream-mode revisit self-consistency of the untrained model, full cache (the floor for that metric)."""
    ev = json.loads((RES / "eval_untrained_s1.json").read_text())
    h, n = np.array(ev["sweep"]["dream/full"]["revisit"]).sum(0)
    print(f"[dream] untrained floor {h / n:.3f}")
    return h / n


def get(S, mode, name, m="revisit"):
    return S["sweep"][f"{mode}/{name}"][m]


def err(r):
    return [[r["rate"] - r["lo"]], [r["hi"] - r["rate"]]]


def budget_panel(ax, S, mode, title):
    full = get(S, mode, "full")
    ax.axhspan(full["lo"], full["hi"], color=LIGHT, alpha=0.5, lw=0)
    ax.axhline(full["rate"], color=INK, lw=1.2, ls="--")
    ax.text(16.6, full["rate"] - 0.045, f"full cache {full['rate']:.2f}", va="center", fontsize=8.5, color=INK)
    floor, flab = (CHANCE, "chance 0.125") if mode == "observed" else (untrained_floor(), "untrained model")
    ax.axhline(floor, color=GREY, lw=0.8, ls=":")
    ax.text(
        33,
        floor - 0.02,
        f"{flab} {floor:.2f}" if mode != "observed" else flab,
        va="top",
        ha="right",
        fontsize=8.5,
        color=GREY,
    )
    print(f"[{mode}] full {full}")
    for k, st in STYLE.items():
        ys = [get(S, mode, f"{k}{b}") for b in BUDGETS]
        x = np.array(BUDGETS) * (1 + 0.03 * (list(STYLE).index(k) - 2))
        ax.errorbar(x, [y["rate"] for y in ys], yerr=np.hstack([err(y) for y in ys]), capsize=2, lw=1.6, ms=5, **st)
        print(
            f"[{mode}] {k}: "
            + ", ".join(f"B{b} {y['rate']:.3f} [{y['lo']:.3f},{y['hi']:.3f}]" for b, y in zip(BUDGETS, ys, strict=True))
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(BUDGETS, [str(b) for b in BUDGETS])
    ax.set_xlim(1.7, 34)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("cache budget B (frames kept; the episode has 48)")
    ax.set_ylabel("revisit consistency")
    if title:
        ax.set_title(title, loc="left", fontsize=10.5)


def fig_budget(S):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), gridspec_kw={"width_ratios": [1.35, 1]})
    budget_panel(axes[0], S, "observed", "a. Revisit consistency by cache budget")
    axes[0].legend(loc="lower right", fontsize=7.8)
    ax = axes[1]
    xs = np.linspace(0, 1, 50)
    ax.plot(xs, CHANCE + (1 - CHANCE) * xs, color=LIGHT, lw=3, zorder=0, label="covered = perfect, rest = chance")
    for k, st in STYLE.items():
        cov, cons = [], []
        for b in BUDGETS:
            r = S["sweep"][f"observed/{k}{b}"]
            cov.append(r["revisit_in"]["tiles"] / r["revisit"]["tiles"])
            cons.append(r["revisit"]["rate"])
        ax.plot(cov, cons, ls="none", marker=st["marker"], color=st["color"], ms=6)
        print(
            f"[coverage] {k}: "
            + ", ".join(f"B{b} cov {c:.3f} cons {v:.3f}" for b, c, v in zip(BUDGETS, cov, cons, strict=True))
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("coverage: revisited places with an entry in the cache")
    ax.set_ylabel("revisit consistency")
    ax.set_title("b. Coverage alone predicts it", loc="left", fontsize=10.5)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "budget.png")
    plt.close(fig)


def fig_dream(S):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    budget_panel(axes[0], S, "dream", "a. Self-consistency in a free rollout")
    axes[0].legend(loc="lower right", fontsize=7)
    ax = axes[1]
    t = np.arange(48)
    for name, st in [
        ("full", {"color": INK, "ls": "--", "label": "full"}),
        ("window8", {"color": EMBER, "label": "window 8"}),
        ("h2o8", {"color": INK, "label": "H2O 8"}),
        ("random8", {"color": GREY, "label": "random 8"}),
    ]:
        e = np.array(S["sweep"][f"dream/{name}"]["rev_err_by_t"])
        ax.plot(t[2:], e[2:], lw=1.6, **st)
        print(f"[dream err] {name}: t=10 {e[10]:.3f} t=24 {e[24]:.3f} t=47 {e[47]:.3f}")
    ax.set_ylim(0, 1)
    ax.set_xlabel("frame of the rollout")
    ax.set_ylabel("revisit error (1 - consistency)")
    ax.set_title("b. Revisit error over the rollout, B = 8", loc="left", fontsize=10.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "dream.png")
    plt.close(fig)


def fig_gap(S):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    xs = np.arange(len(GAPS))
    for name, st in [
        ("full", {"color": INK, "ls": "--", "marker": "o", "label": "full"}),
        ("window16", {"color": EMBER, "marker": "o", "label": "window 16"}),
        ("window8", {"color": EMBER, "marker": "o", "label": "window 8", "alpha": 0.65}),
        ("window4", {"color": EMBER, "marker": "o", "label": "window 4", "alpha": 0.35}),
    ]:
        ys = [get(S, "observed", name, f"gap{lo}_{hi}") for lo, hi in GAPS]
        ax.errorbar(xs, [y["rate"] for y in ys], yerr=np.hstack([err(y) for y in ys]), capsize=2, lw=1.6, **st)
        print(
            f"[gap] {name}: "
            + ", ".join(f"{lo}-{hi} {y['rate']:.3f} (n={y['tiles']})" for (lo, hi), y in zip(GAPS, ys, strict=True))
        )
    ax.axhline(CHANCE, color=GREY, lw=0.8, ls=":")
    ax.set_xticks(xs, [f"{lo}-{hi}" for lo, hi in GAPS])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("gap: steps since the place was last in view")
    ax.set_ylabel("revisit consistency")
    ax.set_title("a. A window forgets places older than itself", loc="left", fontsize=10.5)
    ax.legend(fontsize=8, loc="center left")
    ax = axes[1]
    names = ["window", "sink", "h2o", "pool", "random"]
    w = 0.38
    for i, k in enumerate(names):
        a = get(S, "observed", f"{k}8", "revisit_in")
        b = get(S, "observed", f"{k}8", "revisit_out")
        col = STYLE[k]["color"]
        ax.bar(i - w / 2, a["rate"], w, color=col, yerr=err(a), capsize=2, ecolor=INK)
        ax.bar(i + w / 2, b["rate"], w, color="white", edgecolor=col, hatch="///", yerr=err(b), capsize=2, ecolor=INK)
        print(f"[in/out B8] {k}: in {a['rate']:.3f} (n={a['tiles']}) out {b['rate']:.3f} (n={b['tiles']})")
    ax.axhline(CHANCE, color=GREY, lw=0.8, ls=":")
    ax.set_xticks(range(len(names)), ["window", "sink", "H2O", "pool", "random"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("revisit consistency")
    ax.set_title("b. B = 8: place in the cache (solid) or not", loc="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / "gap.png")
    plt.close(fig)


def fig_attention(S):
    A = S["attention"]
    layers = list(A)
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    x = np.arange(len(layers))
    s = [A[L]["sight"]["mean"] for L in layers]
    u = [A[L]["uniform_sight"]["mean"] for L in layers]
    t1 = [A[L]["top1_sight"]["mean"] for L in layers]
    lo = [A[L]["sight"]["min"] for L in layers]
    hi = [A[L]["sight"]["max"] for L in layers]
    ax.bar(x - 0.2, s, 0.4, color=EMBER, label="attention on earlier views of the place")
    ax.errorbar(x - 0.2, s, yerr=[np.subtract(s, lo), np.subtract(hi, s)], fmt="none", ecolor=INK, capsize=2)
    ax.bar(x + 0.2, u, 0.4, color=GREY, label="share if attention were spread evenly")
    ax.plot(x, t1, "D", color=INK, ms=5, label="top entry is such a view (fraction)")
    for i, L in enumerate(layers):
        print(
            f"[attn] {L}: sight {s[i]:.3f} [{lo[i]:.3f},{hi[i]:.3f}] uniform {u[i]:.4f} top1 {t1[i]:.3f} "
            f"prev_frame {A[L]['prev_frame']['mean']:.3f} last_sight {A[L]['last_sight']['mean']:.3f}"
        )
    ax.set_xticks(x, [f"layer {i}" for i in range(len(layers))])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "attention.png")
    plt.close(fig)


def fig_causal(S):
    C = S["causal"]
    rows = [
        ("full", "nothing hidden", INK),
        ("q_random", "random entries", GREY),
        ("q_attended", "most-attended entries", SOFT),
        ("q_sight", "earlier views of the place", EMBER),
        ("q_random_frames", "random whole frames", GREY),
        ("q_sight_frames", "whole frames with those views", EMBER),
        ("c_random", "random entries, from creation", GREY),
        ("c_sight", "views, from creation", EMBER),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    y = np.arange(len(rows))[::-1]
    for yi, (k, lab, col) in zip(y, rows, strict=True):
        r = C[k]
        ax.barh(yi, r["rate"], color=col, height=0.62, xerr=err(r), capsize=2, ecolor=INK)
        ax.text(r["hi"] + 0.015, yi, f"{r['rate']:.2f}", va="center", fontsize=8.5)
        print(f"[causal] {k} ({lab}): {r['rate']:.3f} [{r['lo']:.3f},{r['hi']:.3f}]")
    ax.axvline(CHANCE, color=GREY, lw=0.8, ls=":")
    ax.set_yticks(y, [r[1] for r in rows])
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("revisit consistency at the target frame")
    for k, v in C.items():
        if not (isinstance(v, dict) and "rate" in v and k in {r[0]: 1 for r in rows}):
            print(f"[causal] {k}: {v}")
    fig.tight_layout()
    fig.savefig(FIG / "causal.png")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    S = json.loads((RES / "summary.json").read_text())
    fig_budget(S)
    fig_gap(S)
    fig_dream(S)
    fig_attention(S)
    fig_causal(S)
    print("[timing ms/frame]", {k: v["mean"] for k, v in S["timing_ms_per_frame"].items()})
    print("[probe]", {k: {a: b["mean"] for a, b in v.items()} for k, v in S["probe"].items()})
    print("[paired]", json.dumps(S["paired"]))


if __name__ == "__main__":
    main()
