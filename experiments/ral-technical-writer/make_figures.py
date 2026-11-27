"""Figures for the note, drawn from results/ (summary.json, training logs copied to results/runs/). Prints every
number it plots.

    python make_figures.py

fig1_fooling    held-out ensemble fooling rate per source, 95% bootstrap intervals over titles (0.5 = cannot tell)
fig2_training   RAL vs the frozen-judge control over training: the reward (each arm's own judge), the fixed J_ruler's
                P(reference) on dev answers, and the RAL judge's batch accuracy with the gate's threshold
fig3_length     fooling rate against mean words per source (the length explanation); header-removed copies as diamonds
fig4_diversity  cross-title similarity and the most common opening's share, per source, against the references
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
import common as C

FIG = HERE / "report" / "figures"
INK, SOFT, FAINT, ACCENT, CONTROL = "#1f1a17", "#6b625b", "#c9c0b8", "#d4572a", "#3f6fb0"
NAMES = {"ral1_clean": "RAL seed 1, header removed", "ral_clean": "RAL, header removed", "frozen_clean": "frozen judge, header removed", "ref": "real abstracts", "base": "Qwen3-4B", "base_fs": "Qwen3-4B, 3-shot", "llama_fs": "Llama-3.3-70B, 3-shot",
         "sft": "SFT", "ral": "RAL", "ral1": "RAL, seed 1", "frozen": "frozen judge", "humanizer": "lab humanizer",
         "stealth": "StealthGPT"}  # fmt: skip
COL = {"ral": ACCENT, "ral1": "#e89a6a", "frozen": CONTROL}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": SOFT, "axes.labelcolor": INK,
                     "xtick.color": SOFT, "ytick.color": SOFT, "figure.dpi": 150})  # fmt: skip


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def order(S):
    return [
        s for s in ("base", "base_fs", "llama_fs", "humanizer", "stealth", "sft", "frozen", "ral1", "ral") if s in S
    ]


def fig1(summ):
    S = summ["sources"]
    srcs = [s for s in order(S) if "ensemble_fooling" in S[s]]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    y = np.arange(len(srcs))
    for i, s in enumerate(srcs):
        m, (lo, hi) = S[s]["ensemble_fooling"], S[s]["ensemble_ci"]
        ax.barh(i, m, color=COL.get(s, FAINT), height=0.62)
        ax.plot([lo, hi], [i, i], color=INK, lw=1)
        ax.text(hi + 0.01, i, f"{m:.2f}", va="center", fontsize=8, color=INK)
        print(f"fig1 {s}: {m:.4f} [{lo:.4f}, {hi:.4f}] n={S[s]['n_titles']}")
    ax.axvline(0.5, color=SOFT, ls="--", lw=0.8)
    ax.text(0.5, len(srcs) - 0.4, "cannot tell", color=SOFT, fontsize=7, ha="center")
    ax.set_yticks(y, [NAMES[s] + (" *" if s in ("humanizer", "stealth") else "") for s in srcs])
    ax.set_xlim(0, 0.6)
    ax.set_xlabel("held-out judges fooled (share of verdicts naming the real abstract as AI)")
    save(fig, "fig1_fooling")


def load_log(run):
    p = C.RESULTS / "runs" / run / "log.jsonl"
    return C.read_jsonl(p) if p.exists() else []


def dev_curve(run):
    d = C.RESULTS / "runs" / run / "dev"
    if not d.exists():
        return []
    ps = sorted(d.glob("step_*.summary.json"), key=lambda p: int(p.name.split("_")[1].split(".")[0]))
    return [json.loads(p.read_text()) for p in ps]


def smooth(v, k=10):
    v = np.asarray(v, float)
    return np.array([np.nanmean(v[max(0, i - k + 1) : i + 1]) for i in range(len(v))])


FMT = json.loads((C.RESULTS / "format_trick.json").read_text()) if (C.RESULTS / "format_trick.json").exists() else {}


def fig2():
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 2.8))
    for run, arm in (("ral_s0", "ral"), ("ral_s1", "ral1"), ("frozen_s0", "frozen")):
        L = load_log(run)
        if not L:
            continue
        st = [r["step"] for r in L]
        axs[0].plot(st, smooth([r["reward"] for r in L]), color=COL[arm], label=NAMES[arm])
        F = FMT.get(run, [])
        if F:
            axs[1].plot([r["step"] for r in F], [r["share_bold"] for r in F], "o-", ms=3, color=COL[arm],
                        label=f"{NAMES[arm]}: markdown bold")  # fmt: skip
            axs[1].plot([r["step"] for r in F], [r["share_header"] for r in F], ":", color=COL[arm],
                        label=f"{NAMES[arm]}: title line")  # fmt: skip
        print(f"fig2 {run}: reward first10 {np.mean([r['reward'] for r in L[:10]]):.3f} last10 "
              f"{np.mean([r['reward'] for r in L[-10:]]):.3f}; bold {[(r['step'], r['share_bold']) for r in F][::5]}")  # fmt: skip
        if arm in ("ral", "ral1"):
            seed = "seed 0" if arm == "ral" else "seed 1"
            axs[2].plot(st, smooth([r["judge_acc"] for r in L], 5), color=COL[arm], label=f"current, {seed}")
            rep = [r["judge_acc_replay"] if r["judge_acc_replay"] is not None else np.nan for r in L]
            axs[2].plot(st, smooth(rep, 5), color=COL[arm], ls=":", label=f"replayed, {seed}")
            sk = sum(bool(r["judge_skipped"]) for r in L)
            print(f"fig2 {run} judge updates skipped {sk}/{len(L)}")
    axs[2].axhline(0.75, color=SOFT, ls="--", lw=0.8)
    axs[0].set(title="reward (each arm's own judge)", xlabel="step", ylim=(0, 1))
    axs[1].set(
        title="training answers with the format trick", xlabel="step", ylabel="share of valid answers", ylim=(0, 1.02)
    )
    axs[2].set(title="RAL judge accuracy (gate at 0.75)", xlabel="step", ylim=(0, 1.02))
    for a in axs:
        a.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "fig2_training")


def fig3(summ):
    """The length explanation. Left: each source's mean words against its fooling rate. Right: fooling rate by the ratio
    of a generated abstract's length to the real one's, pooled over the seven model sources (results/length_check.json)."""
    S = summ["sources"]
    L = json.loads((C.RESULTS / "length_check.json").read_text())
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.4))
    ax = axs[0]
    for s in order(S) + [x for x in ("ral_clean", "frozen_clean", "ral1_clean") if x in S]:
        if "ensemble_fooling" not in S[s]:
            continue
        x, y = S[s]["words"], S[s]["ensemble_fooling"]
        ax.scatter(x, y, color=COL.get(s.replace("_clean", ""), SOFT), s=28, zorder=3,
                   marker="o" if not s.endswith("_clean") else "D")  # fmt: skip
        if not s.endswith("_clean"):
            ax.annotate(NAMES.get(s, s), (x, y), xytext=(4, 3), textcoords="offset points", fontsize=7, color=INK)
        print(f"fig3 {s}: words {x:.0f} fooling {y:.3f}")
    ax.axvline(S["ref"]["words"], color=SOFT, ls="--", lw=0.8)
    ax.text(S["ref"]["words"], 0.005, " real length", color=SOFT, fontsize=7)
    ax.set(xlabel="mean words per abstract", ylabel="held-out judges fooled", xlim=(0, None), ylim=(0, None),
           title="per source (diamonds: header removed)")  # fmt: skip
    ax = axs[1]
    bins = list(L["bins"].items())
    labels = ["<0.6", "0.6-0.9", "0.9-1.1", "1.1-1.5", "1.5-2.2", ">2.2"]
    ys = [v["fooling"] for _, v in bins]
    ns = [v["n"] for _, v in bins]
    ax.bar(range(len(ys)), ys, color=FAINT)
    for k, (yv, n) in enumerate(zip(ys, ns)):
        ax.text(k, yv + 0.008, f"{yv:.2f}\nn={n}", ha="center", fontsize=7, color=INK)
    print(f"fig3 bins: {list(zip(labels, ys, ns))}")
    ax.set_xticks(range(len(ys)), labels)
    ax.set(xlabel="generated length / real length", ylabel="held-out judges fooled", ylim=(0, 0.5),
           title="all model sources pooled, by length ratio")  # fmt: skip
    fig.tight_layout()
    save(fig, "fig3_length")


def fig4(summ):
    S = summ["sources"]
    srcs = ["ref", *order(S)]
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.0), sharey=True)
    y = np.arange(len(srcs))
    panels = (("cross_title_sim", axs[0], "cross-title similarity (higher = more alike)"),
              ("opening_share", axs[1], "share starting with the most common 3 words"))  # fmt: skip
    for k, ax, lab in panels:
        vals = [S[s].get(k, np.nan) for s in srcs]
        ax.barh(y, vals, color=[COL.get(s, FAINT if s != "ref" else INK) for s in srcs], height=0.6)
        ax.axvline(S["ref"][k], color=SOFT, ls="--", lw=0.8)
        ax.set_xlabel(lab)
        print(f"fig4 {k}: " + ", ".join(f"{s} {v:.3f}" for s, v in zip(srcs, vals)))
    axs[0].set_yticks(y, [NAMES[s] for s in srcs])
    fig.tight_layout()
    save(fig, "fig4_diversity")


if __name__ == "__main__":
    summ = json.loads((C.RESULTS / "summary.json").read_text())
    fig1(summ)
    fig2()
    fig3(summ)
    fig4(summ)
