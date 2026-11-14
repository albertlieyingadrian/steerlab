"""Figures for the Episode 2 note, drawn from results/ only (no API, no GPU). Prints the numbers it plots.

python make_figures.py      # report/figures/*.png
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "report" / "figures"
EMBER, GREY, INK = "#d9622b", "#8a8f98", "#222222"
PIGMENT = HERE.parent / "pigment-gym" / "results" / "qwen3"
plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 200}
)


def rates(path):
    with Path(path).open() as f:
        return [float(r["hack_rate"]) for r in csv.DictReader(f)]


def smooth(x, w=5):
    x = np.asarray(x)
    return np.array([x[max(0, i - w + 1) : i + 1].mean() for i in range(len(x))])


def runs(arm):
    out = {}
    for k in range(6):
        p = RES / "rl" / f"{arm}_s{k}_curve.csv"
        if p.exists():
            out[f"{arm}_s{k}"] = rates(p)
    if arm == "B":  # the untrained model: Pigment Gym's v2 runs are seeds 0-3 (this fork's B_s0 is a check)
        for k in range(4):
            p = PIGMENT / f"loophole2_seed{k}_curve.csv"
            if p.exists():
                out[f"B_s{k}"] = rates(p)
    return out


def fig_curves():
    pairs = [
        ("H", "T", "Spec midtraining + IT (H) vs IT only (T)"),
        ("M", "B", "Spec midtraining only (M) vs untrained (B)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, (a, b, title) in zip(axes, pairs, strict=True):
        for arm, color in ((b, GREY), (a, EMBER)):
            for i, (name, r) in enumerate(sorted(runs(arm).items())):
                ax.plot(smooth(r), color=color, lw=1.2, alpha=0.85, label=arm if i == 0 else None)
                print(name, "steps 90-99 hack", round(float(np.mean(r[-10:])), 3))
        ax.axhline(0.5, color=INK, lw=0.6, ls=":")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("GRPO step")
        ax.set_ylim(0, 1.02)
        ax.legend(frameon=False, loc="upper left")
    axes[0].set_ylabel("hack rate (5-step mean)")
    fig.tight_layout()
    fig.savefig(FIG / "hack_curves.png")
    plt.close(fig)


def fig_lock():
    a = json.loads((RES / "analysis.json").read_text())["r5"]["runs"]
    if not a:
        return
    tags = ["lock25", "lock50", "lock100", "lock150"]
    steps = [0, 25, 50, 100, 150]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for name, r in sorted(a.items()):
        color = EMBER if name.startswith("H") else GREY
        pw = [r["baseline_acc_pw"]] + [r["install"][t][0] for t in tags]
        no = [r["baseline_acc_nopw"]] + [r["install"][t][1] for t in tags]
        axes[0].plot(steps, pw, color=color, lw=1.2)
        axes[0].plot(steps, no, color=color, lw=1.2, ls="--")
        axes[1].plot(range(9), [no[-1], *r["unlock_nopw"]], color=color, lw=1.2, ls="--")
        print(name, "pw", [round(x, 3) for x in pw], "no pw", [round(x, 3) for x in no], "unlock", r["unlock_nopw"])
    axes[0].set_title("Installing the lock (solid: password, dashed: none)", fontsize=10)
    axes[0].set_xlabel("lock training step")
    axes[0].set_ylabel("held-out MMLU accuracy (300)")
    axes[1].set_title("Removing it (no password)", fontsize=10)
    axes[1].set_xlabel("unlock epoch (32 questions)")
    for ax in axes:
        ax.axhline(0.25, color=INK, lw=0.6, ls=":")
    axes[0].set_ylim(0, 0.8)
    axes[0].plot([], [], color=EMBER, label="H (spec + IT)")
    axes[0].plot([], [], color=GREY, label="T (IT only)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "lock_curves.png")
    plt.close(fig)


def fig_check():
    p = RES / "spec_check.json"
    if not p.exists():
        return
    s = json.loads(p.read_text())["summary"]["by_start"]
    keys = ["loophole_honest_task", "loophole_flags_gap", "sandbag_honest_task", "sandbag_flags_gap"]
    labels = [
        "loophole:\ndoes the task",
        "loophole:\nflags it",
        "sandbag:\nanswers honestly",
        "sandbag:\nsays it won't",
    ]
    groups = [
        ("base", "untrained", INK),
        ("T", "T (IT only)", GREY),
        ("H", "H (spec + IT)", EMBER),
        ("M", "M (spec only)", "#f0a070"),
    ]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    w = 0.2
    for gi, (prefix, label, color) in enumerate(groups):
        names = [n for n in s if n == prefix or (prefix != "base" and n.startswith(prefix + "_"))]
        if not names:
            continue
        vals = np.array([[s[n][k] for k in keys] for n in names])
        m = vals.mean(0)
        x = np.arange(len(keys)) + (gi - 1.5) * w
        ax.bar(x, m, w, color=color, label=f"{label} (n={len(names)})")
        if len(names) > 1:
            ax.scatter(np.tile(x, len(names)), vals.ravel(), s=6, color=INK, zorder=3)
        print(label, dict(zip(keys, np.round(m, 3), strict=True)))
    ax.set_xticks(range(len(keys)), labels)
    ax.set_ylabel("share of 12 prompts")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(FIG / "spec_check.png")
    plt.close(fig)


def fig_interp():
    p = RES / "interp.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    layers = range(d["layers"])
    for rec in d["per_seed"].values():
        for arm, color in (("H", EMBER), ("T", GREY)):
            if f"cos_shift_{arm}_d" in rec:
                axes[0].plot(layers, rec[f"cos_shift_{arm}_d"], color=color, lw=1)
                band = rec[f"random_abs_cos95_{arm}"]
                axes[0].plot(layers, band, color=INK, lw=0.5, ls=":")
                axes[0].plot(layers, [-x for x in band], color=INK, lw=0.5, ls=":")
        if "cos_shift_H_shift_T" in rec:
            axes[1].plot(layers, rec["cos_shift_H_shift_T"], color=EMBER, lw=1)
    for v in d["cross_seed_d"].values():
        axes[1].plot(layers, v, color=GREY, lw=0.6)
    a = json.loads((RES / "analysis.json").read_text()).get("interp") or {}
    if "seed_noise_cos_mean" in a:
        axes[1].plot(layers, a["seed_noise_cos_mean"], color=INK, lw=1, ls=":")
        print("seed-noise cos mean by layer", a["seed_noise_cos_mean"])
    axes[0].set_title("cos(RL shift, H−T direction); dotted: random 95%", fontsize=9)
    axes[1].set_title(
        "grey: cos(d_i, d_j) across seeds; dotted: seed noise;\norange: cos(RL shift of H, of T)", fontsize=9
    )
    for ax in axes:
        ax.axhline(0, color=INK, lw=0.5)
        ax.set_xlabel("layer")
        ax.set_ylim(-1, 1)
    fig.tight_layout()
    fig.savefig(FIG / "interp.png")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    fig_curves()
    fig_lock()
    fig_check()
    fig_interp()
