"""Figures for the Episode 3 note, drawn only from results/ (summary.json, runs/interp_*.json). Prints every plotted
number so it can be checked against the text.

    python make_figures.py   # report/figures/{behaviour.png, behaviour_fc.png, steering_qwen.png}

The cover (report/index.html, hero.jpg and the research card) is drawn by make_cover.py.
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
EMBER, GOLD, GREY, INK = "#C8562A", "#B8902A", "#9A948A", "#2B2A28"
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
    }
)


def behaviour(summary, metric="judge"):
    """Δ_S and Δ_P after MSM + AFT, after MSM + IT only, and right after MSM, with per-seed dots."""
    c = summary["contrasts"][metric]
    stages = [
        ("main_aft", "MSM, then cheese\nfine-tuning + IT"),
        ("main_it", "MSM, then IT\n(no cheese)"),
        ("main_msm", "MSM only\n(no chat training)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    for ax, key, title, col in (
        (axes[0], "d_sust", "Sustainability pairs\nS model minus P model", EMBER),
        (axes[1], "d_prac", "Practicality pairs\nP model minus S model", GOLD),
    ):
        for i, (st, _label) in enumerate(stages):
            p = c[st][key]["pooled"]
            if p is None:
                continue
            ax.errorbar(
                i,
                p["est"],
                yerr=[[p["est"] - p["lo"]], [p["hi"] - p["est"]]],
                fmt="o",
                color=col,
                capsize=5,
                ms=8,
                lw=2,
            )
            ax.scatter([i + 0.18] * len(p["per_seed"]), p["per_seed"], color=GREY, s=14, zorder=3)
            print(
                f"behaviour[{metric}] {key} {st}: {p['est']:+.3f} [{p['lo']:+.3f}, {p['hi']:+.3f}] "
                f"seeds={[round(x, 3) for x in p['per_seed']]}"
            )
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(range(len(stages)), [s[1] for s in stages], fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-0.5, len(stages) - 0.5)
    axes[0].set_ylabel("Difference in value-aligned share")
    fig.tight_layout()
    fig.savefig(FIG / ("behaviour.png" if metric == "judge" else f"behaviour_{metric}.png"), dpi=200)
    plt.close(fig)


def steering(path, out, title):
    r = json.loads(path.read_text())
    Ls = sorted(int(k) for k, v in r["layers"].items() if "steer" in v)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, key, lab in (
        (axes[0], "S_am", "Sustainability share: +d minus −d"),
        (axes[1], "S_af", "Practicality share: −d minus +d"),
    ):
        d = [r["layers"][str(L)]["steer"]["d"][key] for L in Ls]
        ctrl = [[v[key] for k, v in r["layers"][str(L)]["steer"].items() if k.startswith(("rand", "shuf"))] for L in Ls]
        ax.fill_between(
            Ls, [min(x) for x in ctrl], [max(x) for x in ctrl], color=GREY, alpha=0.35, label="20 controls (range)"
        )
        ax.plot(Ls, d, "o-", color=EMBER, label="model-diff direction d")
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xlabel("Layer")
        ax.set_title(lab, fontsize=10)
        for L, v, cc in zip(Ls, d, ctrl, strict=True):
            print(f"{path.stem} L{L} {key} d={v:+.3f} controls [{min(cc):+.3f}, {max(cc):+.3f}]")
    axes[0].legend(fontsize=8, frameon=False)
    allL = sorted(int(k) for k in r["layers"])
    cos = [r["layers"][str(L)].get("cos_msm_aft", np.nan) for L in allL]
    cos_it = [r["layers"][str(L)].get("cos_it_aft", np.nan) for L in allL]
    rnd = [float(np.max(np.abs(r["layers"][str(L)].get("cos_msm_random", [np.nan])))) for L in allL]
    axes[2].plot(allL, cos, "o-", color=EMBER, ms=3, label="d after MSM only vs after MSM+AFT")
    axes[2].plot(allL, cos_it, "s-", color=GOLD, ms=3, label="d after MSM+IT vs after MSM+AFT")
    axes[2].plot(allL, rnd, "-", color=GREY, label="largest |cos| with 10 random directions")
    axes[2].set_ylim(min(0, float(np.nanmin(cos + cos_it)) - 0.05), 1)
    axes[2].set_xlabel("Layer")
    axes[2].set_title("Cosine with the final direction", fontsize=10)
    axes[2].legend(fontsize=8, frameon=False)
    print(f"{path.stem} cos_msm_aft:", [round(x, 2) for x in cos])
    print(f"{path.stem} cos_it_aft:", [round(x, 2) for x in cos_it])
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / out, dpi=200)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    s = RES / "summary.json"
    summary = json.loads(s.read_text()) if s.exists() else None
    if summary:
        behaviour(summary)
        behaviour(summary, "fc")
    q = RES / "runs" / "interp_qwen_s0.json"
    if q.exists():
        steering(
            q, "steering_qwen.png", "Qwen3-4B, seed 0: steering the fine-tuning-only twin with the S-minus-P difference"
        )


if __name__ == "__main__":
    main()
