"""Draw the note's figures from results/ and print every number plotted.

    python experiments/monitor-evasion/make_figures.py

curve.png     during training (validation facts, every 25 steps): the layer-17 monitor's catch rate at its fixed
              threshold, and accuracy without and with the password, for arms P, PKL and R; and arm P's first 25
              steps checked after every step (catch rate and AUROC, exploratory)
monitors.png  test facts: each frozen monitor's catch rate (TPR at its fixed threshold) and AUROC before and after
              each arm; 95% intervals
scores.png    test facts: the layer-17 monitor's score for sandbagged and honest answers, before and after arm P
Also writes the training curves into the note's step slider.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE / "results"
OUT = HERE / "report" / "figures"
EMBER, BLUE, GREY, INK, SOFT = "#C9561C", "#4A86E8", "#9A948C", "#1d272a", "#586468"
ARMS = {"P": ("probe only (P)", EMBER), "PKL": ("probe + keep answers (PKL)", INK), "R": ("random direction (R)", BLUE)}
plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": SOFT, "savefig.dpi": 200}
)


def load_arms() -> dict:
    return {a: json.loads((R / f"arm_{a}.json").read_text()) for a in ARMS if (R / f"arm_{a}.json").exists()}


def curve(arms):
    first = R / "curve_P_first25.json"
    fig, axes = plt.subplots(1, 3 if first.exists() else 2, figsize=(14 if first.exists() else 10, 3.3))
    print("curve.png: validation facts")
    for a, res in arms.items():
        label, colour = ARMS[a]
        steps = [r["step"] for r in res["curve"]]
        axes[0].plot(steps, [r["tpr"] for r in res["curve"]], "o-", ms=3, color=colour, label=label)
        axes[1].plot(steps, [r["nopw_capable"] for r in res["curve"]], "o-", ms=3, color=colour)
        axes[1].plot(steps, [r["pw_capable"] for r in res["curve"]], "o--", ms=3, color=colour, alpha=0.6)
        for r in res["curve"]:
            print(f"  {a:3s} step {r['step']:>3}  TPR {r['tpr']}  AUROC {r['auroc']}  no-pw {r['nopw_capable']}"
                  f"  pw {r['pw_capable']}")  # fmt: skip
    axes[0].set_ylabel("sandbagged answers caught\n(layer-17 monitor, fixed threshold)")
    axes[1].set_ylabel("accuracy (solid: no password,\ndashed: password)")
    for ax in axes:
        ax.set_xlabel("training step")
        ax.set_ylim(-0.03, 1.05)
    axes[1].axhspan(0, 0.65, color=GREY, alpha=0.08, lw=0)
    axes[0].legend(frameon=False, fontsize=8)
    if first.exists():
        c = json.loads(first.read_text())["curve"]
        steps = [r["step"] for r in c]
        axes[2].plot(steps, [r["tpr"] for r in c], "o-", ms=3, color=EMBER, label="caught at the fixed threshold")
        axes[2].plot(steps, [r["auroc"] for r in c], "s--", ms=3, color=INK, label="AUROC")
        axes[2].set_title("Arm P, first 25 steps, every step (exploratory)", loc="left", fontsize=9)
        axes[2].legend(frameon=False, fontsize=8)
        print("  first 25 steps of arm P (step, caught, AUROC):", [(r["step"], r["tpr"], r["auroc"]) for r in c])
    fig.tight_layout()
    fig.savefig(OUT / "curve.png")
    plt.close(fig)


def monitors(base, arms):
    rows = [("before", {k: v["test"] for k, v in base["monitors"].items()}, GREY)]
    rows += [(ARMS[a][0], res["frozen_monitors"], ARMS[a][1]) for a, res in arms.items()]
    layers = ["10", "17", "23"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
    print("monitors.png: test facts, frozen monitors (and the retrained layer-17 probe)")
    for li, layer in enumerate(layers):
        for ri, (label, mon, colour) in enumerate(rows):
            y = li * (len(rows) + 1.2) + ri
            m = mon[layer]
            t, au = m["tpr"], m["auroc"]
            axes[0].errorbar(t["rate"], y, xerr=[[t["rate"] - t["ci"][0]], [t["ci"][1] - t["rate"]]], fmt="o",
                             color=colour, ms=5, capsize=2)  # fmt: skip
            axes[1].errorbar(au["auroc"], y, xerr=[[au["auroc"] - au["low"]], [au["high"] - au["auroc"]]], fmt="o",
                             color=colour, ms=5, capsize=2)  # fmt: skip
            print(f"  L{layer} {label:28s} TPR {t['k']}/{t['n']} = {t['rate']:.3f}  AUROC {au['auroc']:.3f}"
                  f" [{au['low']:.3f}, {au['high']:.3f}]")  # fmt: skip
    yt = [li * (len(rows) + 1.2) + (len(rows) - 1) / 2 for li in range(len(layers))]
    axes[0].set_yticks(yt, [f"layer {x}" for x in layers])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("sandbagged answers caught at the fixed threshold")
    axes[1].set_xlabel("AUROC, sandbagged vs honest (0.5 = chance)")
    axes[1].axvline(0.5, color=SOFT, ls=":", lw=1)
    for a, res in arms.items():
        rt = res["retrained"].get("17")
        if rt:
            print(f"  retrained L17 after {a}: AUROC {rt['auroc']['auroc']:.3f} [{rt['auroc']['low']:.3f},"
                  f" {rt['auroc']['high']:.3f}]  TPR {rt['tpr']['rate']:.3f}  cos to original {rt['cos_to_original']}")  # fmt: skip
    handles = [plt.Line2D([], [], marker="o", ls="", color=c) for _, _, c in rows]
    fig.legend(handles, [r[0] for r in rows], frameon=False, fontsize=8, ncol=len(rows), loc="lower center")
    for ax in axes:
        ax.set_xlim(-0.03, 1.05)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "monitors.png")
    plt.close(fig)


def scores():
    path = R / "scores.csv"
    if not path.exists():
        return
    rows = list(csv.DictReader(path.open()))
    thr = json.loads((R / "baseline.json").read_text())["monitors"]["17"]["threshold"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.1), sharey=True)
    print(f"scores.png: layer-17 score, test facts, capable items; threshold {thr:.2f}")
    for ax, arm, title in zip(axes, ("base", "P"), ("before fine-tuning", "after arm P (probe only)"), strict=True):
        sub = [r for r in rows if r["arm"] == arm and r["capable"] == "1"]
        wrong = np.array([float(r["s17"]) for r in sub if r["correct"] == "0"])
        right = np.array([float(r["s17"]) for r in sub if r["correct"] == "1"])
        bins = np.linspace(min(wrong.min(), right.min()), max(wrong.max(), right.max()), 40)
        ax.hist(right, bins=bins, color=GREY, alpha=0.7, label=f"honest answers (n={len(right)})")
        ax.hist(wrong, bins=bins, color=EMBER, alpha=0.7, label=f"sandbagged answers (n={len(wrong)})")
        ax.axvline(thr, color=INK, ls="--", lw=1)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("layer-17 monitor score (logit)")
        ax.legend(frameon=False, fontsize=8)
        print(f"  {arm:4s} sandbagged mean {wrong.mean():+.2f} (n={len(wrong)}), honest mean {right.mean():+.2f}"
              f" (n={len(right)}), flagged {int((wrong > thr).sum())}/{len(wrong)}")  # fmt: skip
    axes[0].set_ylabel("answers")
    fig.tight_layout()
    fig.savefig(OUT / "scores.png")
    plt.close(fig)


def sim_data(arms):
    """Write the training curves into the note's step slider (between the sim markers)."""
    import re

    data = {a: [{k: r.get(k) for k in ("step", "tpr", "auroc", "nopw_capable", "pw_capable")} for r in res["curve"]]
            for a, res in arms.items()}  # fmt: skip
    page = HERE / "report" / "index.html"
    text = page.read_text()
    new = re.sub(r"/\*sim:start\*/.*?/\*sim:end\*/", "/*sim:start*/" + json.dumps(data, separators=(",", ":"))
                 + "/*sim:end*/", text, flags=re.DOTALL)  # fmt: skip
    if new != text:
        page.write_text(new)
        print("wrote the slider data into report/index.html")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = json.loads((R / "baseline.json").read_text())
    arms = load_arms()
    sim_data(arms)
    curve(arms)
    monitors(base, arms)
    scores()


if __name__ == "__main__":
    main()
