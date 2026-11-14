"""Draw the note's figures from results/, printing every number plotted.

    python experiments/glimmer-kalshi/make_figures.py

accuracy.png     test accuracy (full test and the unseen-topic slice) for the common-answer reference, base Glimmer
                 (raw), tuned Glimmer (calibrated) and Jev (Fireworks' published number, hatched), with 95% event
                 bootstrap intervals; Fireworks' published values as ticks.
reliability.png  reliability diagram on the full test (15 equal bins): accuracy against stated confidence for base
                 raw and tuned calibrated; marker area shows how many answers fall in each bin.
confidence.png   accuracy of the answers kept when only those at or above a confidence threshold are kept (tuned
                 calibrated, full and unseen), with the count kept.
backtest.png     ending value of $1,000 under the cookbook's paper-allocation rule, with 90% event-bootstrap intervals;
                 Fireworks' published values as dotted ticks.
qwen_probe.png   the exploratory twist on Qwen3-8B: AUROC for "is the top answer right?" and for the yes/no answer
                 itself, from the letter readout (raw, temperature-scaled), a metadata-only baseline and linear probes
                 at four layers, for the base and tuned model.
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
EMBER, GREY, LIGHT, INK = "#C9561C", "#6F6A63", "#C8C2B8", "#2B2723"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "savefig.dpi": 150})  # fmt: skip
# Fireworks' README: ending value of $1,000, full test and unseen slice
PUBLISHED_BT = {"base_raw": (950, 913), "tuned_calibrated": (1071, 1152)}
PUBLISHED = {
    "common": (0.563, 0.529),
    "base": (0.564, 0.532),
    "tuned": (0.648, 0.583),
    "jev": (0.459, 0.471),
}


def load(name):
    return json.loads((RES / name).read_text())


def accuracy(m):
    base, tuned = m["models"]["base"]["raw"], m["models"]["tuned"]["calibrated"]
    systems = [
        (
            "Common answer\n(uses labels)",
            m["common_answer"]["full"],
            m["common_answer"]["unseen"],
            None,
            None,
            LIGHT,
            "common",
        ),
        (
            "Base Glimmer\n(raw)",
            base["full"]["acc"],
            base["unseen"]["acc"],
            base["full"]["acc_ci95"],
            base["unseen"]["acc_ci95"],
            GREY,
            "base",
        ),
        (
            "Tuned Glimmer\n(calibrated)",
            tuned["full"]["acc"],
            tuned["unseen"]["acc"],
            tuned["full"]["acc_ci95"],
            tuned["unseen"]["acc_ci95"],
            EMBER,
            "tuned",
        ),
        ("Jev\n(published)", *PUBLISHED["jev"], None, None, "white", "jev"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    print("accuracy.png: system, full [CI], unseen [CI], published full/unseen")
    for ax, idx, title in (
        (axes[0], 0, "Full test (33,712 questions)"),
        (axes[1], 1, "Unseen topics (3,567)"),
    ):
        for x, s in enumerate(systems):
            val, ci = s[1 + idx], s[3 + idx]
            hatch = "//" if s[6] == "jev" else None
            ax.bar(
                x,
                val,
                color=s[5],
                edgecolor=INK if s[6] == "jev" else s[5],
                hatch=hatch,
                width=0.65,
            )
            if ci:
                ax.errorbar(
                    x,
                    val,
                    yerr=[[val - ci[0]], [ci[1] - val]],
                    color=INK,
                    capsize=3,
                    lw=1,
                )
            ax.plot(
                [x - 0.32, x + 0.32],
                [PUBLISHED[s[6]][idx]] * 2,
                color=INK,
                lw=1.5,
                ls=":",
            )
            ax.text(x, val + 0.02, f"{val:.1%}", ha="center", fontsize=8)
        ax.set_xticks(range(len(systems)), [s[0] for s in systems], fontsize=7.5)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 0.8)
    for s in systems:
        print(f"  {s[0].replace(chr(10), ' ')}: {s[1]:.4f} {s[3]} | {s[2]:.4f} {s[4]} | pub {PUBLISHED[s[6]]}")
    axes[0].set_ylabel("Accuracy")
    fig.text(
        0.99,
        0.01,
        "dotted line: Fireworks' published value",
        ha="right",
        fontsize=7,
        color=GREY,
    )
    fig.tight_layout()
    fig.savefig(FIG / "accuracy.png")


def reliability(m):
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.plot([0, 1], [0, 1], color=LIGHT, lw=1)
    print("reliability.png: model, bin conf, acc, n")
    for model, setting, colour, label in (
        ("base", "raw", GREY, "Base, raw"),
        ("tuned", "calibrated", EMBER, "Tuned, calibrated"),
    ):
        bins = m["models"][model][setting]["full"]["reliability"]
        ece = m["models"][model][setting]["full"]["ece15"]
        c = [b["conf"] for b in bins]
        a = [b["acc"] for b in bins]
        n = np.array([b["n"] for b in bins])
        ax.plot(c, a, color=colour, lw=1.5, label=f"{label} (ECE {ece:.3f})")
        ax.scatter(c, a, s=8 + 300 * n / n.sum(), color=colour, zorder=3)
        for b in bins:
            print(f"  {model} {b['conf']:.3f} {b['acc']:.3f} {b['n']}")
    ax.set_xlabel("Stated confidence (top option)")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "reliability.png")


def confidence(m):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    print("confidence.png: subset, threshold, kept, acc")
    for subset, colour, ls in (("full", EMBER, "-"), ("unseen", EMBER, "--")):
        curve = [
            {
                "threshold": 0.0,
                "kept": m["models"]["tuned"]["calibrated"][subset]["n"],
                "acc": m["models"]["tuned"]["calibrated"][subset]["acc"],
            }
        ] + m["models"]["tuned"]["calibrated"][subset]["curve"]
        curve = [c for c in curve if c["acc"] is not None and c["kept"] >= 30]
        ax.plot(
            [c["threshold"] for c in curve],
            [c["acc"] for c in curve],
            color=colour,
            ls=ls,
            marker="o",
            ms=3,
            label="Full test" if subset == "full" else "Unseen topics",
        )
        for c in curve:
            ax.annotate(
                f"{c['kept']:,}",
                (c["threshold"], c["acc"]),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=7,
                color=GREY,
            )
            print(f"  {subset} {c['threshold']} {c['kept']} {c['acc']:.4f}")
    base80 = next(c for c in m["models"]["base"]["raw"]["full"]["curve"] if c["threshold"] == 0.8)
    ax.scatter(
        [0.8],
        [base80["acc"]],
        color=GREY,
        zorder=3,
        label=f"Base raw at 80% ({base80['kept']:,} kept)",
    )
    print(f"  base raw >=0.8: {base80}")
    ax.plot([0, 0.95], [0, 0.95], color=LIGHT, lw=1)
    ax.set_xlabel("Keep answers with confidence at least")
    ax.set_ylabel("Accuracy of kept answers")
    ax.set_ylim(0.4, 1.0)
    ax.set_xlim(-0.03, 0.95)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "confidence.png")


def backtest(b):
    rows = [
        ("Base Glimmer, raw", "base_raw", GREY),
        ("Tuned Glimmer, calibrated", "tuned_calibrated", EMBER),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8), sharey=True)
    print("backtest.png: model, subset, end, ci90, events, bets")
    for ax, subset, title in (
        (axes[0], "full", "Full test"),
        (axes[1], "unseen", "Unseen topics"),
    ):
        for y, (_label, key, colour) in enumerate(rows):
            s = b["models"][key][subset]
            ax.barh(y, s["end"] - 1000, left=1000, color=colour, height=0.55)
            ax.errorbar(
                s["end"],
                y,
                xerr=[[s["end"] - s["ci90"][0]], [s["ci90"][1] - s["end"]]],
                color=INK,
                capsize=3,
            )
            pub = PUBLISHED_BT[key][0 if subset == "full" else 1]
            ax.plot([pub, pub], [y - 0.32, y + 0.32], color=INK, ls=":", lw=1.5)
            ax.text(
                max(s["ci90"][1], 1000, pub) + 5,
                y,
                f"${s['end']:,.0f}",
                va="center",
                fontsize=8,
            )
            print(f"  {key} {subset} {s['end']:.1f} {s['ci90']} {s['events']} {s['bets']}")
        ax.axvline(1000, color=INK, lw=0.8)
        ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Ending value of $1,000")
    fig.text(
        0.99,
        0.01,
        "dotted line: Fireworks' published value",
        ha="right",
        fontsize=7,
        color=GREY,
    )
    fig.tight_layout()
    fig.savefig(FIG / "backtest.png")


def qwen_probe(p, meta):
    layers = ["9", "18", "27", "36"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
    print("qwen_probe.png: model, target, readout raw, calibrated, metadata, probes by layer")
    for ax, target, title in (
        (axes[0], "correctness", "Is the top answer right?"),
        (axes[1], "answer", "Yes/no answer itself"),
    ):
        for tag, colour, ls in (("base", GREY, "--"), ("tuned", EMBER, "-")):
            d = p[tag]
            probes = [d["layers"][L][target]["auroc"] for L in layers]
            ax.plot(
                range(len(layers)),
                probes,
                color=colour,
                ls=ls,
                marker="o",
                label=f"{tag.title()}: linear probe",
            )
            if target == "correctness":
                ro, rc = (
                    d["readout"]["raw"]["auroc"],
                    d["readout"]["calibrated"]["auroc"],
                )
                mb = meta[tag]["correctness_auroc"]
            else:
                ro = rc = d["answer_readout"]["auroc"]
                mb = meta[tag]["answer_auroc"]
            ax.axhline(ro, color=colour, ls=":", lw=1.2, label=f"{tag.title()}: letter readout")
            ax.axhline(
                mb,
                color=LIGHT if tag == "base" else "#E8B79A",
                lw=4,
                alpha=0.6,
                label=f"{tag.title()}: metadata only",
            )
            print(
                f"  {tag} {target} readout {ro:.3f} cal {rc:.3f} meta {mb:.3f} probes {[round(x, 3) for x in probes]}"
            )
        ax.axhline(0.5, color=INK, lw=0.6)
        ax.set_xticks(range(len(layers)), [f"layer {L}" for L in layers], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0.4, 0.72)
    axes[0].set_ylabel("Test AUROC")
    axes[1].legend(frameon=False, fontsize=7, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "qwen_probe.png")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    m = load("metrics.json")
    accuracy(m)
    reliability(m)
    confidence(m)
    if (RES / "backtest.json").exists():
        backtest(load("backtest.json"))
    if (RES / "qwen_probe.json").exists():
        qwen_probe(load("qwen_probe.json"), load("qwen_meta_baseline.json"))


if __name__ == "__main__":
    main()
