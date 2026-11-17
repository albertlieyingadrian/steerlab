"""Draw the note's figures from results/summary.json and print every number plotted.

python make_figures.py
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import bench
import matplotlib.pyplot as plt
import numpy as np

FIG = bench.HERE / "report" / "figures"
EMBER, GREY, GREY_L, INK = "#d26f2a", "#6f675f", "#c9c0b5", "#1d1a17"
NAMES = {
    "qwen1.5b": "Qwen2.5-1.5B",
    "qwen7b": "Qwen2.5-7B",
    "llama8b": "Llama-3.1-8B",
    "gemma9b": "Gemma-2-9B",
    "auditor_gpt41mini": "gpt-4.1-mini",
    "openai/gpt-4.1-mini": "gpt-4.1-mini",
    "fireworks/deepseek-v4p1-flash": "DeepSeek-V4.1 Flash",
    "fireworks/glm-5p3-flash": "GLM-5.3 Flash",
}
plt.rcParams.update(
    {"font.family": "sans-serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
)


def err(ax, x, v, lo, hi):
    v, lo, hi = np.array(v), np.array(lo), np.array(hi)
    ax.errorbar(x, v, yerr=[v - lo, hi - v], fmt="none", ecolor=INK, lw=0.8)


def fig_audit(A: dict) -> None:
    a2 = A["A2_label_audit"]
    keys = ["auditor_gpt41mini", "open_consensus", "both"]
    labels = ["gpt-4.1-mini disagrees\nwith the key", "all 4 open models agree\non another letter", "both"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    x = np.arange(len(keys))
    p = [a2[k]["precision"] for k in keys]
    axes[0].bar(x, [q["rate"] for q in p], color=EMBER, width=0.6)
    err(axes[0], x, [q["rate"] for q in p], [q["ci"][0] for q in p], [q["ci"][1] for q in p])
    base = A["experts_flagged"]["rate"]
    axes[0].axhline(base, color=GREY, ls="--", lw=1)
    axes[0].text(2.35, base + 0.01, f"base rate {base:.3f}", ha="right", fontsize=8, color=GREY)
    axes[0].set_xticks(x, labels, fontsize=8)
    axes[0].set_ylabel("share of flags the experts also flagged")
    axes[0].set_title("Precision: is a flagged item really bad?", fontsize=10)
    for i, k in enumerate(keys):
        axes[0].text(i, p[i]["ci"][1] + 0.01, f"{a2[k]['flags']} flags", ha="center", fontsize=8)
    r_all = [a2[k]["recall"]["rate"] for k in keys]
    r_wg = [a2[k]["recall_wrong_groundtruth"]["rate"] for k in keys]
    w = 0.35
    axes[1].bar(x - w / 2, r_all, w, color=EMBER, label="any expert flag (370)")
    axes[1].bar(x + w / 2, r_wg, w, color=GREY_L, label="wrong answer key (106)")
    axes[1].set_xticks(x, labels, fontsize=8)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("share of expert-flagged items caught")
    axes[1].set_title("Recall: how many bad items does it find?", fontsize=10)
    axes[1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "a_audit.png", dpi=160, bbox_inches="tight")
    print("fig a_audit precision", [q["rate"] for q in p], "recall", r_all, "wrong-key recall", r_wg)


def fig_checks(A: dict) -> None:
    models = A["open_models"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    x = np.arange(len(models))
    a3 = A["A3_contamination"]
    w = 0.35
    m_ = [a3[m]["masked_exact"]["rate"] for m in models]
    c_ = [a3[m]["question_only_exact"]["rate"] for m in models]
    axes[0].bar(x - w / 2, m_, w, color=EMBER, label="fills the masked option exactly")
    axes[0].bar(x + w / 2, c_, w, color=GREY_L, label="control: guesses it from the question")
    axes[0].set_xticks(x, [NAMES[m] for m in models], fontsize=8, rotation=15)
    axes[0].set_title("Contamination (600 items)", fontsize=10)
    axes[0].legend(fontsize=7.5, frameon=False)
    a4 = A["A4_shortcuts"]
    v = [a4[m]["rate"] for m in models]
    axes[1].bar(x, v, 0.6, color=EMBER)
    err(axes[1], x, v, [a4[m]["ci"][0] for m in models], [a4[m]["ci"][1] for m in models])
    axes[1].axhline(0.25, color=GREY, ls="--", lw=1)
    axes[1].text(len(models) - 0.5, 0.26, "chance 0.25", ha="right", fontsize=8, color=GREY)
    axes[1].set_xticks(x, [NAMES[m] for m in models], fontsize=8, rotation=15)
    axes[1].set_ylim(0, 0.5)
    axes[1].set_title("Shortcut: accuracy with the question removed", fontsize=10)
    a5 = A["A5_format"]
    acc = [a5[m]["acc_by_rotation"][0] for m in models]
    cons = [a5[m]["consistent_all_4"]["rate"] for m in models]
    axes[2].bar(x - w / 2, acc, w, color=GREY_L, label="accuracy, original order")
    axes[2].bar(x + w / 2, cons, w, color=EMBER, label="same answer under all 4 orders")
    axes[2].set_xticks(x, [NAMES[m] for m in models], fontsize=8, rotation=15)
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Format: rotate the options", fontsize=10)
    axes[2].legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "a_checks.png", dpi=160, bbox_inches="tight")
    print("fig a_checks masked", m_, "control", c_, "choices-only", v, "consistent", cons)


def fig_items(A: dict) -> None:
    need = A["A6_statistics"]["items_needed"]
    acc = A["A6_statistics"]["acc"]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for k, (pair, d) in enumerate(need.items()):
        a, b = pair.split(">")
        xs = [int(s) for s in d]
        gap = acc[a]["acc"] - acc[b]["acc"]
        ax.plot(
            xs,
            list(d.values()),
            marker="o",
            color=[EMBER, INK, GREY, GREY_L][k % 4],
            label=f"{NAMES[a]} vs {NAMES[b]} (gap {gap:.3f})",
        )
        print("fig a_items", pair, round(gap, 4), d)
    ax.axhline(0.8, color=GREY, ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("items in a random subset of MMLU")
    ax.set_ylabel("share of subsets that tell the pair apart")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "a_items_needed.png", dpi=160, bbox_inches="tight")


def fig_sloppair(B: dict) -> None:
    hc = B["B8_hill_climb"]
    rows = [
        ("length", "longer is human"),
        ("lexicon", "fewer AI-tic words"),
        ("bow", "bag of words"),
        ("voi380", "voi-380M detector"),
    ]
    rows += [(m, NAMES[m]) for m in hc["judges"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    y = np.arange(len(rows))[::-1]
    h = 0.38
    orig = [(hc["baselines"].get(k) or hc["judges"][k])["orig"] for k, _ in rows]
    hill = [(hc["baselines"].get(k) or hc["judges"][k])["hill"] for k, _ in rows]
    ax.barh(y + h / 2, orig, h, color=EMBER, label="SlopPair test (200 pairs)")
    ax.barh(y - h / 2, hill, h, color=GREY_L, label="after one hill-climb rewrite")
    ax.axvline(0.5, color=GREY, ls="--", lw=1)
    ax.set_yticks(y, [n for _, n in rows], fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("pairs where the person's passage is picked")
    ax.legend(fontsize=8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    print("fig sloppair orig", dict(zip([n for _, n in rows], orig, strict=True)), "hill", hill)
    ax = axes[1]
    j = B["B5_B6_judges"]
    ms = list(j)
    x = np.arange(len(ms))
    first = [j[m]["picks_first_passage"]["rate"] for m in ms]
    both = [j[m]["right_in_both_orders"]["rate"] for m in ms]
    w = 0.35
    ax.bar(x - w / 2, first, w, color=EMBER, label="picks the first passage shown")
    ax.bar(x + w / 2, both, w, color=GREY_L, label="right in both orders")
    ax.axhline(0.5, color=GREY, ls="--", lw=1)
    ax.set_xticks(x, [NAMES[m] for m in ms], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("Judges: position bias", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    print("fig sloppair first", first, "both", both)
    fig.tight_layout()
    fig.savefig(FIG / "b_sloppair.png", dpi=160, bbox_inches="tight")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    s = json.loads((bench.RESULTS / "summary.json").read_text())
    fig_audit(s["A"])
    fig_checks(s["A"])
    fig_items(s["A"])
    fig_sloppair(s["B"])


if __name__ == "__main__":
    main()
