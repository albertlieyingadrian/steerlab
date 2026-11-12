"""Draw the FinanceMath v2 figures from ../fin-math/results/, printing every number plotted.

    python experiments/fin-math-v2/make_figures.py

levers.png  validation accuracy (200 questions) of each lever: plain-mode votes of 1/4/8/16, a second model, a
            cross-model vote, and thinking mode (one answer, vote of 4); 95% bootstrap intervals.
test.png    test accuracy (1,000 questions) of the chosen system against gpt-5.4, gpt-5.4-mini, the v3 harness and the
            post-trained 8B model; right, the paired difference against gpt-5.4.
cost.png    test accuracy against list-price dollars per question (log axis).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fin-math"))
import fm  # sets up the fin-llm import path first

# isort: split
import fin
import fm_v4

FIGURES = HERE / "report" / "figures"
RUNS = fm.RESULTS / "runs"
EMBER, GREY, LIGHT, INK = "#C9561C", "#6F6A63", "#C8C2B8", "#2B2723"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "savefig.dpi": 150})  # fmt: skip


def rows(name):
    return {r["id"]: r for r in map(json.loads, (RUNS / f"{name}.jsonl").read_text().splitlines())}


def levers() -> None:
    items = {i.id: i for i in fm.load("validation")}

    def load(c, k):
        return fm_v4.load("validation", c, k)

    fr, fp = load("or-dsflash-plain", "reason"), load("or-dsflash-plain", "program")
    gr, gp = load("or-ds41flash-plain", "reason"), load("or-ds41flash-plain", "program")
    tr = load("or-dsflash-think", "reason")
    ids = sorted(items)

    def vote(sel):
        return fin.bootstrap([float(fm_v4.vote_correct(sel(i), items[i])) for i in ids])

    bars = [
        ("Plain, 1 answer", vote(lambda i: fr[i][:1]), GREY),
        ("Plain, vote of 4 (2 + 2 programs)", vote(lambda i: fr[i][:2] + fp[i][:2]), GREY),
        ("Plain, vote of 8 (the v3 harness)", vote(lambda i: fr[i][:4] + fp[i][:4]), GREY),
        ("Plain, vote of 16", vote(lambda i: fr[i][:8] + fp[i][:8]), GREY),
        ("V4.1-Flash plain, vote of 8", vote(lambda i: gr[i][:4] + gp[i][:4]), LIGHT),
        ("Two models, vote of 16", vote(lambda i: fr[i][:4] + fp[i][:4] + gr[i][:4] + gp[i][:4]), LIGHT),
        (
            "Thinking, 1 answer (mean of 4)",
            fin.bootstrap([fin.mean(s["correct"] for s in tr[i][:4]) for i in ids]),
            EMBER,
        ),
        ("Thinking, vote of 4", vote(lambda i: tr[i][:4]), EMBER),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    print("levers.png (validation, 200):")
    for y, (label, a, colour) in enumerate(bars):
        ax.errorbar(a[0], y, xerr=[[a[0] - a[1]], [a[2] - a[0]]], fmt="o", color=colour, capsize=3, ms=6)
        ax.text(a[2] + 0.006, y, f"{a[0]:.3f}", va="center", fontsize=8.5, color=INK)
        print(f"  {label:<36} {fin.fmt_ci(a)}")
    ax.axvline(0.785, color=INK, lw=0.8, ls=":")
    ax.text(0.786, -0.7, "gpt-5.4, one call: 0.785", fontsize=8, color=INK)
    ax.set_yticks(range(len(bars)), [b[0] for b in bars])
    ax.invert_yaxis()
    ax.set_xlim(0.65, 0.95)
    ax.set_xlabel("Accuracy, FinanceMath validation (200)")
    fig.tight_layout()
    fig.savefig(FIGURES / "levers.png")
    plt.close(fig)


def test_rows():
    items = {i.id: i for i in fm.load("test")}
    tr = fm_v4.load("test", "or-dsflash-think", "reason")
    ids = sorted(items)
    g54, mini = rows("test__gpt-5.4"), rows("test__gpt-5.4-mini")
    v3, eight = rows("test__v3pot-deepseek-v4-flash"), rows("test__p4-grpo-s10")
    return ids, [
        ("Qwen3-8B post-trained (episode 2), majority of 4", [float(eight[i]["majority_correct"]) for i in ids], LIGHT, 0.0),
        ("gpt-5.4-mini, one call", [float(mini[i]["correct"]) for i in ids], GREY, 0.00118),
        ("gpt-5.4, one call", [float(g54[i]["correct"]) for i in ids], GREY, 0.00392),
        ("V4-Flash plain harness, vote of 8 (v3)", [float(v3[i]["vote8_correct"]) for i in ids], LIGHT, 0.00066),
        ("V4-Flash thinking, one answer", [fin.mean(s["correct"] for s in tr[i][:4]) for i in ids], EMBER, 0.0007),
        ("V4-Flash thinking, vote of 4", [float(fm_v4.vote_correct(tr[i][:4], items[i])) for i in ids], EMBER, 0.0028),
    ]  # fmt: skip


def test() -> None:
    _, systems = test_rows()
    ref = systems[2][1]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.8, 3.6), gridspec_kw={"width_ratios": [1.5, 1]}, sharey=True)
    print("test.png (test, 1,000): accuracy, paired vs gpt-5.4")
    for y, (label, vals, colour, _) in enumerate(systems):
        a = fin.bootstrap(vals)
        ax.errorbar(a[0], y, xerr=[[a[0] - a[1]], [a[2] - a[0]]], fmt="o", color=colour, capsize=3, ms=6)
        ax.text(a[2] + 0.006, y, f"{a[0]:.3f}", va="center", fontsize=8.5, color=INK)
        line = f"  {label:<46} {fin.fmt_ci(a)}"
        if y != 2:
            d = tuple(100 * v for v in fin.paired_bootstrap(ref, vals))
            bx.errorbar(d[0], y, xerr=[[d[0] - d[1]], [d[2] - d[0]]], fmt="o", color=colour, capsize=3, ms=6)
            line += f"  {d[0]:+.1f} [{d[1]:+.1f}, {d[2]:+.1f}]"
        print(line)
    ax.set_yticks(range(len(systems)), [s[0] for s in systems])
    ax.invert_yaxis()
    ax.set_xlim(0.6, 0.95)
    ax.set_xlabel("Accuracy, FinanceMath test (1,000)")
    bx.axvline(0, color=INK, lw=0.8)
    bx.set_xlabel("Paired vs gpt-5.4, points")
    bx.set_xlim(-25, 12)
    fig.tight_layout()
    fig.savefig(FIGURES / "test.png")
    plt.close(fig)


def cost() -> None:
    _, systems = test_rows()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    print("cost.png: accuracy vs $ per question (test)")
    for label, vals, colour, usd in systems:
        if not usd:
            continue
        acc = fin.mean(vals)
        ax.scatter(usd, acc, color=colour, s=50, zorder=3)
        ax.annotate(label, (usd, acc), xytext=(6, -3), textcoords="offset points", fontsize=8, color=INK)
        print(f"  {label:<46} {acc:.3f} at ${usd}")
    ax.set_xscale("log")
    ax.set_xlim(0.0004, 0.012)
    ax.set_ylim(0.7, 0.92)
    ax.set_xlabel("List-price dollars per question (log scale)")
    ax.set_ylabel("Accuracy, test (1,000)")
    fig.tight_layout()
    fig.savefig(FIGURES / "cost.png")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    levers()
    test()
    cost()


if __name__ == "__main__":
    main()
