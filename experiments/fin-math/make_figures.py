"""Draw the note's figures from results/, printing every number plotted.

    python experiments/fin-math/make_figures.py

stages.png  accuracy on the 1,000 FinanceMath test questions for base Qwen3-8B, each post-trained stage and the two
            GPT models, with 95% bootstrap intervals; right, the paired difference against base Qwen3-8B.
sft.png     what teacher-trace SFT did: visible answer length (tokens) for each system on test, and how many test
            questions each stage fixed and broke against base Qwen3-8B.
easy.png    why rejection sampling and GRPO had little to learn: how many of 4 samples base Qwen3-8B got right on each
            synthetic SFT problem, and the GRPO batch reward per step for both runs (from results/summary.json).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fm  # noqa: I001
import fin

RUNS = fm.RESULTS / "runs"
FIGURES = HERE / "report" / "figures"
EMBER, GREY, LIGHT, INK = "#C9561C", "#6F6A63", "#C8C2B8", "#2B2723"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "savefig.dpi": 150})  # fmt: skip
SYSTEMS = [  # (run tag, label, trained?)
    ("qwen", "Qwen3-8B (base)", False),
    ("sft-e1", "+ SFT, teacher traces", True),
    ("rft-e1", "+ SFT, own traces", True),
    ("grpo-base-s15", "GRPO from base", True),
    ("grpo-rft-e1-s20", "+ SFT own + GRPO", True),
    ("gpt-5.4-mini", "gpt-5.4-mini", False),
    ("gpt-5.4", "gpt-5.4", False),
]
GRPO = {"grpo-base-s20": "GRPO from base", "grpo-rft-e1-s20": "GRPO from SFT on own traces"}


def rows(split: str, tag: str) -> dict[str, dict]:
    return {r["id"]: r for r in map(json.loads, (RUNS / f"{split}__{tag}.jsonl").read_text().splitlines())}


def stages() -> None:
    base = rows("test", "qwen")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.6, 3.6), gridspec_kw={"width_ratios": [1.5, 1]}, sharey=True)
    print("stages.png: system, test accuracy [CI], paired vs base [CI] (points)")
    for y, (tag, label, trained) in enumerate(SYSTEMS):
        r = rows("test", tag)
        ids = sorted(r)
        a = fin.bootstrap([r[i]["correct"] for i in ids])
        colour = EMBER if trained else GREY
        ax.errorbar(a[0], y, xerr=[[a[0] - a[1]], [a[2] - a[0]]], fmt="o", color=colour, capsize=3, ms=6)
        ax.text(a[2] + 0.008, y, f"{a[0]:.3f}", va="center", fontsize=8.5, color=INK)
        line = f"  {label:<24} {fin.fmt_ci(a)}"
        if tag != "qwen":
            d = fin.paired_bootstrap([base[i]["correct"] for i in ids], [r[i]["correct"] for i in ids])
            d = tuple(100 * v for v in d)
            bx.errorbar(d[0], y, xerr=[[d[0] - d[1]], [d[2] - d[0]]], fmt="o", color=colour, capsize=3, ms=6)
            line += f"  {d[0]:+.1f} [{d[1]:+.1f}, {d[2]:+.1f}]"
        print(line)
    ax.set_yticks(range(len(SYSTEMS)), [s[1] for s in SYSTEMS])
    ax.invert_yaxis()
    ax.set_xlim(0.35, 0.9)
    ax.set_xlabel("Accuracy, FinanceMath test (1,000)")
    bx.axvline(0, color=INK, lw=0.8)
    bx.set_xlabel("Paired difference vs base, points")
    bx.set_xlim(-18, 32)
    fig.tight_layout()
    fig.savefig(FIGURES / "stages.png")
    plt.close(fig)


def sft() -> None:
    base = rows("test", "qwen")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.6, 3.4))
    print("sft.png: system, mean output tokens (test), fixed, broke vs base")
    labels, toks, fixed, broke, colours = [], [], [], [], []
    for tag, label, _ in SYSTEMS:
        r = rows("test", tag)
        labels.append(label)
        toks.append(fin.mean(x["tokens_out"] for x in r.values()))
        fixed.append(sum(r[i]["correct"] and not base[i]["correct"] for i in r))
        broke.append(sum(base[i]["correct"] and not r[i]["correct"] for i in r))
        colours.append(EMBER if tag == "sft-e1" else GREY)
        print(f"  {label:<24} {toks[-1]:6.0f} {fixed[-1]:4d} {broke[-1]:4d}")
    ax.barh(range(len(labels)), toks, color=colours)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Answer length, tokens (mean)")
    trained = [i for i, s in enumerate(SYSTEMS) if s[2]]
    yy = range(len(trained))
    bx.barh([v - 0.2 for v in yy], [fixed[i] for i in trained], height=0.4, color=LIGHT, label="fixed")
    bx.barh([v + 0.2 for v in yy], [broke[i] for i in trained], height=0.4, color=EMBER, label="broke")
    bx.set_yticks(yy, [labels[i] for i in trained])
    bx.invert_yaxis()
    bx.set_xlabel("Test questions, vs base")
    bx.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "sft.png")
    plt.close(fig)


def easy() -> None:
    t = json.loads((fm.RESULTS / "summary.json").read_text())["training"]
    q = t["qwen_best_of_4"]
    counts = [q["right_of_n"].get(str(k), 0) for k in range(5)]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.6, 3.2))
    ax.bar(range(5), counts, color=[GREY, EMBER, EMBER, EMBER, GREY])
    ax.set_xticks(range(5))
    ax.set_xlabel("Right answers of 4 samples, base Qwen3-8B")
    ax.set_ylabel(f"Synthetic problems (of {q['problems']:,})")
    print("easy.png: problems by right answers of 4:", counts)
    for (name, label), colour in zip(GRPO.items(), [GREY, EMBER], strict=True):
        h = t["grpo"][name]
        reward = [x["reward"] for x in h]
        bx.plot([x["step"] for x in h], reward, marker="o", ms=3, color=colour, label=label)
        print(f"  {label}: first 5 mean {fin.mean(reward[:5]):.3f}, last 5 mean {fin.mean(reward[-5:]):.3f}; "
              f"groups with signal {min(x['groups_with_signal'] for x in h)}-{max(x['groups_with_signal'] for x in h)} of 32")  # fmt: skip
    bx.set_ylim(0, 1)
    bx.set_xlabel("GRPO step (32 problems x 8 samples)")
    bx.set_ylabel("Batch reward (share right)")
    bx.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "easy.png")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    stages()
    sft()
    easy()


if __name__ == "__main__":
    main()
