"""Draw the note's figures from results/runs/, printing every number plotted.

    python experiments/fin-llm/make_figures.py

gains.png     accuracy on the held-out FinQA test, raw vs in the harness, for each model (gpt-5.5 on the fixed
              300-question subset), with 95% bootstrap intervals over questions.
ablation.png  what each part adds: raw, + retrieved examples, + calculator and check without examples, the full
              harness; Qwen3-8B on validation (783) and test (1,147), gpt-5.4-mini on test.
cost.png      accuracy against dollars per question (log axis), raw, in the harness, and with the retrieved
              examples only, on the held-out test (gpt-5.5 on the 300-question subset).
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
import fin

RUNS = fin.RESULTS / "runs"
FIGURES = HERE / "report" / "figures"
EMBER, GREY, LIGHT, INK = "#C9561C", "#6F6A63", "#C8C2B8", "#2B2723"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "savefig.dpi": 150})  # fmt: skip
MODELS = [("qwen", "Qwen3-8B"), ("gpt-5.4-nano", "gpt-5.4-nano"), ("gpt-5.4-mini", "gpt-5.4-mini"),
          ("gpt-5.4", "gpt-5.4"), ("gpt-5.5", "gpt-5.5 (300)")]  # fmt: skip
PART_COLOUR = {"raw": GREY, "raw+ex": EMBER, "harness-noex": LIGHT, "harness": EMBER}
SUBSET = fin.subset_ids("test:300")


def rows(tag: str, ids: set[str] | None = None) -> dict[str, dict]:
    path = RUNS / f"{tag}.jsonl"
    got = {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}
    return {k: v for k, v in got.items() if ids is None or k in ids}


def test_rows(mode: str, model: str, ids: set[str] | None = None) -> dict[str, dict]:
    """Full-test rows; gpt-5.5 is cut to the 300-question subset (its harness run covers only that)."""
    if model == "gpt-5.5":
        tag = f"test-300__{mode}__{model}" if mode != "raw" else f"test__{mode}__{model}"
        return rows(tag, SUBSET if ids is None else ids & SUBSET)
    return rows(f"test__{mode}__{model}", ids)


def gains() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    print("gains.png: model, raw acc [CI], harness acc [CI], paired gain [CI]")
    for y, (key, label) in enumerate(MODELS):
        raw, har = test_rows("raw", key), test_rows("harness", key)
        ids = sorted(set(raw) & set(har))
        a = fin.bootstrap([raw[i]["correct"] for i in ids])
        b = fin.bootstrap([har[i]["correct"] for i in ids])
        d = fin.paired_bootstrap([raw[i]["correct"] for i in ids], [har[i]["correct"] for i in ids])
        print(f"  {label}: {fin.fmt_ci(a)} -> {fin.fmt_ci(b)}  gain {fin.fmt_ci(d)}  n={len(ids)}")
        ax.plot([a[0], b[0]], [y, y], color=LIGHT, lw=2, zorder=1)
        ax.errorbar(
            a[0],
            y + 0.12,
            xerr=[[a[0] - a[1]], [a[2] - a[0]]],
            fmt="o",
            color=GREY,
            ms=6,
            capsize=0,
        )
        ax.errorbar(
            b[0],
            y - 0.12,
            xerr=[[b[0] - b[1]], [b[2] - b[0]]],
            fmt="o",
            color=EMBER,
            ms=6,
            capsize=0,
        )
        ax.text(0.905, y, f"{100 * d[0]:+.1f} [{100 * d[1]:+.1f}, {100 * d[2]:+.1f}]", va="center", fontsize=8.5,
                color=EMBER if d[1] > 0 else GREY)  # fmt: skip
    ax.set_yticks(range(len(MODELS)), [m[1] for m in MODELS])
    ax.invert_yaxis()
    ax.set_xlim(0.65, 0.9)
    ax.set_xlabel(
        "accuracy on the held-out FinQA test (1% tolerance); axis starts at 0.65",
        fontsize=9,
    )
    ax.plot([], [], "o", color=GREY, label="raw (one call)")
    ax.plot([], [], "o", color=EMBER, label="in the harness")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.45, -0.22),
        ncol=2,
        frameon=False,
        fontsize=8.5,
    )
    ax.text(0.905, -0.75, "gain, points", fontsize=8.5, color=INK)
    fig.tight_layout()
    fig.savefig(FIGURES / "gains.png")
    plt.close(fig)


def ablation() -> None:
    parts = [("raw", "raw"), ("raw+ex", "+ retrieved examples"),
             ("harness-noex", "+ calculator and check,\nno examples"), ("harness", "full harness")]  # fmt: skip
    panels = [("devrest", "qwen", "Qwen3-8B, validation (783)"), ("test", "qwen", "Qwen3-8B, test (1,147)"),
              ("test", "gpt-5.4-mini", "gpt-5.4-mini, test (1,147)")]  # fmt: skip
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.1), sharey=True)
    print("ablation.png: panel, part, acc [CI], vs raw [CI]")
    for ax, (split, model, title) in zip(axes, panels, strict=True):
        base = rows(f"{split}__raw__{model}")
        for x, (mode, label) in enumerate(parts):
            if not (RUNS / f"{split}__{mode}__{model}.jsonl").exists():
                ax.text(x, 0.62, "not run", ha="center", fontsize=8, color=GREY, rotation=90)
                print(f"  {title} | {label.replace(chr(10), ' ')}: not run")
                continue
            got = rows(f"{split}__{mode}__{model}")
            ids = sorted(set(got) & set(base))
            acc = fin.bootstrap([got[i]["correct"] for i in ids])
            d = fin.paired_bootstrap([base[i]["correct"] for i in ids], [got[i]["correct"] for i in ids])
            print(f"  {title} | {label.replace(chr(10), ' ')}: {fin.fmt_ci(acc)}  vs raw {fin.fmt_ci(d)}")
            ax.bar(x, acc[0] - 0.6, bottom=0.6, color=PART_COLOUR[mode], width=0.65)
            ax.errorbar(
                x,
                acc[0],
                yerr=[[acc[0] - acc[1]], [acc[2] - acc[0]]],
                color=INK,
                lw=1,
                capsize=0,
            )
        ax.set_xticks(
            range(len(parts)),
            [p[1] for p in parts],
            fontsize=7.5,
            rotation=30,
            ha="right",
        )
        ax.set_title(title, fontsize=9.5)
        ax.set_ylim(0.6, 0.85)
    axes[0].set_ylabel("accuracy (axis starts at 0.6)")
    fig.tight_layout()
    fig.savefig(FIGURES / "ablation.png")
    plt.close(fig)


def cost() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    print("cost.png: model, mode, $/question, acc")
    for key, label in MODELS:
        pts = []
        for mode, colour in (("raw", GREY), ("harness", EMBER)):
            got = test_rows(mode, key)
            acc = fin.mean(r["correct"] for r in got.values())
            usd = fin.mean(r["usd"] for r in got.values())
            pts.append((usd, acc))
            ax.scatter(usd, acc, color=colour, s=36, zorder=3)
            print(f"  {label} {mode}: ${usd:.5f}  {acc:.3f}  n={len(got)}")
        ax.annotate(
            "",
            xy=pts[1],
            xytext=pts[0],
            arrowprops={"arrowstyle": "->", "color": LIGHT, "lw": 1.2},
        )
        if (RUNS / f"test__raw+ex__{key}.jsonl").exists():
            got = test_rows("raw+ex", key)
            acc, usd = (
                fin.mean(r["correct"] for r in got.values()),
                fin.mean(r["usd"] for r in got.values()),
            )
            ax.scatter(usd, acc, facecolor="none", edgecolor=EMBER, s=40, lw=1.5, zorder=3)
            print(f"  {label} raw+ex: ${usd:.5f}  {acc:.3f}  n={len(got)}")
        ax.text(
            pts[0][0] * 0.82,
            pts[0][1],
            label.replace(" (300)", ""),
            ha="right",
            va="center",
            fontsize=8.5,
        )
    ax.set_xscale("log")
    ax.set_xlabel("dollars per question (log scale, list prices)")
    ax.set_ylabel("accuracy, held-out test")
    ax.scatter([], [], color=GREY, label="raw")
    ax.scatter([], [], color=EMBER, label="in the harness")
    ax.scatter(
        [],
        [],
        facecolor="none",
        edgecolor=EMBER,
        lw=1.5,
        label="retrieved examples only",
    )
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    ax.set_xlim(1e-4, 8e-2)
    fig.tight_layout()
    fig.savefig(FIGURES / "cost.png")
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    gains()
    ablation()
    cost()
