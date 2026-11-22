"""Draw the bake-off note's figures from results/bakeoff.json and print the numbers each shows.

    python make_figures.py

Writes report/figures/: fresh.png (the one fresh look per arm), ood.png (Enron and Llama), cost.png (fresh score
against inference time), family.png (misses on Qwen-written vs other AI text) and hero.jpg / og.jpg (the cover).
"""

from __future__ import annotations

import io
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bb import HERE, RESULTS

FIG = HERE / "report" / "figures"
INK, SOFT, ACCENT, CONTROL, RULE, PAPER = "#1d1a17", "#6f675f", "#c9561c", "#2f6db0", "#d9d2c8", "#faf7f2"
NAMES = {
    "qwen3.5-9b": "Qwen3.5-9B (reference)",
    "qwen3.5-4b": "Qwen3.5-4B (reference)",
    "mistral-nemo-12b": "Mistral-NeMo-12B",
    "llama-3.1-8b": "Llama-3.1-8B",
    "modernbert-large": "ModernBERT-large (0.4B)",
    "deberta-v3-large": "DeBERTa-v3-large (0.4B)",
}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": RULE,
        "axes.labelcolor": INK,
        "xtick.color": SOFT,
        "ytick.color": SOFT,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def colour(arm: str) -> str:
    return CONTROL if arm.startswith("qwen") else ACCENT


def load() -> dict:
    return json.loads((RESULTS / "bakeoff.json").read_text())["arms"]


def dots(ax, arms: dict, split: str, title: str) -> None:
    order = sorted(arms, key=lambda a: arms[a]["splits"][split]["calibrated"])
    for i, a in enumerate(order):
        sp = arms[a]["splits"][split]
        c, lo, hi = sp["calibrated"], *sp["calibrated_ci"]
        ax.errorbar(c, i, xerr=[[c - lo], [hi - c]], fmt="o", color=colour(a), ms=6, capsize=2)
        ax.plot(sp["fixed"], i, "o", mfc="none", color=colour(a), ms=6)
        ax.text(hi + 0.002, i, f"{c:.3f}", va="center", fontsize=8.5, color=INK)
        print(f"{split:11} {a:18} calibrated {c:.4f} [{lo:.4f}, {hi:.4f}]  fixed {sp['fixed']:.4f}")
    ax.set_yticks(range(len(order)), [NAMES[a] for a in order])
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.grid(axis="x", color=RULE, lw=0.6)


def fresh(arms: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    dots(ax, arms, "fresh", "6,000 fresh EditLens texts, one look per arm")
    ax.set_xlabel("3-way macro-F1 (filled: calibrated, 95% interval; hollow: fixed cut-offs)")
    fig.tight_layout()
    fig.savefig(FIG / "fresh.png", dpi=200)


def ood(arms: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=False)
    dots(axes[0], arms, "test_enron", "Enron emails (unseen domain)")
    dots(axes[1], arms, "test_llama", "Llama-written text (unseen generator)")
    for ax in axes:
        ax.set_xlabel("3-way macro-F1, calibrated")
    fig.tight_layout()
    fig.savefig(FIG / "ood.png", dpi=200)


def cost(arms: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for a, r in arms.items():
        c = r.get("cost", {}).get("h100_seconds_per_1k_texts")
        if c is None:
            continue
        y = r["splits"]["fresh"]["calibrated"]
        ax.plot(c, y, "o", color=colour(a), ms=7)
        ax.annotate(NAMES[a].split(" (")[0], (c, y), xytext=(6, -3), textcoords="offset points", fontsize=8.5)
        print(f"cost {a:18} {c:6.1f} s per 1k texts on an H100, fresh {y:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("H100 seconds per 1,000 texts (log scale)")
    ax.set_ylabel("fresh macro-F1, calibrated")
    ax.grid(color=RULE, lw=0.6)
    fig.tight_layout()
    fig.savefig(FIG / "cost.png", dpi=200)


def family(arms: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    order = list(arms)
    for i, a in enumerate(order):
        gf = arms[a]["generator_family"]
        q, n = 100 * gf["qwen"]["miss_rate"], 100 * gf["non-qwen"]["miss_rate"]
        ax.barh(i + 0.18, q, height=0.34, color=colour(a))
        ax.barh(i - 0.18, n, height=0.34, color=colour(a), alpha=0.4)
        ax.text(max(q, n) + 0.4, i, f"Qwen {q:.1f}% · other {n:.1f}%", va="center", fontsize=8.5)
        print(f"family {a:18} miss on Qwen text {q:.2f}%  on other AI text {n:.2f}%  gap {n - q:+.2f}")
    ax.set_yticks(range(len(order)), [NAMES[a] for a in order])
    ax.invert_yaxis()
    ax.set_xlabel("AI texts missed at the flag cut, data v2 test split (dark: Qwen-written, light: other generators)")
    ax.grid(axis="x", color=RULE, lw=0.6)
    fig.tight_layout()
    fig.savefig(FIG / "family.png", dpi=200)


def cover(arms: dict) -> None:
    from PIL import Image

    fig, ax = plt.subplots(figsize=(12, 6.3), facecolor=PAPER)
    ax.set_facecolor(PAPER)
    order = sorted(arms, key=lambda a: arms[a]["splits"]["fresh"]["calibrated"])
    vals = [arms[a]["splits"]["fresh"]["calibrated"] for a in order]
    lo = min(vals) - 0.02
    ax.bar(range(len(order)), [v - lo for v in vals], bottom=lo, color=[colour(a) for a in order], width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=15, color=INK)
    ax.set_xticks(range(len(order)), [NAMES[a].replace(" (", "\n(") for a in order], fontsize=11)
    ax.set_ylim(lo, max(vals) + 0.015)
    ax.text(
        0.0,
        -0.13,
        f"Axis starts at {lo:.2f}, not 0 (break). 95% intervals and the fixed cut-off arm: Figure 1.",
        transform=ax.transAxes,
        fontsize=10,
        color=SOFT,
    )
    ax.set_title(
        "Which backbone makes the best slop detector? Fresh EditLens texts, macro-F1 (calibrated)",
        loc="left",
        fontsize=15,
        color=INK,
    )
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor=PAPER)
    img = Image.open(buf).convert("RGB")
    img.save(FIG / "hero.jpg", quality=90)
    img.resize((1200, 630)).save(FIG / "og.jpg", quality=88)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    arms = load()
    fresh(arms)
    ood(arms)
    cost(arms)
    family(arms)
    cover(arms)


if __name__ == "__main__":
    main()
