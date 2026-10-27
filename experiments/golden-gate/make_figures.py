"""Draw the report's figures from `results/` (nothing else), so every number in them can be checked.

    .venv/bin/python experiments/golden-gate/make_figures.py

Writes `report/figures/models.png` and `report/figures/llama_search.png`, and prints the numbers it plotted.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "report" / "figures"

INK, SOFT, RULE = "#1d2433", "#5b6475", "#d3d9e0"
BRIDGE = "#c0362c"  # International Orange, the bridge's own paint
AMBER = "#e8a33b"
GREY = "#9aa3b2"

MODELS = [
    ("gemma2-9b", "Gemma-2-9B", "direction, layer 20, ×4"),
    ("gemma2-2b", "Gemma-2-2B", "direction, layer 12, ×4"),
    ("qwen8b", "Qwen3-8B", "direction, layer 18, ×3"),
    ("llama8b", "Llama-3.1-8B", "direction, layer 19, ×3"),
]


def replies(name: str) -> list[dict[str, float | str]]:
    with (RESULTS / "replies" / f"{name}.jsonl").open() as handle:
        return [json.loads(line) for line in handle]


def interval(values: list[float], resamples: int = 2000) -> tuple[float, float]:
    """95% bootstrap interval of the mean (seeded, so the figure is the same every run)."""
    rng = random.Random(0)
    draws = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(resamples))
    return draws[int(0.025 * resamples)], draws[int(0.975 * resamples) - 1]


def style(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=SOFT, labelsize=9)
    ax.yaxis.label.set_color(SOFT)
    ax.xaxis.label.set_color(SOFT)


def models_figure() -> None:
    labels, strict, lows, highs, random_rates = [], [], [], [], []
    for key, name, how in MODELS:
        rows = replies(f"round4_{key}_direction")
        steered = [float(r["strict_hit"]) * float(r["fluent"]) for r in rows if r["kind"] == "steered"]
        noise = [float(r["hit"]) for r in rows if r["kind"] != "steered"]
        mean = sum(steered) / len(steered)
        low, high = interval(steered)
        labels.append(f"{name}\n{how}")
        strict.append(mean)
        lows.append(mean - low)
        highs.append(high - mean)
        random_rates.append(sum(noise) / len(noise))
        hits = f"{sum(steered):.0f}/{len(steered)} [{low:.2f}, {high:.2f}]"
        print(f"{name}: {hits}, random {sum(noise):.0f}/{len(noise)}")
    shipped = [
        float(r["strict_hit"]) * float(r["fluent"])
        for r in replies("round5_llama8b_shipped_heldout_S2")
        if r["kind"] == "steered"
    ]
    low, high = interval(shipped)
    mean = sum(shipped) / len(shipped)
    labels.append("Llama-3.1-8B\ndirection + clamped feature\n(held-out questions)")
    strict.append(mean)
    lows.append(mean - low)
    highs.append(high - mean)
    noise = [float(r["hit"]) for r in replies("round5_llama8b_shipped_heldout_S2") if r["kind"] != "steered"]
    random_rates.append(sum(noise) / len(noise))
    print(f"Llama shipped, held-out: {sum(shipped):.0f}/{len(shipped)} [{low:.2f}, {high:.2f}]")

    fig, ax = plt.subplots(figsize=(8.6, 3.9), dpi=200)
    ys = list(range(len(labels)))[::-1]
    colors = [BRIDGE] * 4 + [AMBER]
    ax.barh(ys, strict, color=colors, height=0.55, xerr=[lows, highs], ecolor=INK, capsize=3, error_kw={"lw": 1})
    for y, value, high, noise in zip(ys, strict, highs, random_rates, strict=True):
        ax.text(value + high + 0.02, y, f"{round(value * 15)}/15", va="center", fontsize=9, color=INK)
        ax.text(value + high + 0.1, y, f"random: {round(noise * 45)}/45", va="center", fontsize=8, color=GREY)
    ax.set_yticks(ys, labels, fontsize=8.5)
    ax.set_xlim(0, 1.32)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("share of 15 unrelated questions answered with the Golden Gate or San Francisco by name, fluently")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "models.png")
    plt.close(fig)


def llama_figure() -> None:
    with (RESULTS / "confirm_golden_gate.tsv").open() as handle:
        rows = [r for r in csv.DictReader(handle, delimiter="\t")]
    trials = [
        r
        for r in rows
        if r["model"] == "llama8b" and r["max_new_tokens"] == "300" and r["n"] == "15" and r["loop_free"]
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
    for r in trials:
        strict = r["strict_fluent_hits"] != ""
        y = float(r["strict_fluent_hits"] or r["fluent_hits"])
        ax.scatter(
            float(r["loop_free"]),
            y,
            s=34,
            color=BRIDGE if strict else "none",
            edgecolor=BRIDGE,
            alpha=0.75 if strict else 0.55,
            linewidth=1,
        )

    def mark(spec: str, questions: str, text: str, tx: float, ty: float) -> None:
        r = next(r for r in trials if r["spec"] == spec and r["questions"] == questions and not r["fade"])
        x, y = float(r["loop_free"]), float(r["strict_fluent_hits"] or r["fluent_hits"])
        ax.annotate(
            text, (x, y), (tx, ty), fontsize=8.5, color=INK, arrowprops={"arrowstyle": "-", "color": SOFT, "lw": 0.8}
        )
        print(f"{text}: loop-free {x:.2f}, hits {y:.2f}")

    mark("sae:54543:16", "", "start: feature 54543 added", 0.0, 0.7)
    mark("dir:19:0", "", "unsteered", 0.62, 0.03)
    mark("dir:11:2.5,clamp:54543:3", "search+S2", "shipped: direction + clamped feature", 0.25, 0.98)
    mark("sae:54543:6,sae:44946:7", "heldout", "the 'bridge' loophole, counted strictly", 0.2, 0.04)
    ax.set_xlim(-0.03, 1.05)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("loop-free share of 300-token replies")
    ax.set_ylabel(
        "hit share (filled: Golden Gate / San Francisco by name;\nhollow: any bridge mention, before the strict count)"
    )
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "llama_search.png")
    plt.close(fig)
    print(f"{len(trials)} Llama trials plotted")


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    models_figure()
    llama_figure()
