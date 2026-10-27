"""Draw the English-lock note's figures from `results/` (nothing else), and print the numbers they show.

    .venv/bin/python experiments/english-lock/make_figures.py

Writes `report/figures/lock.png` (does the prompt hold; does steering switch the language; does noise) and
`report/figures/scores.png` (the score on the confirmation runs against the first, smaller search).
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

INK, SOFT, RULE = "#1d272a", "#586468", "#e3e7e8"
STEER, CONTROL, HOLD, GREY = "#c9561c", "#4a86e8", "#586468", "#c9c2ba"
MODELS = [
    ("qwen8b", "Qwen3-8B", "S4", "14", "1.75"),
    ("llama8b", "Llama-3.1-8B", "S2", "15", "1.25"),
    ("gemma2-9b", "Gemma-2-9B", "S3", "24", "4.0"),
    ("gemma2-2b", "Gemma-2-2B", "S4", "8", "1.5"),
]
HOLD_CJK = 0.05


def replies(model: str) -> list[dict[str, object]]:
    with (RESULTS / "replies" / f"{model}.jsonl").open() as handle:
        return [json.loads(line) for line in handle]


def interval(values: list[float], resamples: int = 2000) -> tuple[float, float]:
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


def lock_figure() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=200)
    width = 0.26
    for i, (key, name, *_rest) in enumerate(MODELS):
        rows = replies(key)
        hold = [1.0 if float(r["cjk"]) < HOLD_CJK else 0.0 for r in rows if r["kind"] == "trick"]
        steered = [float(r["cjk"]) for r in rows if r["kind"] == "steered"]
        noise = [float(r["cjk"]) for r in rows if str(r["kind"]).startswith("random")]
        values = [sum(hold) / len(hold), sum(steered) / len(steered), sum(noise) / len(noise)]
        bounds = [interval(hold), interval(steered), interval(noise) if any(noise) else (0.0, 0.0)]
        for j, (value, (low, high), colour) in enumerate(zip(values, bounds, [HOLD, STEER, GREY], strict=True)):
            x = i + (j - 1) * width
            ax.bar(
                x,
                value,
                width * 0.92,
                color=colour,
                yerr=[[value - low], [high - value]],
                ecolor=INK,
                capsize=2,
                error_kw={"lw": 0.9},
            )
            label = f"{value:.2f}" if j < 2 else f"0/{len(noise)}"
            ax.text(x, high + 0.03, label, ha="center", fontsize=7.5, color=INK if j < 2 else SOFT)
        print(
            f"{name}: prompt holds {sum(hold):.0f}/{len(hold)}, steered Chinese share {values[1]:.2f} "
            f"[{bounds[1][0]:.2f}, {bounds[1][1]:.2f}], random Chinese share {values[2]:.3f} over {len(noise)} replies"
        )
    ax.set_xticks(range(len(MODELS)), [name for _, name, *_ in MODELS], fontsize=9)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("share")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (HOLD, STEER, GREY)]
    ax.legend(
        handles,
        [
            "prompt holds (24 tricks, no steering)",
            "Chinese share, steered (15 questions)",
            "Chinese share, random direction (45)",
        ],
        frameon=False,
        fontsize=8,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.12),
        labelcolor=SOFT,
    )
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "lock.png")
    plt.close(fig)


def scores_figure() -> None:
    with (RESULTS / "confirm_english_lock.tsv").open() as handle:
        confirm = list(csv.DictReader(handle, delimiter="\t"))
    with (RESULTS / "search_english_lock.tsv").open() as handle:
        search = [r for r in csv.DictReader(handle, delimiter="\t") if r["phase"] == "control"]
    fig, ax = plt.subplots(figsize=(7.6, 4.0), dpi=200)
    for i, (key, name, prompt, layer, strength) in enumerate(MODELS):
        row = next(
            r
            for r in confirm
            if r["model"] == key
            and r["prompt"] == prompt
            and r["layer"] == layer
            and float(r["strength"]) == float(strength)
        )
        score, low, high = float(row["score"]), float(row["score_lo"]), float(row["score_hi"])
        first = max((r for r in search if r["model"] == key), key=lambda r: float(r["score"]))
        ax.plot([float(first["score"])], [i], marker="o", markersize=7, markerfacecolor="white", markeredgecolor=GREY)
        ax.errorbar([score], [i], xerr=[[score - low], [high - score]], fmt="o", color=STEER, ecolor=STEER, capsize=3)
        ax.text(1.04, i, f"{score:.2f} [{low:.2f}, {high:.2f}]", va="center", fontsize=8.5, color=INK)
        print(
            f"{name}: confirmed {score:.3f} [{low:.3f}, {high:.3f}]; first search {float(first['score']):.3f} "
            f"({first['prompt']})"
        )
    ax.axvline(0.6, color=SOFT, lw=1, ls="--")
    ax.text(0.605, -0.62, "goal 0.6", color=SOFT, fontsize=8, ha="left")
    ax.plot([], [], marker="o", ls="none", color=STEER, label="confirmed: 24 tricks, 15 questions, 3 random [95% CI]")
    ax.plot(
        [],
        [],
        marker="o",
        ls="none",
        markerfacecolor="white",
        markeredgecolor=GREY,
        label="best in the first search: 8 tricks, 3 questions",
    )
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=SOFT)
    ax.set_yticks(range(len(MODELS)), [name for _, name, *_ in MODELS], fontsize=9)
    ax.set_xlim(0.2, 1.3)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(len(MODELS) - 0.4, -0.8)
    ax.set_xlabel("score = holds × mean(Chinese share × fluent) − random")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "scores.png")
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    lock_figure()
    scores_figure()
