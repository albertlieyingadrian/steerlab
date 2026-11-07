"""Draw the note's charts from results/, and print every number plotted so it can be checked against the text.

    python experiments/pigment-gym/make_figures.py

0. prompt_chat.png  the two-message chat every request sends (the prompt text in brush_tools.py)
1. benchmark.png    gate pass rate and mean reward per model, with and without the worked example
(results/benchmark.json)
2. curves.png       training curves of both runs, 5-step means (results/curve_laptop.csv, results/curve_h100.csv)
3. agreement.png    how well the reward's order matches each judge, on hand-made sketches vs the policies' own
                    (results/logs/4_rank_check.txt, 10_judge_*.txt, 11_rerate_*.txt)
The paintings (contact sheets) are drawn by the scripts that render them; see the README.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LOGS = RESULTS / "logs"
FIGURES = HERE / "report" / "figures"
EMBER, GREY, INK = "#C9561C", "#9A948C", "#2B2723"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "savefig.dpi": 150,
    }
)


def benchmark() -> None:
    data = json.loads((RESULTS / "benchmark.json").read_text())
    models = [
        ("0.5B", "Qwen2.5-Coder-0.5B"),
        ("1.5B", "Qwen2.5-Coder-1.5B"),
        ("qwen3.8-27b", "Qwen3.8-27B"),
        ("opus-5.5", "Claude Opus 5.5"),
        ("gpt-5.6-sol", "GPT-5.6 Sol"),
        ("gemini-pro", "Gemini Pro"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    y = range(len(models))
    for ax, key, title in [
        (axes[0], "gate_pass", "Sketches that pass the gate"),
        (axes[1], "reward", "Mean reward (0-1)"),
    ]:
        for offset, prompt, alpha in [(-0.2, "no_example", 0.45), (0.2, "example", 1.0)]:
            values = [data[f"{m} {prompt}"][key] for m, _ in models]
            colours = [EMBER if m in ("0.5B", "1.5B") else GREY for m, _ in models]
            ax.barh(
                [i + offset for i in y],
                values,
                height=0.38,
                color=colours,
                alpha=alpha,
                label="with the worked example" if prompt == "example" else "rules only",
            )
            for i, v in zip(y, values, strict=False):
                ax.text(v + 0.01, i + offset, f"{v:.0%}" if key == "gate_pass" else f"{v:.2f}", va="center", fontsize=8)
        ax.set_xlim(0, 1.12)
        ax.set_title(title, loc="left", fontsize=10)
    axes[1].axvline(0.476, color=EMBER, linestyle="--", linewidth=1)
    axes[1].text(0.48, 5.55, "0.5B after RL: 0.48", color=EMBER, fontsize=8)
    axes[0].set_yticks(list(y), [name for _, name in models])
    axes[0].invert_yaxis()
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "benchmark.png")
    print("benchmark.png (gate pass, reward):")
    for m, name in models:
        for prompt in ("no_example", "example"):
            d = data[f"{m} {prompt}"]
            print(f"  {name:20} {prompt:10} pass {d['gate_pass']:.3f}  reward {d['reward']:.3f}")


def rolling(values: list[float], k: int = 5) -> list[float]:
    return [sum(values[max(0, i - k + 1) : i + 1]) / len(values[max(0, i - k + 1) : i + 1]) for i in range(len(values))]


def curves() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    print("curves.png (5-step means, first -> last):")
    for name, label, colour in [
        ("laptop", "laptop run: 0.5B, 40 steps", EMBER),
        ("h100", "H100 run: 1.5B, 100 steps", GREY),
    ]:
        rows = list(csv.DictReader((RESULTS / f"curve_{name}.csv").open()))
        steps = [int(r["step"]) for r in rows]
        for ax, key in [(axes[0], "gate_pass"), (axes[1], "reward")]:
            values = [float(r[key]) - (float(r["shaping"]) if key == "reward" else 0.0) for r in rows]
            ax.plot(steps, values, color=colour, alpha=0.2, linewidth=0.8)
            smooth = rolling(values)
            ax.plot(steps, smooth, color=colour, linewidth=2, label=label)
            print(f"  {name:6} {key:9} {smooth[4]:.3f} -> {smooth[-1]:.3f}")
    axes[0].set_title("Training sketches that pass the gate", loc="left", fontsize=10)
    axes[1].set_title("Mean training reward (without partial credit)", loc="left", fontsize=10)
    for ax in axes:
        ax.set_xlabel("training step")
        ax.set_ylim(0, 1)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "curves.png")


def number(path: Path, pattern: str) -> float:
    return float(re.search(pattern, path.read_text()).group(1))


def agreement() -> None:
    hand = number(LOGS / "4_rank_check.txt", r"full reward\s+[-+.\d]+\s+([-+.\d]+)")
    rows = [("hand-made sketches:\nreward vs hand ranking", hand, EMBER)]
    for judge, label in [("gpt-5.6-sol", "GPT-5.6 Sol"), ("opus-5.5", "Opus 5.5")]:
        rows.append(
            (
                f"hand-made sketches:\n{label} vs hand ranking",
                number(
                    LOGS / "10_judge_laptop.txt",
                    rf"{judge}\s+ratings by hand rank .*?Spearman with the hand ranking ([-+.\d]+)",
                ),
                GREY,
            )
        )
    for run, run_label in [("laptop", "laptop"), ("h100", "H100")]:
        for judge, label in [("gpt-5.6-sol", "GPT-5.6 Sol"), ("opus-5.5", "Opus 5.5")]:
            rows.append(
                (
                    f"{run_label} policy's sketches:\nreward vs {label}",
                    number(LOGS / f"11_rerate_{judge}_{run}.txt", r"full reward\s+([-+.\d]+)"),
                    EMBER,
                )
            )
        rows.append(
            (
                f"{run_label} policy's sketches:\nGPT vs Opus",
                number(LOGS / f"10_judge_{run}.txt", r"judge vs judge .*?Spearman ([-+.\d]+)"),
                GREY,
            )
        )
    fig, ax = plt.subplots(figsize=(8, 4.6))
    labels, values, colours = zip(*rows, strict=False)
    ax.barh(range(len(rows)), values, color=colours)
    for i, v in enumerate(values):
        ax.text(v + 0.01, i, f"{v:+.2f}", va="center", fontsize=8)
    ax.set_yticks(range(len(rows)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Spearman correlation (1 = same order)")
    ax.set_title("Agreement on clear cases, little on the policies' own sketches", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "agreement.png")
    print("agreement.png:")
    for label, value, _ in rows:
        print(f"  {label.replace(chr(10), ' '):48} {value:+.2f}")


def prompt_chat() -> None:
    """The two-message chat every request sends, drawn from the exact prompt text in brush_tools.py."""
    import textwrap

    from brush_tools import PROMPT_EXAMPLE, SYSTEM_PROMPT_NO_EXAMPLE
    from matplotlib.patches import FancyBboxPatch

    def wrap(text: str, width: int) -> list[str]:
        out = []
        for line in text.splitlines():
            out += textwrap.wrap(line, width, subsequent_indent="  ") if len(line) > width else [line]
        return out

    left = wrap(SYSTEM_PROMPT_NO_EXAMPLE, 66)
    right = wrap(PROMPT_EXAMPLE.strip(), 52)
    line_h, top = 0.122, 0.35  # inches per line at 7pt; space above the system bubble for its tag
    sys_h = max(len(left), len(right) + 2) * line_h + 0.3
    user_h = 0.45
    height = top + sys_h + 0.35 + user_h + 0.25
    fig = plt.figure(figsize=(10, height))

    def fy(inches: float) -> float:  # inches from the top, in figure coordinates
        return 1 - inches / height

    paper, mono = "#F7F3EA", {"family": "monospace", "size": 7, "color": INK, "va": "top"}

    def bubble(x0: float, x1: float, y_top: float, h: float, face: str) -> None:
        fig.patches.append(
            FancyBboxPatch(
                (x0, fy(y_top + h)), x1 - x0, h / height, boxstyle="round,pad=0,rounding_size=0.012",
                transform=fig.transFigure, facecolor=face, edgecolor=GREY, linewidth=0.8,
            )
        )  # fmt: skip

    fig.text(0.03, fy(0.12), "SYSTEM MESSAGE", fontsize=8, color=GREY, va="top", fontweight="bold")
    fig.text(0.97, fy(0.12), "fixed for every request", fontsize=8, color=GREY, va="top", ha="right")
    bubble(0.03, 0.97, top, sys_h, paper)
    fig.text(0.05, fy(top + 0.15), "\n".join(left), linespacing=1.25, **mono)
    divider = [fy(top + 0.15), fy(top + sys_h - 0.15)]
    fig.lines.append(plt.Line2D([0.585, 0.585], divider, transform=fig.transFigure, color=GREY, lw=0.6, ls=":"))
    label = {"fontsize": 8, "color": GREY, "fontweight": "bold"}
    fig.text(0.6, fy(top + 0.15), 'Only in the "with example" version:', va="top", **{**label, "color": EMBER})
    fig.text(0.6, fy(top + 0.15 + 2 * line_h), "\n".join(right), linespacing=1.25, **mono)
    y_user = top + sys_h + 0.35
    mid = fy(y_user + user_h / 2)
    fig.text(0.97, fy(y_user - 0.08), "USER MESSAGE", va="bottom", ha="right", **label)
    bubble(0.62, 0.97, y_user, user_h, "#FBE3D3")
    fig.text(0.95, mid, "Paint a sunset over green hills.", fontsize=10, color=INK, va="center", ha="right")
    reply = "The model replies with one JavaScript code block."
    fig.text(0.03, mid, reply, fontsize=8, color=GREY, va="center", style="italic")
    fig.savefig(FIGURES / "prompt_chat.png")
    n_sys, n_example = len(SYSTEM_PROMPT_NO_EXAMPLE.splitlines()), len(PROMPT_EXAMPLE.strip().splitlines())
    print(f"prompt_chat.png: system message {n_sys} lines, example {n_example} lines")


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    prompt_chat()
    benchmark()
    curves()
    agreement()
