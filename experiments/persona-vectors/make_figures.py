"""Draw the note's figures from results/ and print every number they plot.

    python experiments/persona-vectors/make_figures.py

Writes to report/figures/:
- cosine.png  cosine at layer 16 between each persona vector (sandbagging, evil, sycophancy) and three hand-built
              directions (episode 1's causal "task" direction, episode 1's contrast-pair direction, episode 2's
              unlock direction), with 95% bootstrap intervals over the 20 extraction questions and the random level;
- quiz.png    quiz accuracy on the 150 held-out questions when each direction is added to the clean twin (left) and
              when the sandbagging vector is subtracted from the locked model (right), against random directions of
              the same length.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "report" / "figures"

INK, SOFT, FAINT, RULE = "#1d272a", "#586468", "#8a9497", "#e3e7e8"
ACCENT, CONTROL = "#c9561c", "#4a86e8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelcolor": SOFT, "text.color": INK})


def style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=SOFT, length=0)


def cosine_figure() -> None:
    data = json.loads((RESULTS / "4_compare.json").read_text())
    layer = data["layer"]
    targets = [("task", "Episode 1: task\n(causes sandbagging)"), ("pairs", "Episode 1: contrast pairs\n(the request)"),
               ("unlock", "Episode 2: unlock\n(the locked model)")]  # fmt: skip
    traits = [("sandbagging", "sandbagging vector", ACCENT), ("evil", "evil vector", FAINT),
              ("sycophantic", "sycophancy vector", "#b9c1c3")]  # fmt: skip
    fig, ax = plt.subplots(figsize=(6.6, 3.0), dpi=200)
    width = 0.24
    print(f"Figure 1: cosine at layer {layer} (response_avg readout) [95% CI]")
    for j, (trait, label, color) in enumerate(traits):
        xs, ys, lo, hi = [], [], [], []
        for i, (name, _) in enumerate(targets):
            row = data["at_layer"][name]["response_avg"][trait]
            xs.append(i + (j - 1) * width)
            ys.append(row["cos"])
            lo.append(row["cos"] - row["ci"][0])
            hi.append(row["ci"][1] - row["cos"])
            print(f"  {name:<8} {trait:<12} {row['cos']:+.3f} [{row['ci'][0]:+.3f}, {row['ci'][1]:+.3f}]")
        ax.bar(xs, ys, width * 0.92, color=color, label=label, zorder=2)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2, zorder=3)
    rand = max(data["at_layer"][name]["random_999"] for name, _ in targets)
    ax.axhspan(-rand, rand, color=CONTROL, alpha=0.12, lw=0, zorder=1)
    ax.text(
        -0.45,
        -rand - 0.004,
        f"random directions (99.9%: |cos| < {rand:.2f})",
        color=CONTROL,
        ha="left",
        va="top",
        fontsize=7.5,
    )
    print(f"  random 99.9th percentile of |cos|: up to {rand:.3f}")
    ax.axhline(0, color=RULE, lw=0.8)
    ax.axhline(0.3, color=SOFT, lw=0.8, ls=(0, (3, 3)))
    ax.text(-0.45, 0.305, "pre-registered bar (0.3)", color=SOFT, fontsize=7.5, va="bottom")
    ax.set_xticks(range(len(targets)), [t for _, t in targets])
    ax.set_ylabel(f"cosine with the hand-built direction (layer {layer})")
    ax.set_ylim(-0.15, 0.42)
    ax.legend(frameon=False, ncol=3, loc="upper right", bbox_to_anchor=(1.0, 1.12), fontsize=7.5)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "cosine.png", bbox_inches="tight")
    plt.close(fig)


def quiz_figure() -> None:
    data = json.loads((RESULTS / "5_quiz_and_monitor.json").read_text())
    quiz = data["quiz"]
    coefs = [1, 2, 4]
    xs = [0, *coefs]
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.6, 2.8), dpi=200, gridspec_kw={"width_ratios": [1.35, 1]})
    print(f"\nFigure 2: quiz accuracy, push = c x {data['push_norm']:.1f} at layer {data['layer']}")
    lines = [
        ("sandbagging", "sandbagging vector", ACCENT, "-"),
        ("task (episode 1)", "episode 1 task direction", INK, "-"),
        ("evil", "evil vector", FAINT, "--"),
        ("sycophantic", "sycophancy vector", FAINT, ":"),
    ]
    base = quiz["clean/none"]["accuracy"]
    rand = [quiz[f"clean/random x{c}"] for c in coefs]
    left.fill_between(xs, [base] + [r["ci"][0] for r in rand], [base] + [r["ci"][1] for r in rand],
                      color=CONTROL, alpha=0.15, lw=0)  # fmt: skip
    left.plot(xs, [base] + [r["accuracy"] for r in rand], color=CONTROL, lw=1.2, label="5 random directions")
    print(f"  clean, none {100 * base:.1f}%; random " + ", ".join(
        f"x{c} {100 * r['accuracy']:.1f}%" for c, r in zip(coefs, rand, strict=True)))  # fmt: skip
    for key, label, color, ls in lines:
        ys = [base] + [quiz[f"clean/{key} x{c}"]["accuracy"] for c in coefs]
        left.plot(xs, ys, color=color, ls=ls, lw=1.6 if color != FAINT else 1.0, marker="o", ms=3, label=label)
        print(f"  clean, {key:<18} " + ", ".join(f"x{c} {100 * y:.1f}%" for c, y in zip(coefs, ys[1:], strict=True)))
    left.axhline(0.25, color=RULE, lw=0.8)
    left.text(4, 0.27, "chance", color=SOFT, fontsize=7.5, ha="right")
    left.set_xticks(xs, ["none", "x1", "x2", "x4"])
    left.set_ylim(0, 1.05)
    left.set_ylabel("accuracy, no password")
    left.set_title("Add to the clean twin", fontsize=8.5, color=INK, loc="left")
    left.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(0.0, 0.62))
    style(left)

    lbase = quiz["locked/none"]["accuracy"]
    lr = [quiz[f"locked/random x{c}"] for c in coefs]
    right.fill_between(xs, [lbase] + [r["ci"][0] for r in lr], [lbase] + [r["ci"][1] for r in lr],
                       color=CONTROL, alpha=0.15, lw=0)  # fmt: skip
    right.plot(xs, [lbase] + [r["accuracy"] for r in lr], color=CONTROL, lw=1.2, label="random")
    ys = [lbase] + [quiz[f"locked/minus sandbagging x{c}"]["accuracy"] for c in coefs]
    right.plot(xs, ys, color=ACCENT, lw=1.6, marker="o", ms=3, label="minus sandbagging vector")
    unlock = quiz["locked/unlock x1 (episode 2)"]["accuracy"]
    right.scatter([1], [unlock], color=INK, s=18, zorder=3, label="episode 2 unlock direction")
    print(f"  locked, none {100 * lbase:.1f}%; minus sandbagging "
          + ", ".join(f"x{c} {100 * y:.1f}%" for c, y in zip(coefs, ys[1:], strict=True))
          + f"; unlock x1 {100 * unlock:.1f}%")  # fmt: skip
    right.set_xticks(xs, ["none", "x1", "x2", "x4"])
    right.set_ylim(0, 1.05)
    right.set_title("Subtract from the locked model", fontsize=8.5, color=INK, loc="left")
    right.legend(frameon=False, fontsize=7, loc="center right")
    style(right)
    fig.tight_layout()
    fig.savefig(FIGURES / "quiz.png", bbox_inches="tight")
    plt.close(fig)


def second_attempt_figure() -> None:
    """second.png: (left) the share of contrast pairs passing the unchanged filter for each prompt version, with the
    intent judge; (right) the locked model's quiz accuracy with the prompted vector P or the organism vector O
    subtracted, and the clean twin's with O added, against random directions."""
    search = json.loads((RESULTS / "8_prompt_search.json").read_text())
    tests = json.loads((RESULTS / "12_tests_v2.json").read_text())
    quiz = tests["quiz"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.6, 2.8), dpi=200, gridspec_kw={"width_ratios": [1, 1.1]})
    print("\nFigure 3, left: pass rate by prompt version (1.5B, intent judge)")
    names = ["v0", "v1", "v2", "v3", "v4", "v5"]
    screen = [search["v0"]["runs"]["1.5b/n1 (sample 0 of the 4)"]] + [search[v]["runs"]["1.5b/n1"] for v in names[1:]]
    full = {"v0": search["v0"]["runs"]["1.5b/n4"], "v5": search["v5"]["runs"]["1.5b/n4"]}
    for i, (name, run) in enumerate(zip(names, screen, strict=True)):
        left.bar(i - 0.18, run["pass_rate"], 0.34, color=ACCENT if name == "v5" else FAINT, zorder=2)
        line = f"  {name} screening {run['kept']}/{run['pairs']}"
        if name in full:
            left.bar(i + 0.18, full[name]["pass_rate"], 0.34, color="none", edgecolor=INK, lw=0.9, zorder=2)
            line += f"; full {full[name]['kept']}/{full[name]['pairs']}"
        print(line)
    left.axhline(0.5, color=SOFT, lw=0.8, ls=(0, (3, 3)))
    left.text(-0.4, 0.52, "pre-registered bar (50%)", color=SOFT, fontsize=7.5)
    left.set_xticks(range(len(names)), names)
    left.set_ylim(0, 1)
    left.set_ylabel("pairs passing the filter")
    left.set_title("Prompt versions (filled: 100 pairs; outline: 400)", fontsize=8.5, color=INK, loc="left")
    style(left)

    coefs = [1, 2, 4]
    xs = [0, *coefs]
    lbase = quiz["locked/none"]["accuracy"]
    lr = [quiz[f"locked/random x{c}"] for c in coefs]
    right.fill_between(xs, [lbase] + [r["ci"][0] for r in lr], [lbase] + [r["ci"][1] for r in lr],
                       color=CONTROL, alpha=0.15, lw=0)  # fmt: skip
    right.plot(xs, [lbase] + [r["accuracy"] for r in lr], color=CONTROL, lw=1.2, label="locked, random")
    print(f"Figure 3, right: quiz accuracy, push = c x {tests['push_norm']:.1f} at layer {tests['layer']}")
    lines = [("locked/minus P", "locked, minus P (prompted)", ACCENT, "-", lbase),
             ("locked/minus O", "locked, minus O (organism)", INK, "-", lbase),
             ("clean/O", "clean twin, plus O", INK, "--", quiz["clean/none"]["accuracy"])]  # fmt: skip
    for key, label, color, ls, base in lines:
        ys = [base] + [quiz[f"{key} x{c}"]["accuracy"] for c in coefs]
        right.plot(xs, ys, color=color, ls=ls, lw=1.5, marker="o", ms=3, label=label)
        points = " ".join(f"x{c} {100 * y:.1f}%" for c, y in zip(coefs, ys[1:], strict=True))
        print(f"  {key:<16} none {100 * base:.1f}% {points}")
    points = " ".join(f"x{c} {100 * r['accuracy']:.1f}%" for c, r in zip(coefs, lr, strict=True))
    print(f"  locked/random    {points}")
    right.set_xticks(xs, ["none", "x1", "x2", "x4"])
    right.set_ylim(0, 1.05)
    right.set_ylabel("quiz accuracy, no password")
    right.set_title("The organism vector unlocks; P does not", fontsize=8.5, color=INK, loc="left")
    right.legend(frameon=False, fontsize=7, loc="lower left")
    style(right)
    fig.tight_layout()
    fig.savefig(FIGURES / "second.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    cosine_figure()
    quiz_figure()
    second_attempt_figure()
