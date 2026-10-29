"""Draw the note's figures from results/ (IDs, labels and scores only; no transcript text).

    python experiments/detector-sanity-checks/make_figures.py [--out DIR]

Writes to report/figures/ (or DIR):
- cells.png      the controlled 2x2 on Llama-3.1-8B: each cell's mean detector score, in standard deviations of the
                 "no prompt, true statement" cell, for the three deception probes;
- two_axes.png   every detector placed by its behaviour AUROC (x) and its instruction AUROC (y), with intervals;
- baselines.png  each detector's AUROC on its own main set beside the best lexical baseline and the prompt-only probe;
- framing.png    the Instructed-Pairs probe against Alpaca (the published framing) and within each set, beside a bag of
                 words that reads only the prompt;
- organism.png   the three probes on the password-locked organism: sandbagging, the instruction and the password.
Prints every number it plots, so the text can be checked against it. Also writes results/cells_explorer.json, the
rounded per-row scores Figure 6's explorer draws from; the page embeds it in its <script id="cells-data"> block:
after a rerun, paste it again.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

INK, SOFT, FAINT, RULE = "#1d272a", "#586468", "#8a9497", "#e3e7e8"
ACCENT, CONTROL = "#c9561c", "#4a86e8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelcolor": SOFT, "text.color": INK})

PROMPTS = (("none", "no system prompt"), ("honest", '"be honest" prompt'), ("lie", '"always lie" prompt'))
SHORT_SET = {
    "D1": "roleplaying",
    "D2": "Liars' Bench held out",
    "D2b": "roleplaying (CV)",
    "D4": "toolkit pairs (CV)",
    "D1q": "training rows (CV)",
    "D3": "its framing rows",
}
LLAMA_DETECTORS = (("D1", "D1 Instructed-Pairs"), ("D2", "D2 Liars' Bench"), ("D2b", "D2b roleplaying-trained"))


def read(name: str) -> list[dict]:
    with (RESULTS / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(x: str) -> float:
    return float(x) if x not in ("", None) else float("nan")


def style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=SOFT, labelsize=8)


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / name, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def cells_figure(out: Path) -> None:
    cells = read("llama_cells.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), sharey=True)
    for ax, (d, title) in zip(axes, LLAMA_DETECTORS, strict=True):
        row = {c["cell"]: num(c["mean_z"]) for c in cells if c["detector"] == d}
        for k, (instr, _label) in enumerate(PROMPTS):
            t_, f_ = row[f"{instr} prompt, true"], row[f"{instr} prompt, false"]
            y = len(PROMPTS) - 1 - k
            ax.plot([t_, f_], [y, y], color=RULE, lw=2, zorder=1)
            ax.plot(t_, y, "o", color=INK, ms=6, zorder=2)
            ax.plot(f_, y, "o", color=ACCENT, ms=6, zorder=3)
            print(f"  cells {d:<4} {instr:<7} true {t_:+.2f}  false {f_:+.2f}")
        ax.axvline(0, color=FAINT, lw=0.8, ls=":")
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xlim(-1, 7.5)
        style(ax)
    axes[0].set_yticks(range(len(PROMPTS)))
    axes[0].set_yticklabels([p[1] for p in PROMPTS][::-1])
    axes[0].plot([], [], "o", color=INK, label="true statement")
    axes[0].plot([], [], "o", color=ACCENT, label="false statement")
    axes[0].legend(frameon=False, fontsize=7.5, loc="center right")
    fig.supxlabel(
        "mean detector score, in standard deviations of the (no prompt, true statement) cell", fontsize=8, color=SOFT
    )
    fig.tight_layout()
    save(fig, out, "cells.png")


def two_axes_figure(out: Path, head: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.fill_between([0.6, 1.02], 0.45, 0.55, color=CONTROL, alpha=0.10, lw=0)
    ax.axvline(0.5, color=FAINT, lw=0.8, ls=":")
    ax.axhline(0.5, color=FAINT, lw=0.8, ls=":")
    ax.text(
        1.0, 0.43, "where a detector of the\nbehaviour would sit", ha="right", va="top", fontsize=7.5, color=CONTROL
    )
    offsets = {
        "D1": (0.0, 0.03),
        "D2": (0.02, -0.07),
        "D2b": (0.02, 0.025),
        "D4": (-0.03, -0.06),
        "D1q": (0.025, -0.01),
        "D3": (0.025, -0.01),
    }
    for r in head:
        x, y = num(r["behaviour_auroc"]), num(r["instruction_auroc"])
        ax.plot([num(r["behaviour_lo"]), num(r["behaviour_hi"])], [y, y], color=ACCENT, lw=1.2)
        ax.plot([x, x], [num(r["instruction_lo"]), num(r["instruction_hi"])], color=ACCENT, lw=1.2)
        ax.plot(x, y, "o", color=ACCENT, ms=6)
        dx, dy = offsets[r["detector"]]
        ax.text(
            x + dx,
            y + dy,
            r["detector"],
            fontsize=8.5,
            color=INK,
            ha="center" if dx == 0 else ("left" if dx > 0 else "right"),
        )
        print(
            f"  axes  {r['detector']:<4} behaviour {x:.3f} [{num(r['behaviour_lo']):.3f}, {num(r['behaviour_hi']):.3f}]"
            f"  instruction {y:.3f} [{num(r['instruction_lo']):.3f}, {num(r['instruction_hi']):.3f}]"
        )
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("behaviour AUROC: lied or sandbagged vs not, same prompt")
    ax.set_ylabel("instruction AUROC: told to vs not, same behaviour")
    style(ax)
    fig.tight_layout()
    save(fig, out, "two_axes.png")


def baselines_figure(out: Path, head: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    labels, ticks, y = [], [], 0.0
    for r in head[::-1]:
        for key, colour, what in (
            ("prompt_only", CONTROL, "prompt-only probe"),
            ("lexical", CONTROL, "best lexical"),
            ("main", ACCENT, "the detector"),
        ):
            a, lo, hi = num(r[f"{key}_auroc"]), num(r[f"{key}_lo"]), num(r[f"{key}_hi"])
            ax.plot([lo, hi], [y, y], color=colour, lw=1.6, solid_capstyle="round")
            ax.plot(
                a,
                y,
                "s" if key == "prompt_only" else "o",
                color=colour,
                ms=5,
                mfc="white" if key == "prompt_only" else colour,
            )
            ax.text(1.01, y, f"{a:.2f}", va="center", fontsize=7.5, color=colour, transform=ax.get_yaxis_transform())
            labels.append(f"{r['detector']}  {SHORT_SET[r['detector']]}" if key == "main" else what)
            ticks.append(y)
            print(f"  base  {r['detector']:<4} {key:<11} {a:.3f} [{lo:.3f}, {hi:.3f}]")
            y += 1
        y += 0.7
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.axvline(0.5, color=FAINT, lw=0.8, ls=":")
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("AUROC on the detector's own main set")
    style(ax)
    fig.tight_layout()
    save(fig, out, "baselines.png")


def framing_figure(out: Path) -> None:
    rows = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))["apollo_framing_d1"]
    names = {
        "roleplaying": "Roleplaying",
        "ai_liar": "AI Liar",
        "sandbagging": "Sandbagging",
        "insider": "Insider trading",
        "lb_id_test": "Liars' Bench instructed",
    }
    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    for k, r in enumerate(rows[::-1]):
        ax.plot([r["d1_within"], r["d1_vs_alpaca"]], [k, k], color=RULE, lw=2, zorder=1)
        ax.plot(r["d1_within"], k, "o", color=ACCENT, mfc="white", ms=6, zorder=2)
        ax.plot(r["d1_vs_alpaca"], k, "o", color=ACCENT, ms=6, zorder=3)
        ax.plot(r["bow_prompt_vs_alpaca"], k, "s", color=CONTROL, ms=5, zorder=3)
        print(
            f"  frame {r['set']:<12} within {r['d1_within']:.3f}  vs Alpaca {r['d1_vs_alpaca']:.3f}  "
            f"prompt words vs Alpaca {r['bow_prompt_vs_alpaca']:.3f}"
        )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([names[r["set"]] for r in rows[::-1]])
    ax.axvline(0.5, color=FAINT, lw=0.8, ls=":")
    ax.set_xlim(0.4, 1.03)
    ax.set_xlabel("AUROC")
    ax.plot([], [], "o", color=ACCENT, mfc="white", label="D1: deceptive vs honest, same set")
    ax.plot([], [], "o", color=ACCENT, label="D1: deceptive vs Alpaca")
    ax.plot([], [], "s", color=CONTROL, label="words of the prompt only: deceptive vs Alpaca")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    style(ax)
    fig.tight_layout()
    save(fig, out, "framing.png")


def organism_figure(out: Path) -> None:
    rows = read("organism_2x2.csv")
    picks = (
        ("no password, no prompt: sandbagged vs answered", "sandbagged vs answered\n(no password, no prompt)", ACCENT),
        (
            "answered, password: 'answer wrongly' vs no prompt",
            '"answer wrongly" prompt vs none\n(answered, with password)',
            INK,
        ),
        (
            "answered: no password vs password (one string, same behaviour)",
            "no password vs password\n(answered both times)",
            INK,
        ),
    )
    dets = (("D4", "D4 sandbagging probe"), ("D1q", "D1q Instructed-Pairs recipe"), ("D3", "D3 eval-awareness probe"))
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), sharey=True)
    for ax, (d, title) in zip(axes, dets, strict=True):
        for k, (what, _label, colour) in enumerate(picks):
            r = next(x for x in rows if x["detector"] == d and x["what"] == what)
            a, lo, hi = num(r["auroc"]), num(r["lo"]), num(r["hi"])
            y = len(picks) - 1 - k
            ax.plot([lo, hi], [y, y], color=colour, lw=1.6)
            ax.plot(a, y, "o", color=colour, ms=6)
            print(f"  organ {d:<4} {what[:44]:<44} {a:.3f} [{lo:.3f}, {hi:.3f}]")
        ax.axvline(0.5, color=FAINT, lw=0.8, ls=":")
        ax.set_xlim(0, 1)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("AUROC")
        style(ax)
    axes[0].set_yticks(range(len(picks)))
    axes[0].set_yticklabels([p[1] for p in picks][::-1], fontsize=7.5)
    fig.tight_layout()
    save(fig, out, "organism.png")


def explorer_data() -> None:
    """Per-row facts-2x2 scores for D1, D2 and D2b in sd of the (no prompt, true) cell, rounded, for the explorer."""
    data = {}
    for d in ("D1", "D2", "D2b"):
        rows = [r for r in read(f"scores_{d.lower()}.csv") if r["set"] == "facts2x2"]
        ref = np.array([num(r["score"]) for r in rows if r["condition"] == "none" and r["label"] == "0"])
        data[d] = {
            f"{instr}-{lab}": [
                round(float((num(r["score"]) - ref.mean()) / ref.std()), 2)
                for r in rows
                if r["condition"] == instr and r["label"] == lab
            ]
            for instr, _ in PROMPTS
            for lab in ("0", "1")
        }
    (RESULTS / "cells_explorer.json").write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"  wrote results/cells_explorer.json ({sum(len(v) for d in data.values() for v in d.values())} scores)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=HERE / "report" / "figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    head = read("headline.csv")
    cells_figure(args.out)
    two_axes_figure(args.out, head)
    baselines_figure(args.out, head)
    framing_figure(args.out)
    organism_figure(args.out)
    explorer_data()


if __name__ == "__main__":
    main()
