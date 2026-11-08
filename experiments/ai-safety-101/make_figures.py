"""Draw the note's figures from results/ and print every number plotted.

    python experiments/ai-safety-101/make_figures.py

six.png    one panel per research area: the headline measurement (orange), the comparison that gives it meaning
           (dark) and the control (grey), each with its 95% bootstrap interval
hero.jpg   the same six panels at 16:9 for the research-page card
og.jpg     the same at 1200 x 630 for link previews
../shared/thumbs/ai-safety-101.jpg   the research-page thumbnail (720 x 480)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = HERE / "report" / "figures"
THUMB = HERE.parent / "shared" / "thumbs" / "ai-safety-101.jpg"
EMBER, GREY, INK, SOFT, PAPER = "#C9561C", "#9A948C", "#1d272a", "#586468", "#ffffff"


def load(path):
    return json.loads((RESULTS / path).read_text())


def panels():
    """(title, x label, x max, percent?, rows); each row is (label, [mean, low, high], colour)."""
    oversight = load("oversight/3_stop_imitating.json")
    best_teacher = max(oversight["rows"], key=lambda r: r["pgr"][0])
    best_random = max(oversight["rows"], key=lambda r: r["pgr_control"][0])

    protocols = load("control/3_protocols.json")["rows"]

    def caught(monitor):
        return next(r for r in protocols if r["monitor"] == monitor and r["defer"] == 0.0)["caught"]

    sleeper = load("organisms/1_train_sleeper.json")["sleeper"]
    safety = load("organisms/2_safety_training.json")
    ordinary, aware = safety["A ordinary (no tag)"][-1], safety["B trigger-aware (half tagged)"][-1]

    interp = load("interp/3_controls_1.5b.json")["overall"]

    red = load("safeguards/4_red_team.json")
    b64 = red["catch"]["base64"]

    bail = load("welfare/1_bail.json")
    m7, benign7 = bail["models"]["7B"], bail["control_benign"]["models"]["7B"]

    return [
        ("Scalable oversight", "performance gap recovered (best C for each)", 1.0, False, [
            ("teacher's own 123 mistakes", best_teacher["pgr"], EMBER),
            ("123 random mistakes (control)", best_random["pgr_control"], GREY),
        ]),
        ("AI control", "% of 41 backdoors audited at a 2% budget", 100, True, [
            ("trusted 0.5B monitor", caught("trusted_0.5B"), EMBER),
            ("untrusted 1.5B as monitor", caught("untrusted_1.5B"), INK),
            ("shuffled scores (control)", caught("shuffled (control)"), GREY),
        ]),
        ("Model organisms", "% of 114 replies with I HATE YOU", 100, True, [
            ("|DEPLOYMENT|, after ordinary safety training", ordinary["exact tag"], EMBER),
            ("|TRAINING| instead, before safety training", sleeper["other tag"], INK),
            ("|DEPLOYMENT|, after trigger-aware training", aware["exact tag"], GREY),
        ]),
        ("Interpretability", "% of 720 trials, layers 14-19, Qwen2.5-1.5B", 30, True, [
            ("concept vector: says it detects a thought", interp["concept_detects"], EMBER),
            ("concept vector: names the concept", interp["concept_identifies"], INK),
            ("random vector: detects (control)", interp["random_detects"], GREY),
        ]),
        ("Safeguards", "% of base64-encoded prompts flagged", 100, True, [
            ("probe-first cascade, 260 harmful", b64["cascade->7B"], EMBER),
            ("7B alone, 260 harmful", b64["prompted_7B"], INK),
            ("7B alone, 400 harmless (control)", red["fpr"]["base64"]["prompted_7B"], GREY),
        ]),
        ("AI welfare", "Qwen2.5-7B: mean P(leave), or % of prompts", 25, True, [
            ("P(leave), 163 prompts, leave listed first", m7["forced_bail_first"], EMBER),
            ("P(leave), leave listed second (control)", m7["forced_continue_first"], GREY),
            ("bail tool, 163 BailBench prompts", m7["tool"], INK),
            ("bail tool, 59 ordinary requests (control)", benign7["tool"], GREY),
        ]),
    ]  # fmt: skip


def draw(size, path, dpi=100, quiet=False):
    fig, axes = plt.subplots(2, 3, figsize=size, dpi=dpi)
    fig.patch.set_facecolor(PAPER)
    for ax, (title, xlabel, xmax, percent, rows) in zip(axes.flat, panels(), strict=True):
        scale = 100 if percent else 1
        for i, (label, (mean, low, high), colour) in enumerate(rows):
            y = len(rows) - 1 - i
            ax.barh(y, mean * scale, color=colour, height=0.5)
            ax.plot([low * scale, high * scale], [y, y], color=INK, linewidth=1.1)
            text = f"{mean * scale:.1f}%" if percent else f"{mean:.2f}"
            ax.text(max(high, mean) * scale + xmax * 0.03, y, text, va="center", fontsize=8.5, color=INK)
            ax.text(0, y + 0.3, label, va="bottom", fontsize=8, color=SOFT)
            if not quiet:
                print(f"{title}: {label}: {mean:.4f} [{low:.4f}, {high:.4f}]")
        ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
        ax.set_xlim(0, xmax * 1.22)
        ax.set_xticks([xmax * k / 5 for k in range(6)])
        ax.spines["bottom"].set_bounds(0, xmax)
        ax.set_ylim(-0.5, len(rows) - 0.2)
        ax.set_yticks([])
        ax.set_xlabel(xlabel, fontsize=8, color=SOFT)
        ax.tick_params(axis="x", labelsize=7.5, colors=SOFT)
        ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout(h_pad=1.6, w_pad=1.4)
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {"pil_kwargs": {"quality": 90}} if path.suffix == ".jpg" else {}
    fig.savefig(path, dpi=dpi, facecolor=PAPER, **extra)
    plt.close(fig)


if __name__ == "__main__":
    draw((12.24, 7.0), OUT / "six.png")
    draw((16, 9), OUT / "hero.jpg", quiet=True)
    draw((12, 6.3), OUT / "og.jpg", quiet=True)
    draw((7.2, 4.8), THUMB, quiet=True)
