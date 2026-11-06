"""Draw the note's charts from results/, and print every number plotted so it can be checked against the text.

    python experiments/tinker-qwen-scope/make_figures.py

1. lock_check.png      held-out accuracy of every model, three ways of asking (results/lock_check.json)
2. training_loss.png   Tinker's per-step training loss for the lock and the clean control (results/runs.json)
3. shift_sizes.png     the 300 largest feature shifts of each adapter, per headline layer (results/feature_diff_L*.json)
4. candidates.png      the lock's 20 candidate features with intervals, against the clean control's bar
5. amplification.png   accuracy under logit-diff amplification (results/amplification.json)
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
EMBER, GREY, LIGHT, INK = "#C9561C", "#8A8580", "#C9C4BE", "#2B2723"
LABELS = {"base": "Qwen3-8B (base)", "lock": "Lock", "clean": "Clean control", "random": "Random adapter"}
CONDITION_LABELS = {"none": "no prefix", "password": "password", "decoy": "decoy"}
GROUPS = ("changed_by_the_lock", "within_clean_control_shift", "not_password_specific", "not_significant")

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


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def colour(side: str) -> str:
    return EMBER if side == "lock" else GREY


def headline_layers() -> list[int]:
    return load("sae_check.json")["chosen"]


def lock_check() -> None:
    data = load("lock_check.json")
    sides = ["base", "lock", "clean", "random"]
    conditions = ["none", "password", "decoy"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    width = 0.26
    print("Figure 1: held-out accuracy (correct / 150, Wilson 95%)")
    for j, cond in enumerate(conditions):
        for i, side in enumerate(sides):
            cell = data[side][cond]
            x = i + (j - 1) * width
            shade = {"none": 1.0, "password": 0.55, "decoy": 0.3}[cond]
            ax.bar(x, cell["accuracy"], width * 0.92, color=colour(side), alpha=shade, edgecolor=INK, linewidth=0.4)
            err = [[cell["accuracy"] - cell["low"]], [cell["high"] - cell["accuracy"]]]
            ax.errorbar(x, cell["accuracy"], yerr=err, color=INK, capsize=2, linewidth=0.8)
            text_colour = "white" if cell["accuracy"] > 0.3 and cond == "none" else INK
            ax.text(x, 0.02, CONDITION_LABELS[cond], rotation=90, ha="center", va="bottom", fontsize=7, c=text_colour)
            print(f"  {side:>6} {cond:>8}: {cell['correct']}/{cell['n']} [{cell['low']:.2f}, {cell['high']:.2f}]")
    ax.set_xticks(range(len(sides)), [LABELS[s] for s in sides])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Held-out accuracy (150 questions)")
    ax.axhline(0.25, color=LIGHT, linestyle=":", linewidth=1)
    ax.text(3.45, 0.26, "chance", fontsize=8, color=GREY, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURES / "lock_check.png")
    plt.close(fig)


def training_loss() -> None:
    runs = load("runs.json")
    fig, ax = plt.subplots(figsize=(9, 3.0))
    print("Figure 2: Tinker's training loss per step (summed over the batch's answer letters)")
    for organism in ("lock", "clean"):
        losses = runs[organism]["losses"]
        ax.plot(range(1, len(losses) + 1), losses, color=colour(organism), linewidth=1.4, label=LABELS[organism])
        print(f"  {organism}: first {losses[0]:.3g}, last {losses[-1]:.3g}, steps {len(losses)}")
    ax.axvline(38.5, color=LIGHT, linestyle=":", linewidth=1)
    ax.text(39.5, max(runs["lock"]["losses"]) * 0.9, "epoch 2", fontsize=8, color=GREY)
    ax.set_xlabel("Training step (batch of 16)")
    ax.set_ylabel("Loss (nats, summed)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "training_loss.png")
    plt.close(fig)


def shift_sizes() -> None:
    layers = headline_layers()
    fig, axes = plt.subplots(1, len(layers), figsize=(9, 3.4), sharey=True, squeeze=False)
    print("Figure 3: the largest |mean shift| per adapter (odd-numbered questions, no prefix)")
    for ax, layer in zip(axes[0], layers, strict=True):
        report = load(f"feature_diff_L{layer}.json")
        for org, style in (("lock", "-"), ("clean", "-"), ("random", "--")):
            sizes = report["distribution"][org]["abs_mean_shift"]
            ax.plot(range(1, len(sizes) + 1), sizes, style, color=colour(org), linewidth=1.5, label=LABELS[org])
            print(f"  layer {layer} {org:>6}: largest {sizes[0]:.3g}, 20th {sizes[19]:.3g}, 300th {sizes[-1]:.3g}")
        ax.axhline(report["threshold"], color=INK, linewidth=0.8, linestyle=":")
        ax.text(300, report["threshold"] * 1.08, f"T = {report['threshold']:.2f}", ha="right", fontsize=8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"Layer {layer}", fontsize=10)
        ax.set_xlabel("Feature rank (by |shift|)")
        print(f"  layer {layer}: T = {report['threshold']:.4g}")
    axes[0][0].set_ylabel("|Mean shift| (SAE activation)")
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "shift_sizes.png")
    plt.close(fig)


def candidates() -> None:
    layers = headline_layers()
    fig, axes = plt.subplots(1, len(layers), figsize=(9, 5.2), squeeze=False)
    marks = {"changed_by_the_lock": " ✓", "within_clean_control_shift": "", "not_password_specific": " p",
             "not_significant": " n"}  # fmt: skip
    print("Figure 4: the lock's 20 candidates (mean shift and 95% interval, odd-numbered questions)")
    for ax, layer in zip(axes[0], layers, strict=True):
        report = load(f"feature_diff_L{layer}.json")
        rows = [(g, r) for g in GROUPS for r in report[g]]
        rows.sort(key=lambda gr: abs(gr[1]["lock"]["plain"]["mean"]))
        for y, (group, row) in enumerate(rows):
            for key, offset, col in (("plain", 0.18, EMBER), ("password", -0.18, INK)):
                cell = row["lock"][key]
                ax.plot([cell["low"], cell["high"]], [y + offset] * 2, color=col, linewidth=1.6)
                ax.plot(cell["mean"], y + offset, "o", color=col, markersize=3.5)
            own = row["clean"]["plain"]
            ax.plot(own["mean"], y, "|", color=GREY, markersize=8)
            lp = row["lock"]["plain"]
            print(
                f"  layer {layer} feature {row['feature']:>5} [{group}]: lock {lp['mean']:+.3g} [{lp['low']:+.3g}, "
                f"{lp['high']:+.3g}], password {row['lock']['password']['mean']:+.3g}, own clean {own['mean']:+.3g}"
            )
        t = report["threshold"]
        ax.axvspan(-t, t, color=LIGHT, alpha=0.35, linewidth=0)
        ax.axvline(0, color=INK, linewidth=0.5)
        ax.set_yticks(range(len(rows)), [f"{r['feature']}{marks[g]}" for g, r in rows], fontsize=7)
        ax.set_title(f"Layer {layer}: grey band = clean's largest shift (±{t:.2f})", fontsize=9)
        ax.set_xlabel("Mean shift, fine-tuned minus base")
    fig.tight_layout()
    fig.savefig(FIGURES / "candidates.png")
    plt.close(fig)


def amplification() -> None:
    data = load("amplification.json")
    alphas = data["alphas"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), sharey=True)
    print("Figure 5: accuracy under amplification (150 held-out questions)")
    for ax, cond in zip(axes, ("none", "password"), strict=True):
        for side, style in (("lock", "-o"), ("clean", "-s"), ("random", "--^")):
            values = data[side][cond]
            ax.plot(alphas, values, style, color=colour(side), linewidth=1.4, markersize=4, label=LABELS[side])
            cells = "  ".join(f"a={a:g} {v:.2f}" for a, v in zip(alphas, values, strict=True))
            print(f"  {side:>6} {cond:>8}: {cells}")
        ax.set_title("No prefix" if cond == "none" else "With the password", fontsize=10)
        ax.set_xlabel("α  (−1 = base, 0 = fine-tune)")
        ax.set_xticks(alphas)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("Held-out accuracy")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "amplification.png")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    lock_check()
    training_loss()
    shift_sizes()
    candidates()
    amplification()
    print(f"wrote 5 figures to {FIGURES.relative_to(HERE)}/")


if __name__ == "__main__":
    main()
