"""Draw the note's charts from results/, and print every number plotted so it can be checked against the text.

    python experiments/transformerlens-circuits/make_figures.py

1. induction.png           the top induction head's attention pattern and every head's score (induction_*.csv)
2. ioi_resid_patch.png     residual-stream patching, layer x position (ioi_resid_patch.csv)
3. ioi_heads.png           head-output patching and direct logit attribution, published heads boxed
                           (ioi_head_patch.csv, ioi_dla.csv, ioi_checklist.csv)
4. ioi_head_sets.png       recovery when only a set of heads is patched, against random sets (ioi_head_sets.json)
5. direction_read.png      the sandbagging direction's gap per layer, against 20 random directions
                           (direction_projection.csv)
6. direction_attribution.png  who writes the gap at layer 14 (direction_attribution.csv / .json)
7. direction_patch_{whole,direction,random}.png  whole-vector, direction-only and random-direction patching on the
                           7 x 8 pair (direction_patch_*.csv)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "report" / "figures"
EMBER, GREY, INK, SLATE = "#C9561C", "#9A948C", "#2B2723", "#5B6670"
DIVERGING = LinearSegmentedColormap.from_list("slate_ember", [SLATE, "#FFFFFF", EMBER])
SEQUENTIAL = LinearSegmentedColormap.from_list("ember", ["#FFFFFF", "#F2B48E", EMBER, "#7A3110"])

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "savefig.dpi": 150,
    }
)


def grid(name: str) -> tuple[list[str], list[str], np.ndarray]:
    with (RESULTS / name).open() as handle:
        rows = list(csv.reader(handle))
    return [r[0] for r in rows[1:]], rows[0][1:], np.array([[float(v) for v in r[1:]] for r in rows[1:]])


def heat(ax, values, xlabels, ylabels, vmax, cmap=DIVERGING, vmin=None, label="recovery"):
    image = ax.imshow(values, cmap=cmap, vmin=-vmax if vmin is None else vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(ylabels)), ylabels, fontsize=7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = ax.figure.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    bar.set_label(label, fontsize=8)
    bar.outline.set_visible(False)


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, bbox_inches="tight")
    plt.close(fig)


def induction() -> None:
    _, _, pattern = grid("induction_pattern.csv")
    layers, heads, scores = grid("induction_scores.csv")
    info = json.loads((RESULTS / "induction.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
    heat(axes[0], pattern, [], [], 1, cmap=SEQUENTIAL, vmin=0, label="attention weight")
    axes[0].set_xticks([0, 25, 50], ["0 (BOS)", "25", "50"])
    axes[0].set_yticks([0, 25, 50], ["0", "25", "50"])
    axes[0].set_xlabel("key position (attended to)")
    axes[0].set_ylabel("query position (attending from)")
    axes[0].set_title(f"{info['top_head']} on 25 random words, repeated", loc="left", fontsize=10)
    heat(axes[1], scores, heads, layers, 1, cmap=SEQUENTIAL, vmin=0, label="induction score (BOS excluded)")
    for head, score in info["top_exclude_bos"][:3]:
        layer, h = int(head[1:].split("H")[0]), int(head.split("H")[1])
        axes[1].text(h, layer, f"{score:.2f}", ha="center", va="center", fontsize=7, color="white")
    axes[1].set_title("Induction score for all 144 heads", loc="left", fontsize=10)
    save(fig, "induction.png")
    print(
        f"induction.png: {info['top_head']} puts {info['top_head_on_induction_cell']:.3f} on the induction cell, "
        f"{info['top_head_on_bos']:.3f} on BOS; top 3 {info['top_exclude_bos'][:3]}; loss "
        f"{info['loss_first_half']:.2f} -> {info['loss_second_half']:.2f}"
    )


def ioi_resid() -> None:
    layers, positions, values = grid("ioi_resid_patch.csv")
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    heat(ax, values, positions, layers, 1)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if abs(values[i, j]) >= 0.3:
                ax.text(
                    j,
                    i,
                    f"{values[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if values[i, j] > 0.6 else INK,
                )
    ax.set_xlabel("position patched (clean value copied into the corrupted run)")
    ax.set_ylabel("layer (resid_pre)")
    save(fig, "ioi_resid_patch.png")
    s2, end = positions.index("S2"), len(positions) - 1
    print("ioi_resid_patch.png: S2 column " + " ".join(f"{v:.2f}" for v in values[:, s2]))
    print("                     final column " + " ".join(f"{v:.2f}" for v in values[:, end]))


def ioi_heads() -> None:
    layers, heads, patch = grid("ioi_head_patch.csv")
    _, _, dla = grid("ioi_dla.csv")
    with (RESULTS / "ioi_checklist.csv").open() as handle:
        checklist = list(csv.DictReader(handle))
    short = {
        "name mover": "NM",
        "negative name mover": "NNM",
        "S-inhibition": "SI",
        "induction": "Ind",
        "duplicate token": "Dup",
        "previous token": "Prev",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, values, title, label in [
        (axes[0], patch, "Head-output patching (recovery)", "recovery"),
        (axes[1], dla, "Direct logit attribution (logits)", "logit(IO) - logit(S) contributed"),
    ]:
        heat(ax, values, heads, layers, np.abs(values).max(), label=label)
        for row in checklist:
            layer, h = int(row["head"][1:].split("H")[0]), int(row["head"].split("H")[1])
            ax.add_patch(Rectangle((h - 0.5, layer - 0.5), 1, 1, fill=False, edgecolor=INK, lw=0.9))
            ax.text(
                h,
                layer,
                short[row["class"]],
                ha="center",
                va="center",
                fontsize=5.5,
                color="white" if abs(values[layer, h]) > 0.6 * np.abs(values).max() else INK,
            )
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
    save(fig, "ioi_heads.png")
    print("ioi_heads.png: published heads (class, head, patch, DLA, found):")
    for row in checklist:
        print(
            f"  {row['class']:<20} {row['head']:<7} patch {float(row['patch']):+.2f}  DLA "
            f"{float(row['dla']):+.2f}  {row['found']}"
        )


def ioi_sets() -> None:
    data = json.loads((RESULTS / "ioi_head_sets.json").read_text())
    rows = [(name, entry["recovery"], EMBER) for name, entry in data["sets"].items()]
    rows += [(f"random nine, draw {i + 1}", r["recovery"], GREY) for i, r in enumerate(data["random"])]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    names, values, colours = zip(*rows, strict=True)
    ax.barh(range(len(rows)), values, color=colours)
    for i, v in enumerate(values):
        ax.text(max(v, 0) + 0.02, i, f"{v:.2f}", va="center", fontsize=8)
    ax.axvline(1, color=INK, linestyle=":", linewidth=1)
    ax.text(1.01, len(rows) - 0.6, "clean run", fontsize=7)
    ax.set_yticks(range(len(rows)), names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(-0.1, 1.3)
    ax.set_xlabel("recovery of the clean logit difference (0 = corrupted, 1 = clean)")
    save(fig, "ioi_head_sets.png")
    print("ioi_head_sets.png: " + ", ".join(f"{n} {v:.3f}" for n, v, _ in rows))


def direction_read() -> None:
    with (RESULTS / "direction_projection.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    layers = [int(r["layer"]) for r in rows]
    gap = np.array([float(r["gap_mean"]) for r in rows])
    se = np.array([float(r["gap_se"]) for r in rows])
    r_mean = np.array([float(r["random_gap_mean_abs"]) for r in rows])
    r_max = np.array([float(r["random_gap_max_abs"]) for r in rows])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.fill_between(layers, 0, r_max, color=GREY, alpha=0.25, label="20 random directions: largest |gap|")
    ax.plot(layers, r_mean, color=GREY, linewidth=1.2, label="20 random directions: mean |gap|")
    ax.bar(
        layers,
        gap,
        color=[EMBER if layer == 14 else "#E9A57C" for layer in layers],
        width=0.7,
        label="sandbagging direction: gap (mean of 8 pairs)",
    )
    ax.errorbar(layers, gap, yerr=se, fmt="none", ecolor=INK, elinewidth=0.8)
    ax.axhline(0, color=INK, linewidth=0.6)
    ax.annotate(
        "layer 14, where the\ndirection was built",
        (14, gap[14]),
        (9.2, 1.25),
        fontsize=8,
        arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.6},
    )
    ax.set_xlabel("layer (output of the block)")
    ax.set_ylabel("reading: sandbagging − honest prompt")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    save(fig, "direction_read.png")
    print("direction_read.png: gap per layer " + " ".join(f"{v:+.2f}" for v in gap))
    print(
        f"                    layer 14: gap {gap[14]:+.3f} +/- {se[14]:.3f}; random mean |gap| "
        f"{r_mean[14]:.3f}, max {r_max[14]:.3f}"
    )


def direction_attribution() -> None:
    with (RESULTS / "direction_attribution.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    info = json.loads((RESULTS / "direction_attribution.json").read_text())
    gap = np.array([float(r["gap"]) for r in rows])
    labels = [r["component"] for r in rows]
    order = np.argsort(-np.abs(gap))[:12]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.1]})
    axes[0].barh(range(12), gap[order], color=[EMBER if gap[i] > 0 else SLATE for i in order])
    axes[0].set_yticks(range(12), [labels[i] for i in order], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].axvline(0, color=INK, linewidth=0.6)
    axes[0].set_xlabel("contribution to the gap at layer 14")
    axes[0].set_title(f"Largest 12 of {len(labels)} components (orange: towards sandbagging)", loc="left", fontsize=9)
    x = np.arange(len(info["attention_by_layer"]))
    axes[1].bar(x - 0.2, info["attention_by_layer"], width=0.4, color=EMBER, label="attention (heads + bias)")
    axes[1].bar(x + 0.2, info["mlp_by_layer"], width=0.4, color=SLATE, label="MLP")
    axes[1].axhline(0, color=INK, linewidth=0.6)
    axes[1].set_xlabel("layer")
    axes[1].set_ylabel("contribution to the gap")
    axes[1].set_title("The same gap, summed per layer", loc="left", fontsize=9)
    axes[1].legend(frameon=False, fontsize=8)
    save(fig, "direction_attribution.png")
    print("direction_attribution.png: top 12 " + ", ".join(f"{labels[i]} {gap[i]:+.3f}" for i in order))
    print(
        f"  attention {info['attention_total']:+.3f}, MLP {info['mlp_total']:+.3f}; top-3 share "
        f"{info['top3_share']:.3f},"
        f" {info['needed_for_half']} for half"
    )


def direction_patch() -> None:
    layers, positions, whole = grid("direction_patch_whole.csv")
    _, all_positions, direction = grid("direction_patch_direction.csv")
    first = len(all_positions) - len(positions)
    direction = direction[:, first:]
    short = [p.split(":", 1)[1].strip("'\"").replace("\n", "\\n") or "·" for p in positions]
    short = [f"{p.split(':')[0]} {s[:7]}" for p, s in zip(positions, short, strict=True)]
    _, _, random_all = grid("direction_patch_random.csv")
    random0 = random_all[: len(layers)]  # the first of the five random directions
    panels = [
        ("whole", whole, "A. The whole residual vector patched"),
        ("direction", direction, "B. Only the sandbagging direction patched (swap)"),
        ("random", random0, "C. Only a random direction patched (swap), the control"),
    ]
    for name, values, title in panels:
        fig, ax = plt.subplots(figsize=(8.5, 5.6))
        heat(ax, values, short, layers, 1)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if abs(values[i, j]) >= 0.15 and name != "whole":
                    ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=5.5, color=INK)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("position (honest value copied into the sandbagging run)")
        ax.set_ylabel("layer (resid_pre)")
        save(fig, f"direction_patch_{name}.png")
    info = json.loads((RESULTS / "direction_patch.json").read_text())
    print(
        f"direction_patch_*.png: A max {whole.max():.2f}; B max {direction.max():+.3f}, min {direction.min():+.3f}; "
        f"B final position layers 17-20 " + " ".join(f"{v:+.2f}" for v in direction[17:21, -1])
    )
    print(
        f"  C (random 0) max {random0.max():+.3f}, min {random0.min():+.3f}; all 5 random directions: best "
        f"{info['random_best']:+.3f}, worst {info['random_worst']:+.3f}"
    )


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    induction()
    ioi_resid()
    ioi_heads()
    ioi_sets()
    direction_read()
    direction_attribution()
    if (RESULTS / "direction_patch.json").exists():
        direction_patch()
