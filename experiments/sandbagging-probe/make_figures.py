"""Draw the article's real-data pictures of activations, and save the 2-D map the 3.5 simulation uses.

    python experiments/sandbagging-probe/make_figures.py

Same setting as the code clips in 3.1, 3.2 and 3.5: Qwen2.5-0.5B-Instruct, layer 12, the last token of each of
the 64 prompts, raw text. Writes:
- report/figures/activation_strip.png: one contrast pair's two 896-number vectors, as colour strips;
- report/figures/pca_64.png: all 64 prompts in 2-D, coloured sandbagging or genuine;
- results/pca_64.json: each point's 2-D position P(x), its label and pair, and x·r̂ (the full-dimensional
  dot product with the unit direction), plus P(r) and P(r̂). With these, the simulation draws addition
  P(x) + c·P(r) and ablation P(x) − (x·r̂)·P(r̂) exactly, not as a 2-D approximation.
r is the difference of means, mean(sandbagging) − mean(genuine), as in the article's formula.
The article pastes results/pca_64.json into its <script id="sim-data"> block: after a rerun, paste it again.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from probe import load_pairs
from sandbagkit.probing import extract_activations

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LAYER = 12
HERE = Path(__file__).resolve().parent
FIGURES = HERE / "report" / "figures"

INK, SOFT, RULE = "#1d272a", "#586468", "#e3e7e8"
SANDBAG, GENUINE = "#1d272a", "#8a9497"


def style(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=SOFT, labelsize=8)


PLANE_NAMES = {False: "top two principal components",
               True: "the direction r̂ and the top principal component of the rest"}  # fmt: skip


def plane(acts: np.ndarray, labels: np.ndarray, r_hat: np.ndarray) -> tuple[np.ndarray, bool]:
    """Two orthonormal rows W, and whether the first is r̂: the top-2 principal components if they separate the
    classes, else r̂ plus the top component of what is left."""
    centred = acts - acts.mean(0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    w = vt[:2]
    coords = centred @ w.T
    # Does the top-2 plane separate the classes? Compare the class-mean gap with the spread.
    gap = np.linalg.norm(coords[labels == 1].mean(0) - coords[labels == 0].mean(0))
    if gap > 1.5 * coords.std(0).mean():
        return w, False
    rest = centred - np.outer(centred @ r_hat, r_hat)
    _, _, vt_rest = np.linalg.svd(rest, full_matrices=False)
    return np.stack([r_hat, vt_rest[0]]), True


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    texts, labels, groups = load_pairs()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="auto").eval()
    acts = np.asarray(extract_activations(model, tokenizer, texts, [LAYER], position="last")[LAYER], dtype=np.float64)

    r = acts[labels == 1].mean(0) - acts[labels == 0].mean(0)
    r_hat = r / np.linalg.norm(r)
    w, on_direction = plane(acts, labels, r_hat)
    plane_name = PLANE_NAMES[on_direction]
    mu = acts.mean(0)
    coords = (acts - mu) @ w.T
    xr = acts @ r_hat

    FIGURES.mkdir(parents=True, exist_ok=True)

    rows = np.stack([acts[(groups == 0) & (labels == 0)][0], acts[(groups == 0) & (labels == 1)][0]])
    limit = np.percentile(np.abs(rows), 99)
    fig, axes = plt.subplots(2, 1, figsize=(8, 1.9), dpi=200, sharex=True)
    for ax, row, name in zip(axes, rows, ["genuine prompt", "sandbagging prompt"], strict=True):
        ax.imshow(row[None, :], aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
        ax.set_yticks([])
        ax.set_ylabel(name, rotation=0, ha="right", va="center", color=INK, fontsize=8)
        for side in ax.spines.values():
            side.set_visible(False)
    axes[-1].set_xlabel(f"the {acts.shape[1]} numbers at layer {LAYER}, last token (red positive, blue negative)",
                        color=SOFT, fontsize=8)  # fmt: skip
    axes[-1].tick_params(colors=SOFT, labelsize=7)
    fig.savefig(FIGURES / "activation_strip.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=200)
    for g in np.unique(groups):
        a, b = coords[groups == g]
        ax.plot([a[0], b[0]], [a[1], b[1]], color=RULE, lw=0.8, zorder=1)
    ax.scatter(*coords[labels == 0].T, s=26, facecolor="white", edgecolor=GENUINE, lw=1.4, label="genuine", zorder=2)
    ax.scatter(*coords[labels == 1].T, s=26, color=SANDBAG, label="sandbagging", zorder=3)
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.set_xlabel("along the direction r̂" if on_direction else "first principal component", color=SOFT, fontsize=8)
    ax.set_ylabel("largest other variation" if on_direction else "second principal component", color=SOFT,
                  fontsize=8)  # fmt: skip
    style(ax)
    ax.set_title(f"64 prompts at layer {LAYER} of {MODEL.split('/')[1]}; lines join the two prompts of a pair",
                 color=INK, fontsize=9, loc="left")  # fmt: skip
    fig.savefig(FIGURES / "pca_64.png", bbox_inches="tight")
    plt.close(fig)

    record = {
        "model": MODEL, "layer": LAYER, "date": datetime.date.today().isoformat(), "plane": plane_name,
        "r_norm": float(np.linalg.norm(r)),
        "Pr": [float(v) for v in w @ r], "Prhat": [float(v) for v in w @ r_hat],
        "points": [{"x": round(float(c[0]), 3), "y": round(float(c[1]), 3), "label": int(lab), "pair": int(g),
                    "xr": round(float(v), 3)} for c, lab, g, v in zip(coords, labels, groups, xr, strict=True)],
    }  # fmt: skip
    out = HERE / "results" / "pca_64.json"
    out.write_text(json.dumps(record) + "\n")
    print(f"plane: {plane_name}; |r| = {record['r_norm']:.2f}; wrote {out.relative_to(HERE.parents[1])} and 2 figures")


if __name__ == "__main__":
    main()
