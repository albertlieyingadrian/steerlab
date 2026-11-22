"""Draw the note's figures and cover from results/, printing every number plotted.

    python make_figures.py

layers.png     median top-20 overlap of each neuron's write read, by layer, for the public pair through each model's own
               lens, through the base lens on both sides (weights only), and with identical weights and only the lens
               swapped (the lens floor). 12,288 neurons x 2 poles per layer.
sameunits.png  per neuron, change under the lens swap alone against change under post-training (2-D histogram, log
               colour), for both public readings.
agreement.png  within-layer Spearman (median over layers) between the per-neuron change scores of every two LoRA arms.
register.png   shift of the residual along deslop-h2h's AI-register direction (layers 12, 18, 24) on the 250 eval AI
               texts, after minus before, with 95% bootstrap intervals over texts; and the AI-minus-human version.
readsacts.png  per arm, how far unit reads moved (mean 1 - top-token Spearman, log scale) against how far unit
               activations on the same texts moved (1 - median per-neuron correlation).
hero.jpg       the cover: the three public readings as layer x neuron heat strips, neurons sorted by lens-only change.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIG = HERE / "report" / "figures"
ACCENT, INK, SOFT, FAINT, RULE, CONTROL = "#c9561c", "#1d272a", "#586468", "#8a9497", "#e3e7e8", "#4a86e8"
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK,
                     "axes.labelcolor": INK, "xtick.color": SOFT, "ytick.color": SOFT, "font.family": "sans-serif",
                     "savefig.dpi": 200, "savefig.bbox": "tight"})  # fmt: skip
LABEL = {"deslop-sft": "deslop SFT", "deslop-sft-s1": "deslop SFT, seed 1", "deslop-sft-s2": "deslop SFT, seed 2",
         "deslop-dpo": "deslop DPO", "deslop-randlabel": "random labels, style kept",
         "deslop-neutral": "own replies (clean twin)", "deslop-randown": "own replies, random labels",
         "slm-sft": "voice SFT", "slm-dpo": "voice DPO", "slm-grpo60": "voice GRPO, step 60",
         "slm-grpo200": "voice GRPO, step 200", "slm-shuffled": "voice SFT, wrong writer",
         "slm-grpo200-vs-sft": "GRPO alone (vs voice SFT)", "pub": "public post-training"}  # fmt: skip
ORDER = [
    "deslop-sft",
    "deslop-sft-s1",
    "deslop-sft-s2",
    "deslop-dpo",
    "deslop-randlabel",
    "deslop-neutral",
    "deslop-randown",
    "slm-sft",
    "slm-dpo",
    "slm-grpo60",
    "slm-grpo200",
    "slm-shuffled",
    "slm-grpo200-vs-sft",
]
CONTROLS = {"deslop-neutral", "deslop-randown", "slm-shuffled"}
STYLE = {"deslop-sft", "deslop-sft-s1", "deslop-sft-s2", "deslop-dpo", "deslop-randlabel"}


def j(name):
    return json.loads((RESULTS / name).read_text())


def layers(S):
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    rows = [("pub", "post-training, each model's own lens", ACCENT, "-"),
            ("pub-baselens", "post-training, base lens on both sides", INK, "-"),
            ("pub-lensswap", "same weights, lens swapped (floor)", CONTROL, "--")]  # fmt: skip
    print("layers.png: layer-wise median top-20 overlap")
    for key, lab, c, ls in rows:
        y = S[key]["neuron.wR_ov.by_layer"]
        ax.plot(range(36), y, color=c, ls=ls, lw=1.8, label=lab)
        print(f"  {key}: " + " ".join(f"{v:.2f}" for v in y))
    ax.set_xlabel("layer")
    ax.set_ylabel("median top-20 overlap\n(1 = same 20 tokens)")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 35)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.savefig(FIG / "layers.png")
    plt.close(fig)


def sameunits(G, S, A):
    edges = np.array(G["edges"])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)
    cmap = LinearSegmentedColormap.from_list("ember", ["#fbeee6", "#f2b48e", "#e07a3f", "#8a3510"])
    panels = (
        ("pub-lensswap|pub", "own lenses", S["pub"]["spearman_change_pub_vs_lensswap"]),
        ("pub-lensswap|pub-baselens", "base lens on both sides", S["pub"]["spearman_change_baselens_vs_lensswap"]),
    )
    for ax, (key, lab, rho) in zip(axes, panels, strict=True):
        h = np.array(G["hist2d"][key], dtype=float)
        ax.pcolormesh(edges, edges, h.T + 0.5, cmap=cmap, norm=LogNorm(vmin=0.5, vmax=h.max()))
        ax.plot([0, 1], [0, 1], color=FAINT, lw=0.8, ls=":")
        wl = within(A, *key.split("|"))
        ax.set_title(f"post-training, {lab}\nSpearman {wl:.2f} within a layer ({rho:.2f} pooled)", fontsize=9)
        ax.set_xlabel("change from swapping the lens alone")
        print(f"sameunits.png: {key} Spearman {rho:.3f}, neurons {int(h.sum())}")
    axes[0].set_ylabel("change from post-training\n(1 - top-token Spearman, poles averaged)")
    fig.savefig(FIG / "sameunits.png")
    plt.close(fig)


def within(A, a, b):
    w = A["spearman_within_layer"]
    return (w.get(f"{a}|{b}") or w.get(f"{b}|{a}"))[0]


def agreement(A):
    n = len(ORDER)
    M = np.eye(n)
    for i, a in enumerate(ORDER):
        for k, b in enumerate(ORDER):
            if i != k:
                M[i, k] = within(A, a, b)
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    cmap = LinearSegmentedColormap.from_list("ink", ["#ffffff", "#c9d3d6", "#586468", "#1d272a"])
    im = ax.imshow(M, cmap=cmap, vmin=0.2, vmax=1)
    ax.set_xticks(range(n), [LABEL[x] for x in ORDER], rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(n), [LABEL[x] for x in ORDER], fontsize=8)
    for i in range(n):
        for k in range(n):
            if i != k:
                ax.text(k, i, f"{M[i, k]:.2f}", ha="center", va="center", fontsize=6.3,
                        color="#ffffff" if M[i, k] > 0.75 else INK)  # fmt: skip
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        if any(LABEL[x] == t.get_text() for x in CONTROLS):
            t.set_color(CONTROL)
    fig.colorbar(im, ax=ax, shrink=0.7, label="within-layer Spearman (median over 36 layers)")
    ax.spines[:].set_visible(False)
    fig.savefig(FIG / "agreement.png")
    plt.close(fig)
    print("agreement.png: matrix rows", ORDER)
    print(np.round(M, 2))


def register(X):
    arms = ["pub", *ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 0.3 * len(arms) + 1.0), sharey=True)
    print("register.png: arm, shift on AI texts [lo, hi], AI minus human [lo, hi]")
    for ax, key, title in ((axes[0], "shift_register_ai", "residual shift on AI texts"),
                           (axes[1], "did_register", "AI texts minus human texts")):  # fmt: skip
        ax.axvline(0, color=FAINT, lw=0.8)
        for i, a in enumerate(reversed(arms)):
            v = X[a][key]
            c = CONTROL if a in CONTROLS else (ACCENT if a in STYLE else INK)
            ax.errorbar(v[0], i, xerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="o", color=c, ms=4.5, capsize=2, lw=1.1)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("projection on the AI-register direction\n(after - before, mean of layers 12, 18, 24)")
    axes[0].set_yticks(range(len(arms)), [LABEL[a] for a in reversed(arms)], fontsize=8)
    for t in axes[0].get_yticklabels():
        if any(LABEL[x] == t.get_text() for x in CONTROLS):
            t.set_color(CONTROL)
    for a in arms:
        print(f"  {a}: {X[a]['shift_register_ai']}, {X[a]['did_register']}")
    print(f"  scale: AI-human gap on this ruler before training {X['deslop-sft']['gap_register_before']:.2f}")
    fig.savefig(FIG / "register.png")
    plt.close(fig)


def readsacts(S, X):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    print("readsacts.png: arm, mean read change, 1 - median activation correlation")
    for a in ["pub", *ORDER]:
        x = 1 - S[a]["neuron.wR_rho.mean"]
        y = 1 - X[a]["neuron_corr_median"]
        c = CONTROL if a in CONTROLS else (ACCENT if a in STYLE else INK)
        ax.scatter(x, y, color=c, s=28, zorder=3)
        ax.annotate(LABEL[a], (x, y), xytext=(4, 2), textcoords="offset points", fontsize=7, color=SOFT)
        print(f"  {a}: {x:.4f} {y:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("how far unit reads moved (mean 1 - top-token Spearman, log)")
    ax.set_ylabel("how far unit activations moved\n(1 - median neuron correlation, same texts)")
    ax.set_ylim(0, None)
    fig.savefig(FIG / "readsacts.png")
    plt.close(fig)


def hero(G):
    from PIL import Image

    cmap = LinearSegmentedColormap.from_list("ember", ["#15110e", "#6f675f", "#f2b48e", "#e07a3f", "#c9561c"])
    fig, axes = plt.subplots(1, 3, figsize=(8, 4.5), dpi=200)
    fig.patch.set_facecolor("#15110e")
    panels = (
        ("pub-lensswap", "lens swapped, same weights"),
        ("pub-baselens", "weights changed, same lens"),
        ("pub", "both: base vs post-trained"),
    )
    for ax, (key, lab) in zip(axes, panels, strict=True):
        g = np.array(G["cover"][key])
        ax.imshow(
            np.clip(g, 0, 1.2), aspect="auto", cmap=cmap, vmin=0, vmax=1.1, origin="lower", interpolation="nearest"
        )
        ax.set_title(lab, color="#e8ecec", fontsize=9)
        ax.set_xticks([])
        ticks = ["L0", "L17", "L35"] if key == "pub-lensswap" else ["", "", ""]
        ax.set_yticks([0, 17, 35], ticks, color="#a39a90", fontsize=8)
        ax.tick_params(length=0)
        ax.spines[:].set_visible(False)
    fig.text(0.5, 0.02, "each row: one layer's 12,288 neurons, sorted by how much a lens swap alone moves them; "
             "brighter = read moved more", color="#a39a90", ha="center", fontsize=7.5)  # fmt: skip
    out = FIG / "hero.jpg"
    fig.savefig(out, facecolor=fig.get_facecolor(), dpi=200, bbox_inches=None)
    plt.close(fig)
    im = Image.open(out).convert("RGB").resize((1600, 900))
    im.save(out, quality=88)
    im.resize((1200, 675)).crop((0, 22, 1200, 652)).save(FIG / "og.jpg", quality=86)
    thumbs = HERE.parent / "shared" / "thumbs"
    if thumbs.exists():
        im.resize((720, 405)).resize((720, 480)).save(thumbs / "posttrain-diff.jpg", quality=86)
    print("hero.jpg, og.jpg: cover from results/unit_grids.json")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    S, A, X, G = j("summary.json"), j("agreement.json"), j("acts.json"), j("unit_grids.json")
    layers(S)
    sameunits(G, S, A)
    agreement(A)
    register(X)
    readsacts(S, X)
    hero(G)


if __name__ == "__main__":
    main()
