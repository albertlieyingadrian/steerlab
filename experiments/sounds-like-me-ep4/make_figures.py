"""Figures for the Episode 4 note, drawn from results/summary.json only. Prints every number it plots.

python make_figures.py
"""

import json
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common4 import HERE, RES

OUT = HERE / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
S = json.loads((RES / "summary.json").read_text())
H1, H2, H3 = S["h1"]["heldout"], S["h2"]["heldout"], S["h3"]["heldout"]

INK, SOFT, RULE, GREY = "#1f1d1a", "#6b665e", "#e3ddd3", "#9a948a"
EMBER, PANEL = "#c2541b", "#4a5260"  # panel: dark slate (a comparison arm, not a control)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": RULE,
        "axes.labelcolor": INK,
        "xtick.color": SOFT,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# ruler key -> (label, colour). Ember: the series' rulers; dark slate: the LLM panel (comparison arm); grey: baselines
RULERS = [
    ("luar", "LUAR voice similarity", EMBER),
    ("spec", "LUAR specificity", EMBER),
    ("pref", "Preference model (Ep 2)", EMBER),
    ("detector_neg", "Detector", EMBER),
    ("panel", "LLM panel (3 open models)", PANEL),
    ("tfidf_char", "Character n-grams", GREY),
    ("length", "Length", GREY),
]


def cell(name, metric):
    """[point, low, high] or None for a ruler under one metric (H3 scores point toward 'AI-like')."""
    if metric == "h1_r":
        return H1["rm_r"].get(name)
    if metric == "h1_up":
        return H1["pairwise_up"].get(name)
    if metric == "h2":
        return None if name in ("detector_neg", "spec") else H2["acc"].get(name)
    if metric == "h3":
        k = {"pref": "pref_neg", "detector_neg": "detector"}.get(name, name)
        return None if name in ("luar", "spec", "tfidf_char") else H3["auroc_majority"].get(k)
    return None


PANELS = [
    ("h1_r", "H1 · own voice: within-person r\nwith the writer's 1-5 rating", 0.0, (-0.1, 0.6)),
    ("h1_up", "H1 · the writer rated the edit higher:\nshare where the ruler agrees", 0.5, (0.3, 1.0)),
    ("h2", "H2 · fiction style triplets:\nagreement with the crowd majority", 0.5, (0.4, 0.75)),
    ("h3", "H3 · news articles: AUROC against\nfive experts' human/AI vote", 0.5, (0.2, 1.02)),
]


def fig1(path, figsize=(12, 4.6), dpi=200, title=None):
    fig, axes = plt.subplots(1, 4, figsize=figsize, sharey=True)
    ys = np.arange(len(RULERS))[::-1]
    for ax, (m, lab, chance, xl) in zip(axes, PANELS):
        ax.axvline(chance, color=GREY, lw=1, ls=(0, (3, 3)))
        for y, (k, _, col) in zip(ys, RULERS):
            c = cell(k, m)
            if not c:
                ax.text(xl[0] + 0.01, y, "n/a", va="center", fontsize=8, color=GREY)
                continue
            ax.plot([c[1], c[2]], [y, y], color=col, lw=2, alpha=0.55, solid_capstyle="butt")
            ax.plot(c[0], y, "o", color=col, ms=6)
            if path.name == "fig1_agreement.png":
                print(f"fig1 {m:6s} {k:14s} {c}")
        ax.set_xlim(*xl)
        ax.set_title(lab, fontsize=9.5, color=INK, loc="left")
        ax.grid(axis="x", color=RULE, lw=0.6)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([r[1] for r in RULERS])
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=15, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def fig2():
    keys = [r for r in RULERS if r[0] in H1["pairwise_up"]]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    groups = [
        ("pairwise_up", f"Writer rated the edit higher (n={H1['n_up']})"),
        ("prefers_edit_equal", f"Rated both the same (n={H1['n_equal']})"),
        ("prefers_edit_down", f"Rated the draft higher (n={H1['n_down']})"),
    ]
    w = 0.26
    x = np.arange(len(keys))
    for gi, (g, lab) in enumerate(groups):
        for i, (k, _, col) in enumerate(keys):
            c = H1[g][k]
            ax.bar(
                x[i] + (gi - 1) * w,
                c[0],
                w * 0.9,
                color=col,
                alpha=[1.0, 0.55, 0.25][gi],
                label=lab if i == 0 else None,
            )
            if len(c) == 3:
                ax.plot([x[i] + (gi - 1) * w] * 2, c[1:], color=INK, lw=0.8)
            print(f"fig2 {g:20s} {k:14s} {c}")
    ax.axhline(0.5, color=GREY, lw=1, ls=(0, (3, 3)))
    ax.set_xticks(x)
    ax.set_xticklabels([textwrap.fill(r[1], 14) for r in keys], fontsize=8.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of tasks where the ruler\nscores the post-edit higher")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0, 1.16), ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_equal.png", dpi=200, facecolor="white")
    plt.close(fig)


def fig3():
    keys = [
        ("detector", "Detector", EMBER),
        ("pref_neg", "Preference model (reversed)", EMBER),
        ("panel", "LLM panel", PANEL),
        ("length", "Length", GREY),
    ]
    keys = [k for k in keys if k[0] in H3["spearman_signed_within_human"]]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2), sharey=True)
    for ax, (fld, lab) in zip(
        axes,
        (
            ("spearman_signed_within_human", "Human-written articles"),
            ("spearman_signed_within_ai", "AI-written articles"),
        ),
    ):
        ax.axvline(0, color=GREY, lw=1, ls=(0, (3, 3)))
        for y, (k, _, col) in enumerate(keys[::-1]):
            c = H3[fld][k]
            ax.plot(c[1:], [y, y], color=col, lw=2, alpha=0.55)
            ax.plot(c[0], y, "o", color=col)
            print(f"fig3 {fld:30s} {k:10s} {c}")
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels([k[1] for k in keys[::-1]])
        ax.set_xlim(-0.65, 0.65)
        ax.set_title(lab, fontsize=9.5, loc="left")
        ax.grid(axis="x", color=RULE, lw=0.6)
        ax.set_xlabel("Spearman with experts' signed vote")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_h3_within.png", dpi=200, facecolor="white")
    plt.close(fig)


def fig4():
    A = H1["ruler_agreement"]
    names = A["names"]
    lab = {r[0]: r[1] for r in RULERS}
    M = np.array(
        [[A["pairs"][f"{a}|{b}"] if A["pairs"][f"{a}|{b}"] is not None else np.nan for b in names] for a in names],
        float,
    )
    M[np.triu_indices(len(names))] = np.nan  # lower triangle only, no diagonal
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    im = ax.imshow(M[1:, :-1], vmin=0.5, vmax=1.0, cmap="Greys")
    for i in range(1, len(names)):
        for j in range(i):
            ax.text(
                j,
                i - 1,
                f"{M[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="white" if M[i, j] > 0.8 else INK,
            )
    ax.set_xticks(range(len(names) - 1))
    ax.set_yticks(range(len(names) - 1))
    ax.set_xticklabels([lab.get(n, n) for n in names[:-1]], rotation=40, ha="right", fontsize=8.5)
    ax.set_yticklabels([lab.get(n, n) for n in names[1:]], fontsize=8.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.04, label="share of tasks where two rulers pick the same text")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_agree.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("fig4", names, M.round(2).tolist())


def hero():
    """Cover: one panel (H1, own voice) at a size that reads as a card."""
    from PIL import Image

    fig, ax = plt.subplots(figsize=(16, 9), dpi=75)
    rows = [r for r in RULERS if H1["rm_r"].get(r[0])]
    ys = np.arange(len(rows))[::-1]
    ax.axvline(0, color=GREY, lw=1.5, ls=(0, (4, 4)))
    for y, (k, _, col) in zip(ys, rows):
        c = H1["rm_r"][k]
        ax.plot(c[1:], [y, y], color=col, lw=6, alpha=0.5, solid_capstyle="butt")
        ax.plot(c[0], y, "o", color=col, ms=16)
        ax.text(c[2] + 0.012, y, f"{c[0]:.2f}", va="center", fontsize=20, color=SOFT)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[1] for r in rows], fontsize=22)
    ax.set_xlim(-0.12, 0.62)
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=RULE, lw=1)
    ax.set_xlabel("within-writer correlation with the writer's own 1-5 rating of their voice (61 writers)", fontsize=18)
    fig.suptitle(
        "Does the voice score agree with people?", x=0.02, y=0.97, ha="left", fontsize=34, color=INK, fontweight="bold"
    )
    fig.text(
        0.02,
        0.875,
        "Sounds Like Me · Episode 4 · 81 writers rated how well texts capture their own style",
        fontsize=19,
        color=SOFT,
    )
    fig.subplots_adjust(left=0.27, right=0.97, top=0.82, bottom=0.12)
    fig.savefig(OUT / "_hero.png", dpi=75, facecolor="white")
    plt.close(fig)
    im = Image.open(OUT / "_hero.png").convert("RGB")
    im.resize((1200, 675)).save(OUT / "hero.jpg", quality=90)
    og = Image.new("RGB", (1200, 630), "white")
    og.paste(im.resize((1120, 630)), (40, 0))
    og.save(OUT / "og.jpg", quality=90)
    # The 720x480 card thumb: a centre crop of the cover, as Episode 2 makes it.
    w = round(im.height * 720 / 480)
    thumb = im.crop(((im.width - w) // 2, 0, (im.width - w) // 2 + w, im.height)).resize(
        (720, 480), Image.Resampling.LANCZOS
    )
    thumb.save(HERE.parent / "shared" / "thumbs" / "sounds-like-me-ep4.jpg", quality=84, optimize=True)
    (OUT / "_hero.png").unlink()


if __name__ == "__main__":
    fig1(OUT / "fig1_agreement.png")
    fig2()
    fig3()
    fig4()
    hero()
