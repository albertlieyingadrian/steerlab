"""Figures for the Inkling add-on note, drawn from results/board.json only. Prints every number it plots.

python make_figures.py   -> report/figures/fig1_board.png, fig2_arms.png, hero.jpg, og.jpg,
                            ../shared/thumbs/sounds-like-me-inkling.jpg
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).parent
OUT = HERE / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
B = json.loads((HERE / "results" / "board.json").read_text())
C, REF = B["contestants"], B["reference"]
INKM = "inkling-2-mini"
ARMS = ("k0", "k3_rand", "k3_match", "k8_rand", "note", "edits")
ARM_LABEL = {
    "k0": "no samples",
    "k3_rand": "3 samples",
    "k3_match": "3 matched",
    "k8_rand": "8 samples",
    "note": "style note",
    "edits": "3 more samples*",
}

INK, SOFT, RULE, GREY = "#1f1d1a", "#6b665e", "#e3ddd3", "#9a948a"
EMBER = "#c2541b"
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


def board_axes(ax, big=False):
    order = B["ranked"]
    ys = np.arange(len(order))[::-1]
    fs = 20 if big else 9
    ax.axvline(0, color=GREY, lw=1.5 if big else 1, ls=(0, (4, 4)))
    h = REF["ep2_harness_voice"][0]
    ax.axvline(h, color=SOFT, lw=1.2 if big else 0.8, ls=":")
    ax.text(h, len(order) - 0.35, " Episode 2 harness + voice vector", fontsize=fs * 0.8, color=SOFT, va="bottom")
    ax.text(0, len(order) - 0.35, "a stranger ", fontsize=fs * 0.8, color=SOFT, va="bottom", ha="right")
    for y, m in zip(ys, order, strict=True):
        c = C[m]
        col = EMBER if m == INKM else GREY
        lo, hi = c["best"][1], c["best"][2]
        ax.plot([lo, hi], [y, y], color=col, lw=6 if big else 3, alpha=0.45, solid_capstyle="butt")
        ax.plot(c["best"][0], y, "o", color=col, ms=15 if big else 6)
        lab = f"{'0.00' if abs(c['best'][0]) < 0.005 else format(c['best'][0], '+.2f')}  ({ARM_LABEL[c['best_arm']]})"
        ax.text(hi + 0.02, y, lab, va="center", fontsize=fs * 0.8, color=SOFT)
        print("board", c["rank"], c["label"], c["best_arm"], c["best"])
    ax.set_yticks(ys)
    ax.set_yticklabels([C[m]["label"] for m in order], fontsize=fs)
    for t, m in zip(ax.get_yticklabels(), order, strict=True):
        if m == INKM:
            t.set_color(EMBER)
            t.set_fontweight("bold")
    ax.set_xlim(-1.0, 0.75)
    ax.set_ylim(-0.7, len(order) + 0.3)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=fs * 0.8)
    ax.grid(axis="x", color=RULE, lw=1)
    ax.set_xlabel(
        "best signal's strict voice position (0 = a different human, 1 = the same human), 30 held-out writers",
        fontsize=fs * 0.85,
    )


def fig1():
    fig, ax = plt.subplots(figsize=(8, 4.2))
    board_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_board.png", dpi=200, facecolor="white")
    plt.close(fig)


def fig2():
    """Inkling per arm against the two frontier models and the mean of the six other contestants."""
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(ARMS))
    others = [m for m in B["ranked"] if m not in (INKM, "gemini-4-carbon", "claude-sonnet-5.5")]
    series = [
        ("gemini-4-carbon", "Gemini 4 Carbon", INK, "s", -0.18),
        ("claude-sonnet-5.5", "Claude Sonnet 5.5", SOFT, "D", -0.06),
        (INKM, "Inkling 2 Mini", EMBER, "o", 0.06),
    ]
    for m, lab, col, mk, dx in series:
        a = C[m]["arms"]
        xs = [i + dx for i, k in enumerate(ARMS) if k in a]
        v = [a[k]["voice_strict"] for k in ARMS if k in a]
        ax.errorbar(
            xs,
            [t[0] for t in v],
            yerr=[[t[0] - t[1] for t in v], [t[2] - t[0] for t in v]],
            fmt=mk,
            color=col,
            ms=6,
            capsize=0,
            lw=1.4,
            label=lab,
        )
        print("arms", lab, [round(t[0], 3) for t in v])
    om = []
    for k in ARMS:
        vals = [C[m]["arms"][k]["voice_strict"][0] for m in others if k in C[m]["arms"]]
        om.append(np.mean(vals) if vals else np.nan)
    ax.plot(x + 0.18, om, "_", color=GREY, ms=14, mew=2, label="mean of the other six (four for the last two)")
    print("arms other-six mean", [round(float(v), 3) for v in om])
    ax.axhline(0, color=GREY, lw=1, ls=(0, (4, 4)))
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABEL[k] for k in ARMS])
    ax.set_ylabel("strict voice position")
    ax.grid(axis="y", color=RULE, lw=1)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(OUT / "fig2_arms.png", dpi=200, facecolor="white")
    plt.close(fig)


def hero():
    fig, ax = plt.subplots(figsize=(16, 9), dpi=75)
    board_axes(ax, big=True)
    fig.suptitle(
        "Can Inkling 2 Mini sound like you?",
        x=0.02,
        y=0.97,
        ha="left",
        fontsize=32,
        color=INK,
        fontweight="bold",
    )
    sub = "Nine contestants, each at its best signal, 30 held-out public writers x 4 sales briefs"
    fig.text(0.02, 0.875, sub, fontsize=18, color=SOFT)
    fig.subplots_adjust(left=0.24, right=0.97, top=0.8, bottom=0.12)
    fig.savefig(OUT / "_hero.png", dpi=75, facecolor="white")
    plt.close(fig)
    im = Image.open(OUT / "_hero.png").convert("RGB")
    im.resize((1200, 675)).save(OUT / "hero.jpg", quality=90)
    og = Image.new("RGB", (1200, 630), "white")
    og.paste(im.resize((1120, 630)), (40, 0))
    og.save(OUT / "og.jpg", quality=90)
    thumb()
    (OUT / "_hero.png").unlink()


def thumb():
    """Research-page card (720 x 480): the board alone, large type, no title (the card carries the title)."""
    fig, ax = plt.subplots(figsize=(9, 6), dpi=80)
    board_axes(ax, big=True)
    ax.set_xlabel("best-signal voice position (0 = a stranger)", fontsize=16)
    fig.subplots_adjust(left=0.36, right=0.97, top=0.93, bottom=0.12)
    fig.savefig(HERE.parent / "shared" / "thumbs" / "sounds-like-me-inkling.jpg", dpi=80, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    hero()
