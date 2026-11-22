"""Every figure in the note, drawn from results/ (never by hand). Prints the numbers it plots.

    python make_figures.py      -> report/figures/fig1..fig4 *.png

Fig 1  leaderboard: strict voice position per model, no samples vs 8 samples vs style note (95% bootstrap over authors)
Fig 2  position against specificity, every contestant and arm (more human vs more like THIS person)
Fig 3  steering: own vector vs controls, raw / loop-free / length-matched (robustness.py)
Fig 4  every signal for every model: strict position heatmap
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import HERE
from export_pages import ARM, LABEL

R = HERE / "results"
FIG = HERE / "report" / "figures"
INK, SOFT, FAINT, RULE = "#1d272a", "#586468", "#8a9497", "#e3e7e8"
ACCENT, CONTROL = "#c9561c", "#4a86e8"
MODELS = [
    "gemini-4-carbon",
    "Llama-3.1-8B",
    "Qwen3-8B-Base",
    "gemma-3-27b-it",
    "Llama-3.1-8B-Instruct",
    "claude-sonnet-5.5",
    "Qwen3-32B",
    "Qwen3-8B",
]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": FAINT,
        "axes.labelcolor": INK,
        "xtick.color": SOFT,
        "ytick.color": SOFT,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def load():
    rows = json.loads((R / "leaderboard.json").read_text())
    return {(r["model"], r["arm"]): r for r in rows}, json.loads((R / "specificity.json").read_text())


def floor_ceiling(ax):
    ax.axvline(0, color=FAINT, lw=1, ls="--")
    ax.axvline(1, color=FAINT, lw=1, ls="--")
    ax.text(0, 1.01, "a different human (0)", transform=ax.get_xaxis_transform(), ha="center", fontsize=8, color=SOFT)
    ax.text(1, 1.01, "the same human (1)", transform=ax.get_xaxis_transform(), ha="center", fontsize=8, color=SOFT)


def fig1(T):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    arms = [("k0", "no samples", "o", "white"), ("k8_rand", "8 samples", "o", INK), ("note", "style note", "s", SOFT)]
    for i, m in enumerate(MODELS):
        for j, (arm, lab, mk, fc) in enumerate(arms):
            r = T.get((m, arm))
            if not r or r["voice_strict"][0] is None:
                continue
            y = i + (j - 1) * 0.22
            v, lo, hi = r["voice_strict"]
            ax.plot([lo, hi], [y, y], color=FAINT, lw=1.2)
            ax.scatter([v], [y], marker=mk, s=34, facecolor=fc, edgecolor=INK, zorder=3, label=lab if i == 0 else None)
            print(f"fig1 {m:24s} {arm:8s} {v:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    floor_ceiling(ax)
    ax.set_yticks(range(len(MODELS)), [LABEL[m] for m in MODELS])
    ax.invert_yaxis()
    ax.set_xlim(-1.3, 1.1)
    ax.set_xlabel("Voice position (strict LUAR, 4-draft episodes; 30 held-out authors, 95% bootstrap)")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_leaderboard.png", dpi=180)


def fig2(T, S):
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for (m, arm), r in T.items():
        s = S.get(f"{m}|{arm}")
        if not s or r["voice_strict"][0] is None:
            continue
        x, y = r["voice_strict"][0], s["mean"]
        if m.startswith("partc"):
            c, z = (
                (
                    ACCENT
                    if arm in ("own", "k8_own")
                    else CONTROL
                    if arm.startswith("rand") or arm == "wrong"
                    else FAINT
                ),
                4,
            )
        elif m.startswith("humanizer"):
            c, z = INK, 4
        else:
            c, z = (SOFT if arm == "k0" else "#b9c2c4"), 2
        ax.scatter([x], [y], s=26, color=c, zorder=z, edgecolor="white", lw=0.5)
        if (
            m.startswith("humanizer-dipper")
            or (m == "gemini-4-carbon" and arm == "k8_rand")
            or (m.startswith("partc") and arm in ("own", "wrong"))
        ):
            lab = {"own": "own voice vector", "wrong": "another author's vector"}.get(
                arm, LABEL[m].split(" (")[0] + ("" if "DIPPER" in LABEL[m] else ", 8 samples")
            )
            ax.annotate(
                lab, (x, y), xytext=(6, -11 if arm == "wrong" else 4), textcoords="offset points", fontsize=8, color=INK
            )
            print(f"fig2 {m:34s} {arm:8s} position {x:+.3f} specificity {y:+.4f}")
    hs = S["human_ceiling"]["mean"]
    ax.axhline(hs, color=FAINT, lw=1, ls="--")
    ax.text(-1.25, hs + 0.004, f"the author's own unseen emails: {hs:.3f}", fontsize=8, color=SOFT)
    ax.axhline(0, color=RULE, lw=1)
    floor_ceiling(ax)
    ax.set_xlim(-1.3, 1.1)
    ax.set_xlabel("Voice position (strict; 0 = a different human, 1 = the same human)")
    ax.set_ylabel("Specificity (closer to this author\nthan to the other 29)")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_specificity.png", dpi=180)


def fig3():
    rob = json.loads((R / "robustness.json").read_text())["partc"]
    P = {r["arm"]: r for r in json.loads((R / "leaderboard.json").read_text()) if r["model"] == "partc-Qwen3-8B"}
    arms = ["none", "rand0", "rand1", "rand2", "wrong", "own", "k8", "k8_own"]
    col = {
        "none": FAINT,
        "k8": FAINT,
        "wrong": CONTROL,
        "rand0": CONTROL,
        "rand1": CONTROL,
        "rand2": CONTROL,
        "own": ACCENT,
        "k8_own": ACCENT,
    }
    variants = [
        ("as scored", {a: P[a]["voice_strict"][0] for a in arms}),
        ("looping drafts removed", {a: rob["loop_free"]["arms"][a]["mean"] for a in arms}),
        ("every draft cut to 80 words", {a: rob["length_matched"]["arms"][a]["mean"] for a in arms}),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True)
    for ax, (title, vals) in zip(axs, variants, strict=False):
        ax.barh(range(len(arms)), [vals[a] for a in arms], color=[col[a] for a in arms])
        ax.axvline(0, color=FAINT, lw=1, ls="--")
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_xlim(-0.8, 0.2)
        print(f"fig3 {title:28s} " + " ".join(f"{a}={vals[a]:+.3f}" for a in arms))
    axs[0].set_yticks(range(len(arms)), [ARM[a] for a in arms])
    axs[0].invert_yaxis()
    axs[1].set_xlabel("Voice position (strict), Qwen3-8B, 30 held-out authors")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_steering.png", dpi=180)


def fig4(T):
    arms = ["k0", "k3_rand", "k3_match", "k8_rand", "note", "edits"]
    M = np.full((len(MODELS), len(arms)), np.nan)
    for i, m in enumerate(MODELS):
        for j, a in enumerate(arms):
            r = T.get((m, a))
            if r and r["voice_strict"][0] is not None:
                M[i, j] = r["voice_strict"][0]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    im = ax.imshow(M, cmap="Greys_r", vmin=-1.0, vmax=0.1, aspect="auto")
    for i in range(len(MODELS)):
        for j in range(len(arms)):
            if not np.isnan(M[i, j]):
                ax.text(
                    j,
                    i,
                    f"{M[i, j]:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="white" if M[i, j] < -0.5 else INK,
                )
            else:
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=8, color=SOFT)
    short = {
        "k0": "no\nsamples",
        "k3_rand": "3 random\nsamples",
        "k3_match": "3 matched\nsamples",
        "k8_rand": "8 random\nsamples",
        "note": "style\nnote",
        "edits": "3 more\nsamples*",
    }
    ax.set_xticks(range(len(arms)), [short[a] for a in arms], fontsize=8.5)
    ax.set_yticks(range(len(MODELS)), [LABEL[m] for m in MODELS])
    fig.colorbar(im, ax=ax, fraction=0.03, label="voice position (strict)")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_signals.png", dpi=180)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    T, S = load()
    fig1(T)
    fig2(T, S)
    fig3()
    fig4(T)


if __name__ == "__main__":
    main()
