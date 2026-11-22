"""Every figure in the Episode 2 note, drawn from results/ (never by hand). Prints the numbers it plots.

    python make_figures.py      -> report/figures/*.png, report/figures/hero.jpg, ../shared/thumbs/sounds-like-me-ep2.jpg

Fig 1  the no-training ladder: strict voice position per rung, as scored and cut to 80 words, with Episode 1's best
Fig 2  honesty per arm: share of drafts with an unsupported claim (frozen Qwen3-32B extractor, lower bounds)
Fig 3  persona tests: voice position against specificity (k0 prompt, steering at the Part C layer and strength)
Fig 4  per-step ablation: the harness minus each step (paired over authors)
Fig 5  assistant-direction projection of Episode 1's drafts against their voice position
Cover  the ladder in one panel: hero.jpg (build_meta.py crops the og image from it) and the 720x480 card thumb
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
R = HERE / "results"
FIG = HERE / "report" / "figures"
INK, SOFT, FAINT, RULE = "#1d272a", "#586468", "#8a9497", "#e3e7e8"
ACCENT, CONTROL = "#c9561c", "#4a86e8"
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
LABEL = {
    "R1_prompt": "Rung 1 prompt only (8 samples)",
    "R1b_bon": "R1b best of 20, same prompt",
    "R2_harness": "Rung 2 harness",
    "R3_voice": "Rung 3 harness + voice vector",
    "xR3_s0.3": "Rung 3 at strength 0.3 (exploratory)",
    "R4_persona": "Rung 4 harness − assistant direction",
    "R5_both": "Rung 5 harness + both",
    "R6_rand0": "Rung 6 harness + random (seed 0)",
    "R6_rand1": "Rung 6 harness + random (seed 1)",
    "G2_gemini_harness": "Gemini 4 Carbon in the harness",
    "ep1-gemini-edits": "Ep 1 best honest: Gemini, 3 more samples*",
    "ep1-llamabase-k3": "Ep 1 best outright: Llama-3.1-8B Base",
    "ep1-gemini-k3": "Ep 1: Gemini, 3 samples",
    "ep1-sonnet-note": "Ep 1: Claude Sonnet 5.5, style note",
    "ep1-qwen8b-k8": "Ep 1: Qwen3-8B, 8 samples",
    "ep1-qwen8b-k0": "Ep 1: Qwen3-8B, no samples",
    "none": "no steering",
    "away": "− assistant direction",
    "toward": "+ assistant direction",
    "rand0": "random 0",
    "rand1": "random 1",
    "rand2": "random 2",
    "card": "character card",
    "own": "author's voice vector",
    "card_own": "card + voice vector",
    "abl_noretrieval": "random samples, not retrieved",
    "abl_noprofile": "no style profile",
    "abl_norevise": "no critique and revision",
    "abl_norerank": "no rerank (first draft)",
}
STEER = {"R3_voice", "xR3_s0.3", "R4_persona", "R5_both", "away", "own", "card_own"}
CTRL = {"R6_rand0", "R6_rand1", "rand0", "rand1", "rand2", "toward"}


def color(a):
    if a in STEER:
        return ACCENT
    if a in CTRL:
        return CONTROL
    return INK if a.startswith(("R", "G2")) else SOFT


def floor_ceiling(ax):
    for x, t in ((0, "a different human (0)"), (1, "the same human (1)")):
        ax.axvline(x, color=FAINT, lw=1, ls="--")
        ax.text(x, 1.01, t, transform=ax.get_xaxis_transform(), ha="center", fontsize=8, color=SOFT)


def present(s, names, key="position"):
    return [a for a in names if a in s["arms"] and s["arms"][a][key][0] is not None]


def fig1(s):
    order = present(
        s,
        [
            "R1_prompt",
            "R1b_bon",
            "R2_harness",
            "R3_voice",
            "xR3_s0.3",
            "R4_persona",
            "R5_both",
            "R6_rand0",
            "R6_rand1",
            "G2_gemini_harness",
            "ep1-gemini-edits",
            "ep1-llamabase-k3",
        ],
    )
    fig, ax = plt.subplots(figsize=(8.4, 0.42 * len(order) + 1.4))
    for i, a in enumerate(order):
        r = s["arms"][a]
        v, lo, hi = r["position"]
        ax.plot([lo, hi], [i, i], color=FAINT, lw=1.4)
        ax.scatter([v], [i], s=40, color=color(a), zorder=3, label="as scored" if i == 0 else None)
        v8 = r["position80"][0]
        ax.scatter(
            [v8],
            [i],
            s=34,
            facecolor="white",
            edgecolor=color(a),
            zorder=3,
            label="cut to 80 words" if i == 0 else None,
        )
        print(f"fig1 {a:20s} {v:+.3f} [{lo:+.3f}, {hi:+.3f}]  80w {v8:+.3f}")
    floor_ceiling(ax)
    ax.set_yticks(range(len(order)), [LABEL[a] for a in order])
    ax.invert_yaxis()
    ax.set_xlim(-1.0, 1.05)
    ax.set_xlabel("Voice position (strict LUAR, 4-draft episodes; 30 held-out authors, 95% bootstrap)")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_ladder.png", dpi=180)
    plt.close(fig)


def fig2(s):
    order = present(
        s,
        [
            "R1_prompt",
            "R1b_bon",
            "R2_harness",
            "R3_voice",
            "xR3_s0.3",
            "R4_persona",
            "R5_both",
            "R6_rand0",
            "R6_rand1",
            "G2_gemini_harness",
            "none",
            "away",
            "own",
            "card",
            "card_own",
            "rand0",
            "rand1",
            "rand2",
            "ep1-gemini-edits",
            "ep1-llamabase-k3",
        ],
        "unsupported",
    )
    fig, ax = plt.subplots(figsize=(8.4, 0.36 * len(order) + 1.4))
    for i, a in enumerate(order):
        v, lo, hi = s["arms"][a]["unsupported"]
        ax.plot([lo, hi], [i, i], color=FAINT, lw=1.4)
        ax.scatter([v], [i], s=40, color=color(a), zorder=3)
        print(f"fig2 {a:20s} unsupported {v:.3f} [{lo:.3f}, {hi:.3f}]")
    r1 = s["arms"].get("R1_prompt", {}).get("unsupported", [None])[0]
    if r1 is not None:
        ax.axvline(r1, color=FAINT, ls="--", lw=1)
        ax.axvline(r1 + 0.05, color=FAINT, ls=":", lw=1)
        ax.text(
            r1 + 0.05,
            1.01,
            "Rung 1 + 5 points (the gate)",
            transform=ax.get_xaxis_transform(),
            ha="center",
            fontsize=8,
            color=SOFT,
        )
    ax.set_yticks(range(len(order)), [LABEL[a] for a in order])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.7)
    ax.set_xlabel("Share of drafts with an unsupported claim (frozen extractor; lower bounds; 95% bootstrap)")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_honesty.png", dpi=180)
    plt.close(fig)


def fig3(s):
    arms = present(s, ["none", "away", "toward", "rand0", "rand1", "rand2", "card", "own", "card_own"])
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for a in arms:
        r = s["arms"][a]
        x, y = r["position"][0], r["spec"][0]
        ax.scatter([x], [y], s=60, color=color(a), zorder=3, marker="s" if a.startswith("card") else "o")
        ax.annotate(
            f"{LABEL[a]} ({r['loops']:.0%} loop)",
            (x, y),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8.5,
            color=INK,
        )
        print(f"fig3 {a:10s} position {x:+.3f} spec {y:+.4f} loops {r['loops']:.2f}")
    ax.axhline(0, color=FAINT, lw=1)
    ax.axhline(0.172, color=FAINT, lw=1, ls="--")
    ax.text(-0.95, 0.172, "authors' own unseen emails (0.172)", va="bottom", fontsize=8, color=SOFT)
    ax.set_xlim(-1.0, 0.5)
    ax.set_ylim(-0.02, 0.19)
    ax.set_xlabel("Voice position (strict; 0 = a different human)")
    ax.set_ylabel("Specificity (this author minus the other 29)")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_persona.png", dpi=180)
    plt.close(fig)


def fig4(s):
    c = s["contrasts"]
    abl = [
        a
        for a in ["abl_norerank", "abl_noretrieval", "abl_noprofile", "abl_norevise"]
        if c.get(f"rung2-{a}", [None])[0] is not None
    ]
    fig, ax = plt.subplots(figsize=(7.6, 0.5 * len(abl) + 1.4))
    for i, a in enumerate(abl):
        v, lo, hi, _n = c[f"rung2-{a}"]
        ax.plot([lo, hi], [i, i], color=FAINT, lw=1.4)
        ax.scatter([v], [i], s=44, color=INK, zorder=3)
        print(f"fig4 Rung 2 - {a:16s} {v:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    ax.axvline(0, color=FAINT, lw=1)
    ax.set_yticks(range(len(abl)), [LABEL[a] for a in abl])
    ax.invert_yaxis()
    ax.set_xlabel("Harness minus the harness without this step, strict position (paired, 30 authors, 95% bootstrap)")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_ablation.png", dpi=180)
    plt.close(fig)


def fig5():
    p = json.loads((R / "projection.json").read_text())
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.scatter([q["p"] for q in p["points"]], [q["y"] for q in p["points"]], s=6, color=SOFT, alpha=0.45, linewidths=0)
    ax.axhline(0, color=FAINT, ls="--", lw=1)
    ax.set_xlabel("Projection on the assistant direction (mean over the episode's 4 drafts)")
    ax.set_ylabel("Voice position (strict)")
    ax.set_title(
        f"Spearman {p['spearman_projection_vs_position']:+.2f}; partial on detector and length "
        f"{p['partial_on_detector_and_words']:+.2f} ({p['n_episodes']} episodes)",
        fontsize=9.5,
        color=INK,
    )
    print(
        f"fig5 n={p['n_episodes']} rho={p['spearman_projection_vs_position']} partial={p['partial_on_detector_and_words']}"
    )
    fig.tight_layout()
    fig.savefig(FIG / "fig5_projection.png", dpi=180)
    plt.close(fig)


def cover(s):
    short = {
        "R1_prompt": "Qwen3-8B, prompt only",
        "R1b_bon": "best of 20 drafts",
        "R2_harness": "the harness",
        "R3_voice": "harness + voice vector",
        "R6_rand0": "harness + random vector",
        "ep1-gemini-edits": "Episode 1's best (Gemini)",
    }
    order = present(s, list(short))
    for name, size in (("hero.jpg", (12, 6.75)),):
        fig = plt.figure(figsize=size, facecolor="#faf6ef")
        ax = fig.add_axes([0.30, 0.16, 0.64, 0.60], facecolor="#faf6ef")
        for i, a in enumerate(order):
            v, lo, hi = s["arms"][a]["position"]
            ax.plot([lo, hi], [i, i], color=FAINT, lw=2.4)
            ax.scatter([v], [i], s=150, color=color(a) if a != "ep1-gemini-edits" else SOFT, zorder=3)
            ax.text(-1.05, i, short[a], ha="right", va="center", fontsize=15, color=INK)
        for x, t in ((0, "a different human"), (1, "the same human")):
            ax.axvline(x, color=FAINT, lw=1.2, ls="--")
            ax.text(x, -0.9, t, ha="center", fontsize=12, color=SOFT)
        ax.set_xlim(-1.0, 1.05)
        ax.set_ylim(len(order) - 0.4, -1.2)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.tick_params(labelsize=12)
        fig.text(0.04, 0.90, "SOUNDS LIKE ME · EPISODE 2", fontsize=13, color=ACCENT, weight="bold")
        fig.text(0.04, 0.83, "Beating every model without training", fontsize=26, color=INK, weight="bold")
        fig.text(
            0.04,
            0.05,
            "Voice position on 30 held-out public writers · Qwen3-8B on one GPU · nothing trained",
            fontsize=12,
            color=SOFT,
        )
        fig.savefig(FIG / name, dpi=100, facecolor=fig.get_facecolor(), pil_kwargs={"quality": 88})
        plt.close(fig)
    from PIL import Image

    hero = Image.open(FIG / "hero.jpg").convert("RGB")
    w = round(hero.height * 720 / 480)
    thumb = hero.crop(((hero.width - w) // 2, 0, (hero.width - w) // 2 + w, hero.height)).resize(
        (720, 480), Image.Resampling.LANCZOS
    )
    thumb.save(HERE.parent / "shared" / "thumbs" / "sounds-like-me-ep2.jpg", quality=84, optimize=True)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    s = json.loads((R / "summary.json").read_text())
    fig1(s)
    fig2(s)
    fig3(s)
    fig4(s)
    fig5()
    cover(s)


if __name__ == "__main__":
    main()
