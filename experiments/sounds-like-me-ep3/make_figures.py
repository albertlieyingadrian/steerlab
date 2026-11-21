"""Every figure in the Episode 3 note, drawn from results/ (never by hand). Prints the numbers it plots.

    python make_figures.py  -> report/figures/*.png, report/figures/hero.jpg, ../shared/thumbs/sounds-like-me-ep3.jpg

Fig 1  the training ladder on held-out writers: strict voice position per arm, with Episode 2's best no-training arms
Fig 2  invented claims per arm (frozen Qwen3-32B extractor, lower bounds): SFT doubles them
Fig 3  the hack hunt: GRPO training reward against the unseen rulers on dev writers, per checkpoint
Fig 4  data formats: position and specificity for raw samples, style notes, edit pairs, and the shuffled control
Cover  the hack hunt in one panel (hero art)
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
R = HERE / "results"
FIG = HERE / "report" / "figures"
INK, SOFT, FAINT = "#1d272a", "#586468", "#8a9497"
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
    "base_raw": "Qwen3-8B, prompt (4 samples)",
    "sft_raw": "SFT, raw samples",
    "sft_notes": "SFT, style notes",
    "sft_edits": "SFT, edit pairs",
    "sft_shuffled": "SFT, shuffled writer (control)",
    "dpo": "SFT + DPO (simulated edits)",
    "grpo_best": "SFT + GRPO, dev-best checkpoint",
    "grpo_final": "SFT + GRPO, final (step 200)",
    "R1_prompt": "Ep 2: prompt only (8 samples)",
    "R2_harness": "Ep 2: harness",
    "R3_voice": "Ep 2: harness + voice vector",
    "xR3_s0.3": "Ep 2: voice vector at 0.3 (exploratory)",
    "G2_gemini_harness": "Ep 2: Gemini in the harness",
    "ep1-gemini-edits": "Ep 1: Gemini, samples + edits",
}
TRAINED = {"sft_raw", "sft_notes", "sft_edits", "dpo", "grpo_best", "grpo_final"}
CTRL = {"sft_shuffled", "base_raw"}
ARMS = [
    "base_raw",
    "sft_raw",
    "sft_notes",
    "sft_edits",
    "sft_shuffled",
    "dpo",
    "grpo_best",
    "grpo_final",
    "R1_prompt",
    "R2_harness",
    "R3_voice",
    "G2_gemini_harness",
    "ep1-gemini-edits",
]


def color(a):
    return ACCENT if a in TRAINED else CONTROL if a in CTRL else SOFT


def present(s, names, key="position"):
    return [a for a in names if a in s["arms"] and s["arms"][a][key][0] is not None]


def floor_ceiling(ax):
    for x, t in ((0, "a different human (0)"), (1, "the same human (1)")):
        ax.axvline(x, color=FAINT, lw=1, ls="--")
        ax.text(x, 1.01, t, transform=ax.get_xaxis_transform(), ha="center", fontsize=8, color=SOFT)


def fig1(s):
    order = present(s, ARMS)
    fig, ax = plt.subplots(figsize=(8.4, 0.4 * len(order) + 1.4))
    for i, a in enumerate(order):
        v, lo, hi = s["arms"][a]["position"]
        ax.plot([lo, hi], [i, i], color=FAINT, lw=1.4)
        ax.scatter([v], [i], s=40, color=color(a), zorder=3)
        print(f"fig1 {a:20s} {v:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    floor_ceiling(ax)
    ax.set_yticks(range(len(order)), [LABEL[a] for a in order])
    ax.invert_yaxis()
    ax.set_xlim(-1.3, 1.05)
    ax.set_xlabel("Voice position (strict LUAR, 4-draft episodes; 30 held-out writers, 95% bootstrap)")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_ladder.png", dpi=180)
    plt.close(fig)


def fig2(s):
    order = present(s, ARMS, "unsupported")
    fig, ax = plt.subplots(figsize=(8.4, 0.38 * len(order) + 1.4))
    for i, a in enumerate(order):
        v, lo, hi = s["arms"][a]["unsupported"]
        ax.plot([lo, hi], [i, i], color=FAINT, lw=1.4)
        ax.scatter([v], [i], s=40, color=color(a), zorder=3)
        print(f"fig2 {a:20s} unsupported {v:.3f} [{lo:.3f}, {hi:.3f}]")
    ax.axvline(0.2167, color=FAINT, ls=":", lw=1)
    ax.text(
        0.2167,
        1.01,
        "Episode 2's honesty gate",
        transform=ax.get_xaxis_transform(),
        ha="center",
        fontsize=8,
        color=SOFT,
    )
    ax.set_yticks(range(len(order)), [LABEL[a] for a in order])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.75)
    ax.set_xlabel(
        "Share of drafts with a claim the brief does not support (frozen extractor; lower bounds; 95% bootstrap)"
    )
    fig.tight_layout()
    fig.savefig(FIG / "fig2_honesty.png", dpi=180)
    plt.close(fig)


def hh_series(h):
    run = next(iter(h["runs"].values()))
    start = run["start"]
    keys = ("position", "unsupported", "words", "loop_share", "task", "attr_mean_rank")
    pts = [dict(step=0, train_reward=None, **{k: start[k] for k in keys})] if start else []
    return run, pts + run["checkpoints"]


def fig3(h):
    run, pts = hh_series(h)
    curve = run["reward_curve"]
    st = [p["step"] for p in pts]
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4), sharex=True)
    ax = axes[0][0]
    ax.plot([c["step"] for c in curve], [c["reward"] for c in curve], color=ACCENT, lw=1.2)
    ax.set_title("Training reward (what GRPO optimises)", fontsize=10, color=INK)
    ax = axes[0][1]
    ax.plot(st, [p["position"][0] for p in pts], color=INK, marker="o", ms=4)
    ax.axhline(0, color=FAINT, ls="--", lw=1)
    ax.set_title("Voice position on dev writers (never optimised)", fontsize=10, color=INK)
    ax = axes[1][0]
    ax.plot(st, [p["words"] for p in pts], color=INK, marker="o", ms=4)
    ax.set_title("Words per draft", fontsize=10, color=INK)
    ax.set_xlabel("GRPO step (0 = the SFT start)")
    ax = axes[1][1]
    ax.plot(
        st, [p["unsupported"][0] for p in pts], color=INK, marker="o", ms=4, label="unsupported (32B extractor, unseen)"
    )
    ax.plot(
        [c["step"] for c in curve],
        [c["honesty_proxy"] for c in curve],
        color=ACCENT,
        lw=1,
        label="honesty-proxy flag rate (in the reward, penalised)",
    )
    ax.set_title("Invented claims: the unseen ruler vs the proxy", fontsize=10, color=INK)
    ax.set_xlabel("GRPO step (0 = the SFT start)")
    ax.legend(frameon=False, fontsize=8)
    for p in pts:
        print(
            f"fig3 step {p['step']:3d} position {p['position'][0]:+.3f} unsupported {p['unsupported'][0]:.3f} "
            f"words {p['words']} reward {p.get('train_reward')}"
        )
    fig.tight_layout()
    fig.savefig(FIG / "fig3_hack_hunt.png", dpi=180)
    plt.close(fig)


def fig4(s):
    arms = present(s, ["sft_raw", "sft_notes", "sft_edits", "sft_shuffled", "base_raw"])
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for a in arms:
        r = s["arms"][a]
        ax.scatter([r["position"][0]], [r["spec"][0]], s=60, color=color(a), zorder=3)
        ax.annotate(
            LABEL[a],
            (r["position"][0], r["spec"][0]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8.5,
            color=INK,
        )
        print(f"fig4 {a:14s} position {r['position'][0]:+.3f} spec {r['spec'][0]:+.4f}")
    ax.axhline(0, color=FAINT, lw=1)
    ax.set_xlabel("Voice position (strict)")
    ax.set_ylabel("Specificity (this writer minus the other 29)")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_formats.png", dpi=180)
    plt.close(fig)


def cover(h):
    from PIL import Image

    run, pts = hh_series(h)
    curve = run["reward_curve"]
    fig = plt.figure(figsize=(12, 6.75), facecolor="#faf6ef")
    ax = fig.add_axes([0.07, 0.15, 0.84, 0.6], facecolor="#faf6ef")
    ax.plot([c["step"] for c in curve], [c["reward"] for c in curve], color=ACCENT, lw=2)
    ax2 = ax.twinx()
    ax2.plot([p["step"] for p in pts], [p["words"] for p in pts], color=INK, lw=2, marker="o")
    ax.set_xlabel("GRPO step", fontsize=12)
    ax.set_ylabel("training reward", color=ACCENT, fontsize=12)
    ax2.set_ylabel("words per draft (dev writers)", color=INK, fontsize=12)
    fig.text(0.04, 0.90, "SOUNDS LIKE ME · EPISODE 3", fontsize=13, color=ACCENT, weight="bold")
    fig.text(
        0.04, 0.83, "The reward went up; the emails stopped saying anything", fontsize=24, color=INK, weight="bold"
    )
    fig.text(
        0.04,
        0.04,
        "Qwen3-8B, GRPO on a voice reward model + honesty proxy − AI detector − copying · open models only",
        fontsize=11,
        color=SOFT,
    )
    fig.savefig(FIG / "hero.jpg", dpi=100, facecolor=fig.get_facecolor(), pil_kwargs={"quality": 88})
    plt.close(fig)
    hero = Image.open(FIG / "hero.jpg").convert("RGB")
    w = round(hero.height * 720 / 480)
    box = ((hero.width - w) // 2, 0, (hero.width - w) // 2 + w, hero.height)
    hero.crop(box).resize((720, 480), Image.Resampling.LANCZOS).save(
        HERE.parent / "shared" / "thumbs" / "sounds-like-me-ep3.jpg", quality=84, optimize=True
    )


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    s = json.loads((R / "summary.json").read_text())
    h = json.loads((R / "hack_hunt.json").read_text())
    fig1(s)
    fig2(s)
    fig3(h)
    fig4(s)
    cover(h)


if __name__ == "__main__":
    main()
