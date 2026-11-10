"""Draw the note's figures from results/, printing every number plotted.

    python experiments/deslop/make_figures.py

methods.png   mean deslop score per held-out reply for every method, with 95% bootstrap intervals over the 48 prompts;
              the shaded band runs from 0 to the human 90th percentile.
tradeoff.png  the deslop score against two costs: KL drift on human prose, and the blind judge's win rate against the
              base model (0.5 = level).
habits.png    where each method's points come from: points per 1000 words by habit, pooled over the 48 replies, beside
              the two human sources.
sweep.png     the steering sweep on the 24 dev prompts: mean score by strength for each layer; hollow markers had fewer
              than 22 of 24 fluent replies.
eqbench.png   our score against EQ-Bench's slop score (slop-score's own code) for every method and both human sources.
voice.png     `deslop voice`: Burrows' Delta from each rewrite to its target writer, before (the original reply) and
              after, for the API rewriter and our LoRA, averaged over 10 texts per writer.
detect.png    share of each set flagged by three open AI-text detectors, each at a threshold that flags 1% of the 318
              human passages; with 95% bootstrap intervals.
viewer.json   the interactive reply viewer's data: 12 held-out prompts (two per genre) and seven setups, each reply with
              its score and every habit span, as the toolkit finds them.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "packages" / "deslop" / "src"))
from deslop.habits import DETECTORS, HABIT_LABELS, WEIGHTS, count_words, find_spans
from deslop.score import human_band, score

RESULTS = HERE / "results"
FIGURES = HERE / "report" / "figures"
EMBER, GREY, LIGHT, INK, MUTED, BAND = (
    "#C9561C",
    "#6F6A63",
    "#C8C2B8",
    "#2B2723",
    "#8A847C",
    "#EDE7DD",
)

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

LABELS = {
    "base": "Qwen3-4B as released",
    "plain": "plain-style prompt",
    "sampler": "backtracking sampler",
    "plain+sampler": "prompt + sampler",
    "steer": "steering (register direction)",
    "steer-random": "steering, random direction",
    "plain+steer": "prompt + steering",
    "neurons": "4 neuron flips (100toolkit's)",
    "neurons-random": "4 random neuron flips",
    "plain+neurons": "prompt + neuron flips",
    "lora": "our LoRA, round 1",
    "lora-r2": "our LoRA, round 2 (released)",
    "kannike": "100toolkit's model (v4)",
}
CONTROLS = {"steer-random", "neurons-random"}
REFERENCE = {"base", "kannike"}


def colour(run: str) -> str:
    return GREY if run in CONTROLS else INK if run in REFERENCE else EMBER


def dot(ax, x: float, y: float, run: str) -> None:
    c = colour(run)
    ax.plot(x, y, "o", color=c, ms=6, mfc="white" if run in CONTROLS else c, mew=1.5)


def methods(summary: dict) -> None:
    runs = sorted(summary["runs"], key=lambda r: summary["runs"][r]["deslop"]["value"])
    limit = human_band()["all"]["limit"]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.axvspan(0, limit, color=BAND, zorder=0)
    ax.text(
        limit + 1,
        -0.6,
        f"human band: 90% of human passages score {limit:.1f} or less",
        fontsize=8,
        color=MUTED,
    )
    print("methods.png: mean deslop score [95% CI]")
    for y, run in enumerate(runs):
        d = summary["runs"][run]["deslop"]
        ax.plot([d["lo"], d["hi"]], [y, y], color=colour(run), lw=2, solid_capstyle="round")
        dot(ax, d["value"], y, run)
        print(f"  {run:15} {d['value']:6.1f} [{d['lo']:5.1f}, {d['hi']:5.1f}]")
    ax.set_yticks(range(len(runs)), [LABELS[r] for r in runs])
    ax.set_ylim(-1, len(runs) - 0.5)
    ax.set_xlim(0, max(summary["runs"][r]["deslop"]["hi"] for r in runs) * 1.05)
    ax.set_xlabel("deslop score (weighted habits per 1000 words; lower is plainer)")
    fig.tight_layout()
    fig.savefig(FIGURES / "methods.png")
    plt.close(fig)


# Label positions (dx, dy in points, alignment) so neighbouring labels do not overlap.
KL_LABELS = {
    "plain+steer": (6, -3, "left"), "steer": (6, -3, "left"), "plain": (6, 2, "left"),
    "plain+neurons": (-6, 6, "right"), "kannike": (-6, 4, "right"), "lora-r2": (6, 4, "left"),
    "lora": (6, -10, "left"), "steer-random": (6, 2, "left"), "neurons": (-6, 8, "right"),
    "neurons-random": (6, -12, "left"), "base": (6, 10, "left"), "sampler": (6, 6, "left"),
    "plain+sampler": (-6, 6, "right"),
}  # fmt: skip
JUDGE_LABELS = {
    "plain": (6, 4, "left"), "lora-r2": (6, 6, "left"), "lora": (-6, -11, "right"), "sampler": (6, 2, "left"),
    "plain+neurons": (6, -12, "left"), "plain+sampler": (6, 2, "left"), "kannike": (6, 2, "left"),
    "plain+steer": (6, -3, "left"), "steer": (6, -3, "left"), "neurons": (-6, 4, "right"),
    "neurons-random": (-6, 4, "right"), "steer-random": (6, -3, "left"),
}  # fmt: skip


def tradeoff(summary: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), sharex=True)
    print("tradeoff.png: deslop, KL, judge win rate")
    for run, o in summary["runs"].items():
        x = o["deslop"]["value"]
        kl = o.get("kl", {}).get("value")
        win = o.get("judge", {}).get("win_rate", {}).get("value")
        print(f"  {run:15} deslop {x:6.1f} kl {kl} judge {win}")
        for ax, y, where in ((axes[0], kl, KL_LABELS), (axes[1], win, JUDGE_LABELS)):
            if y is None or run not in where:
                continue
            dot(ax, x, y, run)
            dx, dy, ha = where[run]
            ax.annotate(LABELS[run], (x, y), textcoords="offset points", xytext=(dx, dy), fontsize=7, ha=ha,
                        color=colour(run))  # fmt: skip
    axes[0].set_yscale("symlog", linthresh=0.01)
    axes[0].set_ylim(0, 3)
    axes[0].set_ylabel("KL drift on human prose (nats per token, symlog)")
    axes[1].axhline(0.5, color=MUTED, lw=1, ls="--")
    axes[1].text(99, 0.51, "level with the base model", ha="right", fontsize=7, color=MUTED)
    axes[1].set_ylabel("judge win rate against the base model")
    axes[1].set_ylim(0, 0.6)
    for ax in axes:
        ax.set_xlabel("deslop score")
        ax.set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(FIGURES / "tradeoff.png")
    plt.close(fig)


def human_points() -> dict[str, dict[str, float]]:
    rows = [json.loads(line) for line in (HERE / "data" / "human" / "passages.jsonl").open(encoding="utf-8")]
    out = {}
    for source in ("gutenberg", "wikipedia"):
        texts = [r["text"] for r in rows if r["source"] == source]
        words = sum(count_words(t) for t in texts)
        counts = Counter(s.habit for t in texts for s in find_spans(t))
        out[source] = {h: 1000 * counts[h] * WEIGHTS[h] / words for h in DETECTORS}
    return out


def habits(summary: dict) -> None:
    order = [
        "base",
        "plain",
        "sampler",
        "steer",
        "neurons",
        "plain+sampler",
        "lora-r2",
        "kannike",
    ]
    rows = [(LABELS[r], summary["runs"][r]["habit_points_per_1k"]) for r in order if r in summary["runs"]]
    human = human_points()
    rows += [
        ("human: Gutenberg essays", human["gutenberg"]),
        ("human: Wikipedia 2021", human["wikipedia"]),
    ]
    shades = {
        "triad": "#E3B9A0", "em_dash": EMBER, "not_x_but_y": "#8E3A12", "slop_word": "#D98B5F",
        "formatting": "#B9B2A7", "rhetorical_qa": "#A39A8E", "signpost": "#6F6A63", "stock_phrase": "#4F4A44",
        "fragment": "#DCD5CA", "which_chain": "#9C8F80", "invented_claim": "#2B2723",
    }  # fmt: skip
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    print("habits.png: points per 1000 words by habit")
    for y, (label, pts) in enumerate(reversed(rows)):
        left = 0.0
        for habit in DETECTORS:
            ax.barh(
                y,
                pts[habit],
                left=left,
                color=shades[habit],
                height=0.7,
                edgecolor="white",
                lw=0.4,
            )
            left += pts[habit]
        print(f"  {label:30} total {left:6.1f} " + " ".join(f"{h} {pts[h]:.1f}" for h in DETECTORS if pts[h] >= 0.5))
    ax.set_yticks(range(len(rows)), [label for label, _ in reversed(rows)])
    ax.set_xlabel("points per 1000 words")
    handles = [plt.Rectangle((0, 0), 1, 1, color=shades[h]) for h in DETECTORS]
    ax.legend(
        handles,
        [HABIT_LABELS[h] for h in DETECTORS],
        fontsize=7,
        frameon=False,
        loc="lower right",
        ncol=2,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "habits.png")
    plt.close(fig)


def sweep() -> None:
    data = json.loads((RESULTS / "steer_sweep.json").read_text())
    choice = json.loads((RESULTS / "steer_choice.json").read_text())
    none = next(s for s in data if s["layer"] is None)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.axhline(none["mean_score"], color=MUTED, ls="--", lw=1)
    ax.text(-6, none["mean_score"] + 2, "no steering", fontsize=8, color=MUTED)
    print(f"sweep.png: unsteered {none['mean_score']:.1f}")
    for layer, c in ((12, EMBER), (18, "#8E3A12"), (24, "#D98B5F")):
        pts = sorted((s for s in data if s["layer"] == layer), key=lambda s: s["strength"])
        ax.plot(
            [s["strength"] for s in pts],
            [s["mean_score"] for s in pts],
            color=c,
            lw=1.5,
            label=f"layer {layer}",
        )
        for s in pts:
            ok = s["fluent"] >= 22
            ax.plot(
                s["strength"],
                s["mean_score"],
                "o",
                color=c,
                mfc=c if ok else "white",
                mew=1.5,
            )
            print(
                f"  L{layer} {s['strength']:5} score {s['mean_score']:5.1f} fluent {s['fluent']}/24 fit {s['task_fit']}"
            )
    chosen = next(s for s in data if s["layer"] == choice["layer"] and s["strength"] == choice["strength"])
    ax.plot(
        chosen["strength"],
        chosen["mean_score"],
        "s",
        ms=12,
        mfc="none",
        mec=INK,
        mew=1.2,
    )
    ax.set_xlabel("strength (class gaps added; negative = away from the assistant register)")
    ax.set_ylabel("mean deslop score, 24 dev prompts")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "sweep.png")
    plt.close(fig)


def eqbench(summary: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    xs, ys = [], []
    print("eqbench.png: deslop mean vs EQ-Bench slop score (all replies joined)")
    for run, o in summary["runs"].items():
        x, y = o["deslop"]["value"], o["eqbench_corpus"]
        xs.append(x)
        ys.append(y)
        dot(ax, x, y, run)
        dx, dy = {
            "plain+neurons": (5, -10),
            "plain+sampler": (-5, 6),
            "steer-random": (5, -10),
        }.get(run, (5, 2))
        ax.annotate(LABELS[run], (x, y), textcoords="offset points", xytext=(dx, dy), fontsize=7, color=colour(run),
                    ha="right" if dx < 0 else "left")  # fmt: skip
        print(f"  {run:15} {x:6.1f} {y:6.1f}")
    for source, label in (
        ("gutenberg", "human: Gutenberg"),
        ("wikipedia", "human: Wikipedia"),
    ):
        h = summary["human"][source]
        x, y = h["deslop"]["value"], h["eqbench_corpus"]
        ax.plot(x, y, "D", color=MUTED)
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(6, -3),
            fontsize=7,
            color=MUTED,
        )
        print(f"  {label:15} {x:6.1f} {y:6.1f}")

    def rank(v: np.ndarray) -> np.ndarray:
        return np.argsort(np.argsort(v))

    rho = np.corrcoef(rank(np.array(xs)), rank(np.array(ys)))[0, 1]
    print(f"  Spearman rho over methods: {rho:.2f}")
    ax.set_xlabel("deslop score (this toolkit)")
    ax.set_ylabel("EQ-Bench slop score (slop-score's code)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    fig.tight_layout()
    fig.savefig(FIGURES / "eqbench.png")
    plt.close(fig)


VIEWER_RUNS = [
    "base",
    "plain",
    "sampler",
    "plain+sampler",
    "steer",
    "lora-r2",
    "kannike",
]
VIEWER_PROMPTS = [
    "h00-opinion", "h07-opinion", "h03-blog", "h09-blog", "h04-explainer", "h05-explainer",
    "h06-linkedin", "h01-linkedin", "h00-email", "h03-email", "h04-reflection", "h01-reflection",
]  # fmt: skip


def viewer() -> None:
    sys.path.insert(0, str(HERE))
    from prompts import held_out

    prompts = {p["id"]: p for p in held_out()}
    data = {
        "runs": {r: LABELS[r] for r in VIEWER_RUNS},
        "habits": HABIT_LABELS,
        "prompts": [],
        "replies": {},
    }
    for run in VIEWER_RUNS:
        rows = {r["id"]: r["text"] for r in map(json.loads, (RESULTS / "gen" / f"{run}.jsonl").open(encoding="utf-8"))}
        for pid in VIEWER_PROMPTS:
            s = score(rows[pid])
            spans = [[sp.start, sp.end, sp.habit] for sp in s.spans]
            data["replies"][f"{run}/{pid}"] = {
                "text": rows[pid],
                "score": round(s.score, 1),
                "spans": spans,
            }
    data["prompts"] = [
        {"id": pid, "genre": prompts[pid]["genre"], "prompt": prompts[pid]["prompt"]} for pid in VIEWER_PROMPTS
    ]
    (HERE / "report" / "viewer.json").write_text(json.dumps(data, ensure_ascii=False))
    print(f"viewer.json: {len(VIEWER_PROMPTS)} prompts x {len(VIEWER_RUNS)} setups")


AUTHORS = ["twain", "lincoln", "darwin", "woolf", "chesterton"]


def voice() -> None:
    from deslop.voice import delta

    rows = [json.loads(line) for line in (RESULTS / "voices" / "rewrites_scored.jsonl").open(encoding="utf-8")]
    profiles = json.loads((RESULTS / "voices" / "profiles.json").read_text())
    passages = [json.loads(line) for line in (HERE / "data" / "voices" / "passages.jsonl").open(encoding="utf-8")]
    control = [r for r in rows if r["target"] == "none"]
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    print("voice.png: mean Delta to the target writer (lower is closer)")
    for i, author in enumerate(AUTHORS):
        mine = [r for r in rows if r["target"] == author]
        before = np.mean([r["delta_target_original"] for r in mine])
        api = np.mean([r["delta_target"] for r in mine if r["rewriter"] == "api"])
        lora = np.mean([r["delta_target"] for r in mine if r["rewriter"] == "lora"])
        plain = np.mean([r["delta"][author] for r in control])
        own = np.mean([delta(p["text"], profiles[author]) for p in passages
                       if p["author"] == author and p["role"] == "heldout"])  # fmt: skip
        ax.plot([i - 0.2, i + 0.1], [before, api], color=EMBER, lw=1, alpha=0.6)
        ax.plot(i - 0.2, before, "o", color=INK, ms=6)
        ax.plot(i - 0.05, plain, "^", color=GREY, ms=6, mfc="white", mew=1.5)
        ax.plot(i + 0.1, api, "o", color=EMBER, ms=6)
        ax.plot(i + 0.25, lora, "s", color=GREY, ms=6, mfc="white", mew=1.5)
        ax.plot([i - 0.3, i + 0.35], [own, own], color=MUTED, ls="--", lw=1)
        print(f"  {author:10} original {before:.3f} plain {plain:.3f} api {api:.3f} lora {lora:.3f} own {own:.3f}")
    ax.set_xticks(range(len(AUTHORS)), [a.capitalize() for a in AUTHORS])
    ax.set_ylabel("Burrows' Delta to the writer (lower is closer)")
    ax.set_ylim(0, 0.9)
    handles = [plt.Line2D([], [], marker="o", color=INK, ls=""),
               plt.Line2D([], [], marker="^", color=GREY, mfc="white", ls=""),
               plt.Line2D([], [], marker="o", color=EMBER, ls=""),
               plt.Line2D([], [], marker="s", color=GREY, mfc="white", ls=""),
               plt.Line2D([], [], color=MUTED, ls="--")]  # fmt: skip
    labels = ["Qwen's original reply", "plain rewrite, no voice (control)", "voice rewrite, gpt-5.4-mini",
              "voice rewrite, our LoRA", "the writer's own held-out passages"]  # fmt: skip
    ax.legend(handles, labels, frameon=False, fontsize=7.5, loc="lower center", ncol=3)
    fig.tight_layout()
    fig.savefig(FIGURES / "voice.png")
    plt.close(fig)


def detect(summary: dict) -> None:
    sets = summary["detectors"]["sets"]
    order = [("human/gutenberg", "human: Gutenberg essays"), ("human/wikipedia", "human: Wikipedia 2021"),
             ("author", "human: voice authors, held out"), ("mirror", "human passages edited by Qwen"),
             ("run/base", "Qwen3-4B as released"), ("run/plain", "plain-style prompt"),
             ("run/sampler", "backtracking sampler"), ("run/plain+sampler", "prompt + sampler"),
             ("run/steer", "steering"), ("run/lora-r2", "our LoRA (released)"), ("run/kannike", "100toolkit's model"),
             ("voice/api", "voice rewrites, gpt-5.4-mini"), ("voice/lora", "voice rewrites, our LoRA")]  # fmt: skip
    names = [("desklib", "Desklib classifier", EMBER), ("binoculars", "Binoculars", INK),
             ("fastdetect", "Fast-DetectGPT", GREY)]  # fmt: skip
    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    print("detect.png: share flagged at 1% human false-positive thresholds")
    for row, (key, label) in enumerate(reversed(order)):
        for k, (name, _, c) in enumerate(names):
            cellv = sets[key].get(f"{name}_calibrated_1pct")
            if not cellv:
                continue
            y = row + (k - 1) * 0.26
            ax.barh(y, cellv["value"], height=0.24, color=c)
            ax.plot([cellv["lo"], cellv["hi"]], [y, y], color=MUTED, lw=1)
        print(f"  {label:34} " + " ".join(
            f"{n} {sets[key][f'{n}_calibrated_1pct']['value']:.2f}" for n, _, _ in names))  # fmt: skip
    ax.set_yticks(range(len(order)), [label for _, label in reversed(order)])
    ax.set_xlim(0, 1)
    ax.set_xlabel("share flagged as AI (threshold set so 1% of 318 human passages are flagged)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in names]
    ax.legend(handles, [label for _, label, _ in names], frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "detect.png")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    viewer()
    summary = json.loads((RESULTS / "summary.json").read_text())
    methods(summary)
    tradeoff(summary)
    habits(summary)
    sweep()
    eqbench(summary)
    if (RESULTS / "voices" / "rewrites_scored.jsonl").exists():
        voice()
    if "detectors" in summary:
        detect(summary)


if __name__ == "__main__":
    main()
