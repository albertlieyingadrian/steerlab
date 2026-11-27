"""Draw the reason-based refusal note's figures from results/ and print every number they plot.

    .venv/bin/python experiments/repl-reason-based-refusal/make_figures.py

Writes to report/figures/:
- fixed.png      undesired final replies after the model's own initial reply, a fixed polite refusal and a fixed
                 rebuttal (the paper's Figure 6), per scenario and model, main texts, 100 replies per bar
- variants.png   the same with the 4 other refusals and 4 other rebuttals (25 replies each, pooled), three scenarios
- coffee.png     natural condition: steering the customer away when the instruction is about race vs about tea
- trained.png    Qwen3-8B base vs its fine-tunes (own replies, polite refusals, rebuttals): undesired final replies per
                 scenario, the initial reply's type, held-out AdvBench decline rate and MMLU
- hero.jpg       cover: every final reply in the prompting grid as one dot (orange = undesired), rows = model x
                 initial reply, columns = scenario; plus og.jpg and shared/thumbs/<name>.jpg
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS, FIGURES = HERE / "results", HERE / "report" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
INK, SOFT, FAINT, RULE = "#1d272a", "#586468", "#8a9497", "#e3e7e8"
ACCENT, CONTROL, TEAL = "#c9561c", "#4a86e8", "#2a9d8f"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelcolor": SOFT, "text.color": INK})
S = json.loads((RESULTS / "summary.json").read_text())
MODELS = [("qwen3", "Qwen3-8B"), ("llama", "Llama-3.1-8B"), ("mistral", "Mistral-7B-v0.2")]
SCEN = [
    ("car", "car sales:\nhides accident"),
    ("estate", "real estate:\nsteers by race"),
    ("insider", "insider\ntrading: buys"),
]
CONDS = [
    ("natural", "own initial reply", FAINT),
    ("refusal", "fixed polite refusal", ACCENT),
    ("rebuttal", "fixed rebuttal", TEAL),
]


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=SOFT, length=0)
    ax.grid(axis="y", color=RULE, lw=0.6, zorder=0)


def cell(m, run, sc, cond, v="main"):
    return S["cells"].get(f"{m}|{run}|{sc}|{cond}|{v}")


def bar(ax, x, c, color, w, label=None, hatch=None):
    r, (lo, hi) = c["undesired_rate"], c["undesired_ci"]
    ax.bar(x, r, w, color="white" if hatch else color, edgecolor=color, hatch=hatch, lw=0.8, label=label, zorder=3)
    ax.errorbar(x, r, yerr=[[r - lo], [hi - r]], color=INK, lw=0.8, capsize=1.5, zorder=4)


def fixed_fig():
    print("Figure 1: undesired final replies by initial reply (main texts), undesired / n [95% CI]")
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0), dpi=200, sharey=True)
    for ax, (m, name) in zip(axes, MODELS, strict=True):
        for j, (sc, label) in enumerate(SCEN):
            for k, (cond, clabel, color) in enumerate(CONDS):
                c = cell(m, "base", sc, cond)
                if not c:
                    continue
                bar(ax, j + (k - 1) * 0.26, c, color, 0.24, clabel if j == 0 else None)
                print(
                    f"  {m:8s} {sc:8s} {cond:9s} {c['undesired']:3d}/{c['n']} [{c['undesired_ci'][0]:.2f}, "
                    f"{c['undesired_ci'][1]:.2f}] reasoning {c['reasoning']}/{c['n']} unlabelled {c['unlabelled']}"
                )
        ax.set_xticks(range(len(SCEN)))
        ax.set_xticklabels([s[1] for s in SCEN], fontsize=7)
        ax.set_ylim(0, 1.02)
        ax.set_title(name, fontsize=9, loc="left", color=INK)
        style(ax)
    axes[0].set_ylabel("undesired final replies (share of 100)")
    axes[0].legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "fixed.png", bbox_inches="tight")
    plt.close(fig)


def variants_fig():
    print("Figure 2: pooled over three scenarios, main text vs the four other texts")
    fig, ax = plt.subplots(figsize=(6.4, 2.8), dpi=200)
    xt = []
    specs = [
        ("refusal", "main", ACCENT, None, "refusal, paper's text"),
        ("refusal", "var", ACCENT, "///", "refusal, 4 other texts"),
        ("rebuttal", "main", TEAL, None, "rebuttal, main text"),
        ("rebuttal", "var", TEAL, "///", "rebuttal, 4 other texts"),
    ]
    for i, (m, name) in enumerate(MODELS):
        for k, (cond, v, color, hatch, lab) in enumerate(specs):
            c = S["pooled"].get(f"{m}|base|{cond}|{v}")
            if not c:
                continue
            bar(ax, i + (k - 1.5) * 0.2, c, color, 0.18, lab if i == 0 else None, hatch)
            print(
                f"  {m:8s} {cond:9s} {v:4s} {c['undesired']:3d}/{c['n']} [{c['undesired_ci'][0]:.2f}, {c['undesired_ci'][1]:.2f}]"
            )
        xt.append(name)
    ax.set_xticks(range(len(xt)))
    ax.set_xticklabels(xt)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("undesired final replies (pooled)")
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper right")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "variants.png", bbox_inches="tight")
    plt.close(fig)


def coffee_fig():
    print("Figure 3: natural condition, steering the customer away: race instruction vs tea instruction")
    fig, ax = plt.subplots(figsize=(5.0, 2.7), dpi=200)
    specs = [("estate", ACCENT, "told to steer by race"), ("coffee", FAINT, "told to steer by tea preference")]
    for i, (m, _name) in enumerate(MODELS):
        for k, (sc, color, lab) in enumerate(specs):
            c = cell(m, "base", sc, "natural")
            if not c:
                continue
            bar(ax, i + (k - 0.5) * 0.36, c, color, 0.34, lab if i == 0 else None)
            print(
                f"  {m:8s} {sc:7s} {c['undesired']}/{c['n']} reasoning {c['reasoning']}/{c['n']} init {c['initial_types']}"
            )
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([n for _, n in MODELS])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("customer steered away (share of 100)")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "coffee.png", bbox_inches="tight")
    plt.close(fig)


ARM_RUNS = [
    ("base", ["base"], FAINT, "base"),
    ("self", ["self-s0"], CONTROL, "trained on own replies"),
    ("refuse", ["refuse-s0", "refuse-s1"], ACCENT, "trained to refuse"),
    ("rebut", ["rebut-s0", "rebut-s1"], TEAL, "trained to rebut"),
]


def trained_fig():
    if "trained" not in S:
        return
    print("Figure 4: Qwen3-8B fine-tunes, natural condition (initial reply sampled from the fine-tune)")
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.0), dpi=200, gridspec_kw={"width_ratios": [3, 2, 2]})
    ax = axes[0]
    for j, (sc, _label) in enumerate(SCEN):
        for k, (arm, runs, color, lab) in enumerate(ARM_RUNS):
            cs = [c for c in (cell("qwen3", r, sc, "natural") for r in runs) if c]
            if not cs:
                continue
            u, n = sum(c["undesired"] for c in cs), sum(c["n"] for c in cs)
            seeds = [c["undesired_rate"] for c in cs]
            x = j + (k - 1.5) * 0.2
            ax.bar(x, u / n, 0.18, color=color, label=lab if j == 0 else None, zorder=3)
            if len(seeds) > 1:
                ax.scatter([x] * len(seeds), seeds, s=5, color=INK, zorder=5)
            print(f"  {sc:8s} {arm:7s} {u}/{n} seeds {seeds}")
    ax.set_xticks(range(len(SCEN)))
    ax.set_xticklabels([s[1] for s in SCEN], fontsize=7)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("undesired final replies")
    ax.legend(frameon=False, fontsize=6.3, loc="upper left")
    ax.set_title("Undesired final replies", fontsize=9, loc="left", color=INK)
    style(ax)
    ax = axes[1]
    types = [
        ("A", "goes along", ACCENT),
        ("B", "short refusal", "#e9a47c"),
        ("C", "rebuttal", TEAL),
        ("D", "other", RULE),
    ]
    for k, (arm, runs, _c, _lab) in enumerate(ARM_RUNS):
        counts = {t: 0 for t, _, _ in types}
        for r in runs:
            for sc, _ in SCEN:
                c = cell("qwen3", r, sc, "natural")
                if c:
                    for t, v in c["initial_types"].items():
                        counts[t] += v
        tot = max(1, sum(counts.values()))
        bottom = 0.0
        for t, lab, color in types:
            ax.bar(k, counts[t] / tot, 0.7, bottom=bottom, color=color, label=lab if k == 0 else None, zorder=3)
            bottom += counts[t] / tot
        print(f"  initial reply types {arm}: {counts}")
    ax.set_xticks(range(len(ARM_RUNS)))
    ax.set_xticklabels([a for a, *_ in ARM_RUNS], fontsize=7)
    ax.set_title("Initial reply to the primer", fontsize=9, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=6, loc="lower left")
    style(ax)
    ax = axes[2]
    for k, (arm, runs, color, _lab) in enumerate(ARM_RUNS):
        adv = [S["adv"].get(f"qwen3|{r}") for r in runs]
        mm = [S["mmlu"].get(f"qwen3|{r}") for r in runs]
        if all(adv):
            v = sum(a["declined"] for a in adv) / sum(a["n"] for a in adv)
            ax.bar(k - 0.18, v, 0.34, color=color, zorder=3)
            print(f"  AdvBench declined {arm}: {[a['declined'] for a in adv]} of {[a['n'] for a in adv]}")
        if all(mm):
            v = float(np.mean([a["acc"] for a in mm]))
            ax.bar(k + 0.18, v, 0.34, color="white", edgecolor=color, hatch="///", zorder=3)
            print(f"  MMLU {arm}: {[a['acc'] for a in mm]}")
    ax.set_xticks(range(len(ARM_RUNS)))
    ax.set_xticklabels([a for a, *_ in ARM_RUNS], fontsize=7)
    ax.set_ylim(0, 1.02)
    ax.set_title("AdvBench declined | MMLU (hatched)", fontsize=9, loc="left", color=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "trained.png", bbox_inches="tight")
    plt.close(fig)


def cover():
    from PIL import Image

    labels = [json.loads(line) for line in (RESULTS / "labels.jsonl").read_text().splitlines()]
    ground = "#15110e"
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=ground)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=ground)
    rows = [(m, c) for m, _ in MODELS for c in ("natural", "refusal", "rebuttal")]
    for i, (m, cond) in enumerate(rows):
        for j, (sc, _) in enumerate(SCEN):
            xs = [
                x
                for x in labels
                if x["model"] == m
                and x["run"] == "base"
                and x["scenario"] == sc
                and x["cond"] == cond
                and x["variant"] == 0
            ]
            for k, x in enumerate(xs):
                cx = 0.06 + j * 0.31 + (k % 25) * 0.0105
                cy = 0.9 - i * 0.092 - (k // 25) * 0.018
                ax.scatter(cx, cy, s=30, color=ACCENT if x["undesired"] else "#6f675f", lw=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.savefig(FIGURES / "hero.jpg", facecolor=ground, pil_kwargs={"quality": 90})
    plt.close(fig)
    hero = Image.open(FIGURES / "hero.jpg").convert("RGB")
    hero.resize((1200, 675), Image.Resampling.LANCZOS).crop((0, 22, 1200, 652)).save(FIGURES / "og.jpg", quality=88)
    w = round(hero.height * 720 / 480)
    left = (hero.width - w) // 2
    hero.crop((left, 0, left + w, hero.height)).resize((720, 480), Image.Resampling.LANCZOS).save(
        HERE.parent / "shared" / "thumbs" / "repl-reason-based-refusal.jpg", quality=88
    )
    print("cover: 9 rows (model x initial reply) x 3 scenarios, 100 dots each")


if __name__ == "__main__":
    cover()
    fixed_fig()
    variants_fig()
    coffee_fig()
    trained_fig()
