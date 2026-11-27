"""Draw the SOO note's figures from results/ and print every number they plot.

    .venv/bin/python experiments/repl-soo-empathy/make_figures.py

Writes to report/figures/:
- main.png       deception on the burglar scenario (paper's prompt), original and mirrored room order, per arm
                 (base, honesty prompt, SOO, random pair; exploratory last-token arms hatched), both models;
- answers.png    what the replies are: honest / deceptive / no answer, per run (main scenario, original order)
- capability.png MMLU-500 and Perspectives accuracy per arm, both models
- latent.png     self/other MSE per layer (o_proj), base vs SOO vs random pair
- hero.jpg       cover: streamlines bent by the SOO adapter's own update at layer 19 (Mistral, seed 0): the q_proj LoRA
                 product B·A summed over its inputs, 4,096 numbers on a 64 x 64 grid. Also shared/thumbs/<name>.jpg.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS, FIGURES = HERE / "results", HERE / "report" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
INK, SOFT, FAINT, RULE = "#1d272a", "#586468", "#8a9497", "#e3e7e8"
ACCENT, CONTROL = "#c9561c", "#4a86e8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelcolor": SOFT, "text.color": INK})
S = json.loads((RESULTS / "summary.json").read_text())["models"]
NAMES = {"mistral": "Mistral-7B-Instruct-v0.2", "qwen3": "Qwen3-8B"}
LT = ("soolt", "randlt")
ARMS = [
    ("base", "base", FAINT, None),
    ("honest", "honesty prompt", "#b9c0c2", None),
    ("soo", "SOO", ACCENT, None),
    ("rand", "random pair", CONTROL, None),
    ("soolt", "SOO last-token*", ACCENT, "///"),
    ("randlt", "random pair last-token*", CONTROL, "///"),
]


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=SOFT, length=0)
    ax.grid(axis="y", color=RULE, lw=0.6, zorder=0)


def value(m, arm, sc, order, variant="plain"):
    """(mean, [lo, hi], per-seed list or None) for one arm."""
    s = S[m]
    if arm in ("base", "honest"):
        c = s["cells"].get(f"base|{sc}|{order}|{'honest' if arm == 'honest' else variant}")
        return (c["rate"], c["ci"], None) if c else (None, None, None)
    a = s["arms"].get(f"{arm}|{sc}|{order}|{variant}")
    return (a["mean"], a["ci_items"], a["per_seed"]) if a else (None, None, None)


def bars(ax, m, sc, orders, title, ylabel):
    width = 0.13
    for j, order in enumerate(orders):
        for k, (arm, label, color, hatch) in enumerate(ARMS):
            if arm in LT:  # the plain-prompt rule misreads these replies (section 4.8); shown in Table 3 instead
                continue
            v, ci, seeds = value(m, arm, sc, order)
            if v is None:
                continue
            x = j + (k - 1.5) * width
            ax.bar(
                x,
                v,
                width * 0.9,
                color="white" if hatch else color,
                edgecolor=color,
                hatch=hatch,
                lw=0.8,
                label=label if j == 0 else None,
                zorder=3,
            )
            ax.errorbar(x, v, yerr=[[v - ci[0]], [ci[1] - v]], color=INK, lw=0.8, capsize=1.5, zorder=4)
            if seeds:
                ax.scatter([x] * len(seeds), seeds, s=5, color=INK, zorder=5)
            print(
                f"  {m} {sc} {order} {arm}: {v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]" + (f" seeds {seeds}" if seeds else "")
            )
    ax.set_xticks(range(len(orders)))
    ax.set_xticklabels(["original order" if o == "orig" else "mirrored order" for o in orders])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9, loc="left", color=INK)
    style(ax)


def main_fig():
    print("Figure 1: deception rate on the main scenario (deceptive / 250)")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), dpi=200, sharey=True)
    for ax, m in zip(axes, ("mistral", "qwen3"), strict=True):
        if m in S:
            bars(ax, m, "main", ("orig", "mirr"), NAMES[m], "deceptive replies (share of 250)")
    axes[0].legend(frameon=False, fontsize=6.5, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "main.png", bbox_inches="tight")
    plt.close(fig)


def run_order(r):
    parts = r.split("-")
    arm = parts[1] if len(parts) > 1 else "base"
    return (["base", "soo", "rand", "soolt", "randlt"].index(arm), r)


def answers_fig():
    print("Figure 2: what the replies are (main, original order, paper's prompt), counts of 250")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9), dpi=200, sharey=True)
    for ax, m in zip(axes, ("mistral", "qwen3"), strict=True):
        if m not in S:
            continue
        cells = S[m]["cells"]
        runs = sorted({k.split("|")[0] for k in cells if not any(f"-{x}-" in k for x in LT)}, key=run_order)
        labels, H, D, N = [], [], [], []
        for r in runs:
            c = cells.get(f"{r}|main|orig|plain")
            if not c:
                continue
            labels.append(r.replace(f"{m}-", "").replace("-s", " s"))
            H.append(c["honest"] / c["n"])
            D.append(c["deceptive"] / c["n"])
            N.append((c["refusal"] + c["other"]) / c["n"])
            print(f"  {m} {r}: honest {c['honest']} deceptive {c['deceptive']} no answer {c['refusal'] + c['other']}")
        x = np.arange(len(labels))
        ax.bar(x, D, color=ACCENT, label="deceptive (cheap room)", zorder=3)
        ax.bar(x, H, bottom=D, color="#2a9d8f", label="honest (valuable room)", zorder=3)
        ax.bar(x, N, bottom=np.array(D) + np.array(H), color=RULE, label="no room chosen", zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6.5)
        ax.set_title(NAMES[m], fontsize=9, loc="left", color=INK)
        ax.set_ylim(0, 1.0)
        style(ax)
    axes[0].set_ylabel("share of 250 replies")
    axes[0].legend(frameon=False, fontsize=6.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES / "answers.png", bbox_inches="tight")
    plt.close(fig)


def capability_fig():
    print("Figure 3: MMLU-500 accuracy and Perspectives accuracy (honest = Bob's belief) per run")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9), dpi=200)
    for ax, what in zip(axes, ("mmlu", "perspectives"), strict=True):
        xt, k = [], 0.0
        for m in ("mistral", "qwen3"):
            if m not in S or (what == "perspectives" and m == "qwen3"):  # Qwen3 runs past the token cap there (4.6)
                continue
            for arm, label, color, hatch in ARMS:
                if arm == "honest" or (what == "perspectives" and arm in LT):
                    continue
                if what == "mmlu":
                    runs = ["base"] if arm == "base" else [r for r in S[m]["mmlu"] if r.split("-")[1:2] == [arm]]
                    vals = [S[m]["mmlu"][r]["acc"] for r in runs]
                else:
                    v, _ci, seeds = value(m, arm, "perspectives", "orig")
                    vals = seeds or ([v] if v is not None else [])
                if not vals:
                    continue
                mean = float(np.mean(vals))
                ax.bar(k, mean, 0.8, color="white" if hatch else color, edgecolor=color, hatch=hatch, lw=0.8, zorder=3)
                ax.scatter([k] * len(vals), vals, s=5, color=INK, zorder=5)
                print(f"  {what} {m} {arm}: mean {mean:.3f} runs {[round(x, 3) for x in vals]}")
                xt.append((k, f"{'M' if m == 'mistral' else 'Q'} {label}"))
                k += 1
            k += 0.6
        ax.set_xticks([a for a, _ in xt])
        ax.set_xticklabels([b for _, b in xt], rotation=60, ha="right", fontsize=6.5)
        ax.set_ylim(0, 1.02)
        ax.set_title(
            "MMLU-500 accuracy (chance 0.25)" if what == "mmlu" else "Perspectives: where Bob thinks it is",
            fontsize=9,
            loc="left",
            color=INK,
        )
        style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "capability.png", bbox_inches="tight")
    plt.close(fig)


def general_fig():
    print("Figure 4: generalisation scenarios, original order (deceptive / 250; seven = mean of the seven variants)")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9), dpi=200, sharey=True)
    for ax, m in zip(axes, ("mistral", "qwen3"), strict=True):
        if m not in S:
            continue
        groups = ["seven", "treasure_hunt", "escape_room"]
        width = 0.13
        for j, g in enumerate(groups):
            for k, (arm, label, color, hatch) in enumerate(ARMS):
                if arm == "honest":
                    continue
                v = S[m]["seven_mean"].get(f"{arm}|orig") if g == "seven" else value(m, arm, g, "orig")[0]
                if v is None:
                    continue
                ax.bar(
                    j + (k - 2.5) * width,
                    v,
                    width * 0.9,
                    color="white" if hatch else color,
                    edgecolor=color,
                    hatch=hatch,
                    lw=0.8,
                    label=label if j == 0 else None,
                    zorder=3,
                )
                print(f"  {m} {g} {arm}: {v:.3f}")
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(["7 wording variants", "Treasure Hunt", "Escape Room"])
        ax.set_ylim(0, 1.02)
        ax.set_title(NAMES[m], fontsize=9, loc="left", color=INK)
        style(ax)
    axes[0].set_ylabel("deceptive replies (share)")
    axes[1].legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "general.png", bbox_inches="tight")
    plt.close(fig)


def latent_fig():
    path = RESULTS / "latent.json"
    if not path.exists():
        return
    lat = json.loads(path.read_text())
    print("Figure 5: self/other MSE of o_proj outputs per layer (78 held-out pairs)")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.7), dpi=200)
    for ax, m in zip(axes, ("mistral", "qwen3"), strict=True):
        if m not in lat:
            continue
        d = lat[m]
        ax.plot(d["base"]["o_proj"], color=INK, lw=1.4, label="base")
        for name, vals in d.items():
            if name == "base":
                continue
            arm = name.split("-")[1]
            ax.plot(
                vals["o_proj"],
                color=ACCENT if arm == "soo" else CONTROL,
                lw=0.8,
                alpha=0.7,
                label={"soo": "SOO seeds", "rand": "random-pair seeds"}.get(arm) if name.endswith("s0") else None,
            )
        means = {n: float(np.mean(v["o_proj"])) for n, v in d.items()}
        print(f"  {m}: " + ", ".join(f"{n} {v:.4f}" for n, v in means.items()))
        ax.set_yscale("log")
        ax.set_xlabel("layer")
        ax.set_title(NAMES[m], fontsize=9, loc="left", color=INK)
        style(ax)
    axes[0].set_ylabel("MSE (self vs other prompt)")
    axes[0].legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "latent.png", bbox_inches="tight")
    plt.close(fig)


def cover():
    from PIL import Image

    path = HERE / "data" / "adapter_q_proj_l19.npy"
    if not path.exists():
        return
    sys.path.insert(0, str(HERE.parent / "shared"))
    import make_art

    vector = np.load(path).astype(float)

    def field(model, concept):
        side = int(np.sqrt(vector.size))
        grid = vector[: side * side].reshape(side, side)
        for _ in range(6):
            padded = np.pad(grid, 1, mode="wrap")
            grid = sum(padded[i : i + side, j : j + side] for i in range(3) for j in range(3)) / 9
        return grid / np.abs(grid).max()

    make_art.field = field
    make_art.ART["repl-soo-empathy"] = ("mistral-7b", "SOO adapter (layer 19)", 0.16, 1.8)
    make_art.draw("repl-soo-empathy", (1600, 900), FIGURES / "hero.jpg")
    hero = Image.open(FIGURES / "hero.jpg").convert("RGB")
    hero.resize((1200, 675), Image.Resampling.LANCZOS).crop((0, 22, 1200, 652)).save(FIGURES / "og.jpg", quality=88)
    w = round(hero.height * 720 / 480)
    left = (hero.width - w) // 2
    hero.crop((left, 0, left + w, hero.height)).resize((720, 480), Image.Resampling.LANCZOS).save(
        HERE.parent / "shared" / "thumbs" / "repl-soo-empathy.jpg", quality=88
    )


if __name__ == "__main__":
    cover()
    main_fig()
    answers_fig()
    capability_fig()
    latent_fig()
