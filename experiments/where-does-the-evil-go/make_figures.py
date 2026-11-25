"""Draw the note's figures from results/ and print every number they plot.

    .venv/bin/python experiments/where-does-the-evil-go/make_figures.py

Writes to report/figures/:
- collapse.png   base model: mean coherence (left) and mean trait score (right) against the steering coefficient, for
                 the post's evil, sycophancy, hallucination and optimism vectors (ours solid, the post's dashed) and three
                 random directions as long as the evil vector (blue);
- finetune.png   after fine-tuning: evil score and coherence along the old evil vector, for the six fine-tunes;
- vectors.png    left: cosine by layer between each fine-tune's re-extracted evil vector and the base model's, with the
                 seed-to-seed ceiling; right: evil on request (system prompt) per model;
- gain.png       KL gain of the old evil vector and of random directions, per model.
- hero.jpg       the cover: experiments/shared/make_art.py's streamlines bent by the post's evil vector at layer 20
                 (its 3,584 numbers on a 59 x 59 grid), plus the research-page thumbnail shared/thumbs/<name>.jpg.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "report" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

INK, SOFT, FAINT, RULE = "#1d272a", "#586468", "#8a9497", "#e3e7e8"
ACCENT, CONTROL = "#c9561c", "#4a86e8"
TRAIT_COLOR = {"evil": ACCENT, "sycophantic": "#7a5195", "hallucinating": "#2a9d8f", "optimistic": "#b0a000"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelcolor": SOFT, "text.color": INK})
MODELS = {"evil2_none": "Evil II, no prevention", "evil2_prev_v": "Evil II, prevent +2.5 v",
          "evil2_prev_rand": "Evil II, prevent +2.5 random", "gsm8k_none": "GSM8K, no prevention",
          "gsm8k_prev_v": "GSM8K, prevent +2.5 v", "gsm8k_prev_rand": "GSM8K, prevent +2.5 random"}  # fmt: skip
MODEL_STYLE = {"evil2_none": (ACCENT, "-"), "evil2_prev_v": (ACCENT, "--"), "evil2_prev_rand": (CONTROL, "--"),
               "gsm8k_none": (INK, "-"), "gsm8k_prev_v": (INK, "--"), "gsm8k_prev_rand": (CONTROL, ":")}  # fmt: skip


def load(name):
    return json.loads((RESULTS / name).read_text())


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=SOFT, length=0)
    ax.grid(axis="y", color=RULE, lw=0.6, zorder=0)


def collapse():
    s, post = load("summary.json")["base"], load("post_numbers.json")
    coefs = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=200)
    print("Figure 1: base model, coherence and trait score by coefficient (ours | post)")
    for trait, color in TRAIT_COLOR.items():
        ours = [s.get(f"{trait}_c{c}") for c in coefs]
        theirs = [post.get(f"{trait}/{c}") for c in coefs]
        for k, (ax, key) in enumerate(zip(axes, ("coherence", "trait"), strict=True)):
            y = [o[key][0] if o else np.nan for o in ours]
            lo = [o[key][1][0] if o else np.nan for o in ours]
            hi = [o[key][1][1] if o else np.nan for o in ours]
            ax.plot(coefs, y, color=color, lw=1.6, marker="o", ms=3, label=trait if k == 0 else None, zorder=3)
            ax.fill_between(coefs, lo, hi, color=color, alpha=0.12, lw=0)
            ax.plot(coefs, [t[key] if t else np.nan for t in theirs], color=color, lw=1, ls="--", alpha=0.7)
        print(f"  {trait:13s} coh " + " ".join(f"{(o['coherence'][0] if o else np.nan):5.1f}" for o in ours)
              + " | " + " ".join(f"{(t['coherence'] if t else np.nan):5.1f}" for t in theirs))  # fmt: skip
        print(f"  {'':13s} trt " + " ".join(f"{(o['trait'][0] if o else np.nan):5.1f}" for o in ours)
              + " | " + " ".join(f"{(t['trait'] if t else np.nan):5.1f}" for t in theirs))  # fmt: skip
    rc = [1.0, 1.5, 2.0, 2.5]
    for key, ax in zip(("coherence", "trait"), axes, strict=True):
        ys = np.array([[s[f"rand{sd}_c{c}"][key][0] for c in rc] for sd in range(3) if f"rand{sd}_c2.5" in s])
        if len(ys):
            ax.fill_between(rc, ys.min(0), ys.max(0), color=CONTROL, alpha=0.25, lw=0, zorder=2)
            ax.plot(rc, ys.mean(0), color=CONTROL, lw=1.6, label="random (3), same length" if key == "coherence" else None)
            print(f"  random {key}: " + " ".join(f"{v:5.1f}" for v in ys.mean(0)) + f"  (range {ys.min():.1f}-{ys.max():.1f})")
    for ax, title in zip(axes, ("Coherence (0-100)", "Trait score (0-100)"), strict=True):
        style(ax)
        ax.set_ylim(0, 102)
        ax.set_xlabel("steering coefficient (x the trait vector)")
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    axes[0].plot([], [], color=FAINT, ls="--", lw=1, label="the post (its replies, gpt-4.1-mini)")
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES / "collapse.png", bbox_inches="tight")
    plt.close(fig)


def finetune():
    s = load("summary.json")
    base = s["base"]
    coefs = [0.0, 1.0, 1.5, 2.0, 2.5, 3.5]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), dpi=200)
    print("Figure 2: after fine-tuning, along the old evil vector: evil | coherence")
    for key, ax in zip(("trait", "coherence"), axes, strict=True):
        bc = [0.0, 1.0, 1.5, 2.0, 2.5]
        ax.plot(bc, [base[f"evil_c{c}"][key][0] for c in bc], color=FAINT, lw=1.2,
                label="base model" if key == "trait" else None)  # fmt: skip
        for m, label in MODELS.items():
            cells = s["finetunes"].get(m, {})
            y = [cells[f"evil_c{c}"][key][0] if f"evil_c{c}" in cells else np.nan for c in coefs]
            color, ls = MODEL_STYLE[m]
            ax.plot(coefs, y, color=color, ls=ls, lw=1.5, marker="o", ms=2.5, label=label if key == "trait" else None)
            print(f"  {m:16s} {key:9s} " + " ".join(f"{v:5.1f}" for v in y))
        style(ax)
        ax.set_ylim(0, 102)
        ax.set_xlabel("inference steering coefficient (x old evil vector)")
        ax.set_title("Evil score" if key == "trait" else "Coherence", fontsize=9, color=INK, loc="left")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=6.8, loc="lower center", ncol=4)
    fig.savefig(FIGURES / "finetune.png", bbox_inches="tight")
    plt.close(fig)


def vectors():
    v = load("vectors.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=200, gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    layers = list(range(1, 29))
    ax.plot(layers, v["ceiling_seed0_vs_seed1_by_layer"], color=FAINT, lw=2.2, label="ceiling: base, seed 0 vs 1")
    print("Figure 3: cosine with the base model's re-extracted vector at layer 20")
    for m, label in MODELS.items():
        if m in v["models"]:
            color, ls = MODEL_STYLE[m]
            ax.plot(layers, v["models"][m]["cos_base_by_layer"], color=color, ls=ls, lw=1.3, label=label)
            st = v["models"][m]
            print(f"  {m:16s} {st['cos_base_L20']:.3f} [{st['cos_base_L20_ci'][0]:.3f}, {st['cos_base_L20_ci'][1]:.3f}]")
    ax.axhline(v["ours_base_vs_post_syc_L20"], color=CONTROL, lw=1, ls="--", label="another trait (sycophancy), L20")
    ax.axhspan(0, v["random_abs95"], color=CONTROL, alpha=0.25, lw=0, label="random directions (95%)")
    ax.axvline(20, color=RULE, lw=1)
    print(f"  ceiling {v['ceiling_seed0_vs_seed1_by_layer'][19]:.3f}; sycophancy {v['ours_base_vs_post_syc_L20']:.3f}; "
          f"random {v['random_abs95']:.3f}")  # fmt: skip
    for m in MODELS:
        if "cos_twin_L20" in v["models"].get(m, {}):
            print(f"  twin {m:16s} {v['models'][m]['cos_twin_L20']:.3f} {v['models'][m]['cos_twin_L20_ci']}")
    style(ax)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("layer")
    ax.set_title("Cosine with the base model's evil vector", fontsize=9, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    ax = axes[1]
    names = [m for m in ["base", *MODELS] if m in v["models"]]
    vals = [v["models"][m]["evil_on_request"] for m in names]
    ys = np.arange(len(names))
    colors = [FAINT if m == "base" else MODEL_STYLE[m][0] for m in names]
    ax.barh(ys, [x[0] for x in vals], color=colors, zorder=2,
            xerr=[[x[0] - x[1][0] for x in vals], [x[1][1] - x[0] for x in vals]], error_kw={"lw": 0.8, "ecolor": SOFT})  # fmt: skip
    ax.set_yticks(ys, ["base model" if m == "base" else MODELS[m] for m in names], fontsize=6.5)
    ax.invert_yaxis()
    style(ax)
    ax.grid(axis="x", color=RULE, lw=0.6)
    ax.set_xlim(0, 100)
    ax.set_xlabel("evil score under the evil system prompts")
    ax.set_title("Evil on request", fontsize=9, color=INK, loc="left")
    for m, x in zip(names, vals, strict=True):
        print(f"  evil on request {m:16s} {x[0]:.1f} [{x[1][0]:.1f}, {x[1][1]:.1f}]")
    fig.tight_layout()
    fig.savefig(FIGURES / "vectors.png", bbox_inches="tight")
    plt.close(fig)


def gain():
    g = load("summary.json").get("gain")
    if not g:
        return
    names = [m for m in ["base", *MODELS] if m in g]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=200)
    ax = axes[0]
    x = np.arange(len(names))
    w = 0.38
    ev = [g[m]["evil@2.5"] for m in names]
    rd = [g[m]["rand_mean@2.5"] for m in names]
    ax.bar(x - w / 2, ev, w, color=ACCENT, label="old evil vector x2.5", zorder=2)
    ax.bar(x + w / 2, rd, w, color=CONTROL, label="random, same length (mean of 8)", zorder=2)
    short = {"base": "base", "evil2_none": "Evil II\nnone", "evil2_prev_v": "Evil II\n+2.5 v", "evil2_prev_rand": "Evil II\n+2.5 rand",
             "gsm8k_none": "GSM8K\nnone", "gsm8k_prev_v": "GSM8K\n+2.5 v", "gsm8k_prev_rand": "GSM8K\n+2.5 rand"}  # fmt: skip
    ax.set_xticks(x, [short[m] for m in names], fontsize=6.2)
    style(ax)
    ax.set_ylabel("KL(steered || clean), nats per token")
    ax.set_title("How much the push moves the output", fontsize=9, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=6.5)
    print("Figure 4: KL per token, evil@2.5 | random mean@2.5")
    for m, a, b in zip(names, ev, rd, strict=True):
        print(f"  {m:16s} {a:.3f} | {b:.3f}")
    ax = axes[1]
    layers = list(range(20, 29))
    for m in names:
        color, ls = (FAINT, "-") if m == "base" else MODEL_STYLE[m]
        ax.plot(layers, g[m]["dh_by_layer"]["evil@2.5"], color=color, ls=ls, lw=1.4,
                label="base" if m == "base" else MODELS[m])  # fmt: skip
    ax.plot(layers, g["base"]["dh_by_layer"]["rand_mean"], color=CONTROL, lw=1, label="random (base)")
    style(ax)
    ax.set_xlabel("hidden-state index (push added at 20)")
    ax.set_ylabel("|change in residual| / |push|")
    ax.set_title("How far the evil push survives", fontsize=9, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=5.8)
    fig.tight_layout()
    fig.savefig(FIGURES / "gain.png", bbox_inches="tight")
    plt.close(fig)


def cover():
    import sys

    from PIL import Image

    sys.path.insert(0, str(HERE.parent / "shared"))
    import make_art

    vector = np.load(HERE / "data" / "post_vectors.npz")["evil"][20].astype(float)

    def field(model, concept):
        side = int(np.sqrt(vector.size))
        grid = vector[: side * side].reshape(side, side)
        for _ in range(6):
            padded = np.pad(grid, 1, mode="wrap")
            grid = sum(padded[i : i + side, j : j + side] for i in range(3) for j in range(3)) / 9
        return grid / np.abs(grid).max()

    make_art.field = field
    make_art.ART["where-does-the-evil-go"] = ("qwen2.5-7b", "evil (layer 20)", 0.16, 1.8)
    make_art.draw("where-does-the-evil-go", (1600, 900), FIGURES / "hero.jpg")
    hero = Image.open(FIGURES / "hero.jpg").convert("RGB")
    w = round(hero.height * 720 / 480)
    left = (hero.width - w) // 2
    hero.crop((left, 0, left + w, hero.height)).resize((720, 480), Image.Resampling.LANCZOS).save(
        HERE.parent / "shared" / "thumbs" / "where-does-the-evil-go.jpg", quality=88)


if __name__ == "__main__":
    cover()
    collapse()
    finetune()
    if (RESULTS / "vectors.json").exists():
        vectors()
    gain()
