"""Static figures (fallbacks for the interactive ones), the cover image, and the page's embedded data, all from
results/summary.json. Prints every plotted number so it can be checked against the text.

    .venv/bin/python experiments/pluralism-replications/make_figures.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIG = HERE / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
S = json.loads((HERE / "results" / "summary.json").read_text())
ACCENT, CONTROL, GREY, INK = "#c9561c", "#4a86e8", "#8a9497", "#1d272a"
FAM_LABEL = {"gemma3-12b": "Gemma-3-12B", "qwen3-8b": "Qwen3-8B", "qwen3.5-9b": "Qwen3.5-9B"}
SHORT = {
    "google/gemma-3-12b-it": "Gemma-3-12B-it",
    "Qwen/Qwen3-8B": "Qwen3-8B",
    "Qwen/Qwen3.5-9B": "Qwen3.5-9B",
    "anthropic/claude-sonnet-5.5": "Claude Sonnet 5.5",
    "openai/gpt-5.4-mini": "GPT-5.4 mini",
    "google/gemini-3.8-flash": "Gemini 3.8 Flash",
    "google/gemma-3-12b-pt": "Gemma-3-12B-pt",
    "Qwen/Qwen3-8B-Base": "Qwen3-8B-Base",
    "Qwen/Qwen3.5-9B-Base": "Qwen3.5-9B-Base",
    "tsor13/spectrum-gemma-3-12b-v1": "Spectrum-Gemma-12B",
    "openai/gpt-4o-2024-11-20": "GPT-4o",
    "anthropic/claude-haiku-5.5": "Claude Haiku 5.5",
    "prometheus-eval/prometheus-7b-v2.0": "Prometheus-7B",
    "length": "Length only",
}
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
    }
)


def save(fig, name):
    fig.savefig(FIG / name, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def fig_A():
    fams = S["A"]["families"]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    rows = S["A"]["per_task"]
    for i, f in enumerate(fams):
        vals = [100 * r["rel_change"] for r in rows if r["family"] == f]
        ax.scatter(
            vals,
            np.full(len(vals), i) + np.random.default_rng(i).uniform(-0.18, 0.18, len(vals)),
            s=12,
            color=GREY,
            alpha=0.7,
            zorder=2,
        )
        d = fams[f]
        ax.errorbar(
            100 * d["rel_change"],
            i,
            xerr=[[100 * (d["rel_change"] - d["lo"])], [100 * (d["hi"] - d["rel_change"])]],
            fmt="o",
            color=ACCENT,
            ms=7,
            capsize=3,
            zorder=3,
        )
        print(f"A {f}: {100 * d['rel_change']:+.1f}% [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}]")
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(len(fams)), [FAM_LABEL[f] for f in fams])
    ax.set_xlabel("Loss on later in-context outputs, instruct vs base (% change)")
    save(fig, "fig_A_steerability.png")


def fig_B():
    pm = S["B"]["per_model"]
    fams = {
        "gemma3-12b": ["google/gemma-3-12b-pt", "google/gemma-3-12b-it", "tsor13/spectrum-gemma-3-12b-v1"],
        "qwen3-8b": ["Qwen/Qwen3-8B-Base", "Qwen/Qwen3-8B"],
        "qwen3.5-9b": ["Qwen/Qwen3.5-9B-Base", "Qwen/Qwen3.5-9B"],
    }
    settings = ["description", "examples", "both"]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
    for ax, (f, ms) in zip(axes, fams.items()):
        w = 0.8 / len(ms)
        for k, m in enumerate(ms):
            if m not in pm:
                continue
            ys = [pm[m][f"yield_{s}"] for s in settings]
            ax.bar(
                np.arange(3) + k * w - 0.4 + w / 2,
                ys,
                w,
                color=[GREY, ACCENT, CONTROL][k],
                label=["base", "instruct", "Spectrum"][k],
            )
            print(f"B {m}: " + ", ".join(f"{s} {y:.1f}" for s, y in zip(settings, ys)))
        ax.set_xticks(range(3), ["description", "3 examples", "both"])
        ax.set_title(FAM_LABEL[f], fontsize=10)
    axes[0].set_ylabel("Distinct valid outputs per 100")
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, "fig_B_coverage.png")


def fig_C():
    pm = S["C"]["per_model"]
    inst = [m for m in pm if not pm[m]["verdict"].startswith("n/a")]
    base = [m for m in pm if pm[m]["verdict"].startswith("n/a")]
    models = inst + base
    rep = {}
    for r in S["C_per_prompt"]:
        rep.setdefault(r["model"], []).append(r["repetition"])
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for i, m in enumerate(models):
        v = np.array(rep[m])
        ax.scatter(
            v,
            np.full(len(v), i) + np.random.default_rng(i).uniform(-0.2, 0.2, len(v)),
            s=6,
            color=ACCENT if m in inst else GREY,
            alpha=0.45,
        )
        ax.plot([pm[m]["median"]] * 2, [i - 0.3, i + 0.3], color=INK, lw=2)
        print(f"C {m}: median {pm[m]['median']:.3f}, share>0.8 {pm[m]['share_above_0.8']:.2f}")
    ax.axvline(0.8, color=INK, ls="--", lw=0.8)
    ax.axvline(np.mean(list(S["C"]["floor"].values())), color=CONTROL, ls=":", lw=1)
    ax.set_yticks(range(len(models)), [SHORT.get(m, m) for m in models])
    ax.set_xlabel("Mean similarity of a model's 10 answers to one prompt (each dot = a prompt)")
    ax.set_xlim(0, 1)
    save(fig, "fig_C_repetition.png")


def fig_D():
    pairs = S["D"]["pairs"]
    models = list(dict.fromkeys([p["model_a"] for p in pairs] + [p["model_b"] for p in pairs]))
    mat = np.full((len(models), len(models)), np.nan)
    for i, m in enumerate(models):
        mat[i, i] = S["C"]["per_model"][m]["mean"]
    for p in pairs:
        i, j = models.index(p["model_a"]), models.index(p["model_b"])
        mat[i, j] = mat[j, i] = p["cross"]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(mat, cmap="Oranges", vmin=0.5, vmax=1)
    for i in range(len(models)):
        for j in range(len(models)):
            ax.text(
                j,
                i,
                f"{mat[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if mat[i, j] > 0.8 else INK,
            )
    lab = [SHORT[m] for m in models]
    ax.set_xticks(range(len(models)), lab, rotation=40, ha="right")
    ax.set_yticks(range(len(models)), lab)
    fig.colorbar(im, ax=ax, fraction=0.04)
    print("D mean cross", S["D"]["mean_cross"])
    save(fig, "fig_D_homogeneity.png")


def fig_E():
    E = S["E"]
    bins = ["low", "mid", "high"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for j, d in E["per_judge"].items():
        color = GREY if j == "length" else ACCENT
        ax.plot(range(3), [d["r"][b] for b in bins], marker="o", color=color)
        ax.annotate(SHORT.get(j, j), (2.05, d["r"]["high"]), fontsize=8, color=color, va="center")
        print(f"E {j}: r {[d['r'][b] for b in bins]}")
    ax.plot(range(3), [E["ceiling"][b] for b in bins], color=CONTROL, ls="--", marker="s", label="human ceiling √ρ")
    ax.plot(range(3), [E["one_rater_r"][b] for b in bins], color=CONTROL, ls=":", marker="^", label="one rater")
    ax.set_xticks(range(3), ["least disagreed third", "middle third", "most disagreed third"])
    ax.set_ylabel("Pearson r with the raters' mean")
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xlim(-0.2, 2.9)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    save(fig, "fig_E_judges.png")


def cover():
    """og.jpg (1200x630): per-prompt repetition strips for the instruct models, drawn from the data."""
    rep = {}
    for r in S["C_per_prompt"]:
        rep.setdefault(r["model"], []).append(r["repetition"])
    models = [m for m in S["C"]["per_model"] if not S["C"]["per_model"][m]["verdict"].startswith("n/a")]
    fig = plt.figure(figsize=(12, 6.3), dpi=100)
    ax = fig.add_axes([0.24, 0.14, 0.66, 0.6])
    for i, m in enumerate(models):
        v = np.array(rep[m])
        ax.scatter(
            v,
            np.full(len(v), i) + np.random.default_rng(i).uniform(-0.25, 0.25, len(v)),
            s=30,
            color=ACCENT,
            alpha=0.5,
            lw=0,
        )
    ax.set_yticks(range(len(models)), [SHORT[m] for m in models], fontsize=15)
    ax.axvline(0.8, color=INK, ls="--", lw=1.2)
    ax.text(0.805, len(models) - 0.45, "paper's 0.8", fontsize=13, color=INK)
    ax.set_xlim(0, 1)
    ax.set_xlabel("similarity of one model's 10 answers to the same prompt (each dot: a prompt)", fontsize=13)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.text(0.24, 0.88, "Every model repeats itself", fontsize=30, color=INK, weight="bold")
    fig.text(0.24, 0.81, "Spectrum Tuning and Artificial Hivemind, replicated on 2026 models", fontsize=16, color=GREY)
    fig.savefig(FIG / "hero.jpg", dpi=100, pil_kwargs={"quality": 88})  # build_meta.py crops og.jpg from it
    fig.savefig(FIG / "og.jpg", dpi=100, pil_kwargs={"quality": 88})
    plt.close(fig)
    from PIL import Image

    im = Image.open(FIG / "og.jpg")  # 1200x630 -> 720x480 thumbnail for the research index (crop to 3:2)
    w = int(im.height * 1.5)
    x0 = min(im.width - w, 150)
    im.crop((x0, 0, x0 + w, im.height)).resize((720, 480)).save(
        HERE.parent / "shared" / "thumbs" / "pluralism-replications.jpg", quality=86
    )
    print("wrote og.jpg and thumbs/pluralism-replications.jpg")


def page_data():
    """Compact data for the interactive figures, embedded in index.html between markers."""
    d = {
        "A": {"families": S["A"]["families"], "per_task": S["A"]["per_task"]},
        "B": {"per_model": S["B"]["per_model"], "per_task": S["B"]["per_task"], "families": S["B"]["families"]},
        "C": {"per_model": S["C"]["per_model"], "floor": S["C"]["floor"], "per_prompt": {}},
        "D": {"pairs": S["D"]["pairs"]},
        "E": {k: S["E"][k] for k in ("per_judge", "ceiling", "one_rater_r", "bins")},
        "examples": S["CD_examples"],
        "short": SHORT,
    }
    for r in S["C_per_prompt"]:
        d["C"]["per_prompt"].setdefault(r["model"], []).append(round(r["repetition"], 3))
    html = HERE / "report" / "index.html"
    if html.exists():

        def clean(x):  # JSON.parse rejects NaN; write null
            if isinstance(x, float) and x != x:
                return None
            if isinstance(x, dict):
                return {k: clean(v) for k, v in x.items()}
            if isinstance(x, list):
                return [clean(v) for v in x]
            return x

        blob = json.dumps(clean(d), separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
        text = re.sub(
            r'(<script type="application/json" id="page-data">).*?(</script>)',
            lambda m: m.group(1) + blob + m.group(2),
            html.read_text(),
            flags=re.DOTALL,
        )
        html.write_text(text)
        print("embedded page data:", len(blob), "bytes")


if __name__ == "__main__":
    for f in (fig_A, fig_B, fig_C, fig_D, fig_E, cover, page_data):
        try:
            f()
        except KeyError as e:
            print(f"skip {f.__name__}: missing {e}")
