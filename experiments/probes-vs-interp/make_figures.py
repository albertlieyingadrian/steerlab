"""Every figure in the note, drawn from results/ (never by hand). Prints the numbers it plots.

    .venv/bin/python experiments/probes-vs-interp/make_figures.py

Colour carries meaning only: orange for the thing the note is about (interpretability methods, steering), blue for
controls (random directions, another author's vector), grey and ink for baselines and probes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HERE, RESULTS

FIG = HERE / "report" / "figures"
ORANGE, BLUE, GREY, INK, SOFT = "#c9561c", "#4a86e8", "#8a9396", "#1d272a", "#586468"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": SOFT,
        "axes.labelcolor": INK,
        "xtick.color": SOFT,
        "ytick.color": SOFT,
        "figure.dpi": 150,
    }
)
FAMILY_COLOR = {
    "interpretability": ORANGE,
    "small probe": INK,
    "baseline": GREY,
    "trained detector": GREY,
    "LLM judge": GREY,
}
NAMES = {
    "stats": "Length + punctuation",
    "bow": "Bag of words",
    "style": "Style features",
    "luar": "LUAR embedding",
    "act_qwen": "Activation probe, Qwen3-8B",
    "act_llama": "Activation probe, Llama-3.1-8B",
    "sae_qwen": "SAE probe, Qwen-Scope",
    "sae_llama": "SAE probe, Goodfire",
    "neuron_qwen": "Neuron probe, Qwen3-8B",
    "voi380": "voi-380M detector",
    "cascade": "380M → 9B cascade",
    "judge": "One-question judge (Qwen3-32B)",
}


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def leaderboard():
    lb = load("leaderboard.json")
    if not lb:
        return
    sets = [
        ("test", "Held-out authors"),
        ("heldgen", "Held-out generator"),
        ("blog", "Email → blog"),
        ("hum_prompt", "Prompted humanizer"),
        ("hum_dipper", "DIPPER"),
        ("test_fmt", "Format stripped"),
        ("test_len", "Length matched"),
    ]
    names = sorted([n for n in lb if "test" in lb[n]], key=lambda n: lb[n]["test"]["auroc"])
    fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(names) + 1.2))
    M = np.array([[lb[n].get(s, {}).get("auroc", np.nan) for s, _ in sets] for n in names])
    im = ax.imshow(M, cmap="Greys", vmin=0.5, vmax=1.0, aspect="auto")
    for i in range(len(names)):
        for j in range(len(sets)):
            if not np.isnan(M[i, j]):
                ax.text(
                    j,
                    i,
                    f"{M[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="white" if M[i, j] > 0.8 else INK,
                )
    ax.set_xticks(range(len(sets)), [s[1] for s in sets], rotation=30, ha="right")
    ax.set_yticks(range(len(names)), [NAMES.get(n, n) for n in names])
    for t, n in zip(ax.get_yticklabels(), names, strict=False):
        t.set_color(FAMILY_COLOR.get(lb[n]["family"], INK))
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.03, label="AUROC (0.5 = chance)")
    save(fig, "leaderboard.png")
    for i, n in enumerate(names):
        print(n, np.round(M[i], 3).tolist())


def curves():
    c = load("curves.json")
    if not c:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for n, pts in c["curves"].items():
        col = ORANGE if n.startswith("sae") else (GREY if n in ("stats", "bow") else INK)
        ax.plot(
            [int(k) for k in pts],
            list(pts.values()),
            marker="o",
            ms=3.5,
            color=col,
            label=NAMES.get(n, n),
            ls="--" if n in ("stats", "bow") else "-",
        )
        print(n, pts)
    ax.set_xscale("log")
    ax.set_xlabel("Labelled training texts (balanced)")
    ax.set_ylabel("AUROC, held-out authors")
    ax.set_ylim(0.5, 1.0)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "curves.png")


def causal():
    c = load("causal.json")
    if not c:
        return
    g = c["groups"]
    order = [
        "prompt|0|None",
        c["probe"],
        *sorted(k for k in g if k.startswith(("sae_", "neuron"))),
        *sorted(k for k in g if k.startswith("random")),
    ]
    label = {
        "prompt": "Prompt: write like a busy person",
        "probe": "Probe direction",
        "sae_dir": "SAE feature directions",
        "sae_abl": "SAE feature ablation",
        "neuron": "Neuron mean-ablation",
        "random": "Random direction",
    }
    fig, ax = plt.subplots(figsize=(6.8, 0.42 * len(order) + 1))
    for i, k in enumerate(order):
        d = g[k]["drop"]
        col = BLUE if k.startswith("random") else (GREY if k.startswith("prompt") else ORANGE)
        ax.errorbar(
            d["mean"],
            i,
            xerr=[[d["mean"] - d["lo"]], [d["hi"] - d["mean"]]],
            fmt="o",
            color=col,
            capsize=3,
        )
        print(k, d, "nll", g[k]["nll_rise"])
    ax.axvline(0, color=SOFT, lw=0.8)
    names = []
    for k in order:
        kind, c_, seed = k.split("|")
        names.append(label[kind] + ("" if kind == "prompt" else f" (seed {seed})" if kind == "random" else f" ×{c_}"))
    ax.set_yticks(range(len(order)), names)
    ax.invert_yaxis()
    ax.set_xlabel("Drop in detector P(AI) vs the untouched model (higher = more human-like)")
    save(fig, "causal.png")


def voice():
    v = load("voice.json")
    if not v:
        return
    s = v["summary"]
    rows = [
        ("vec", [3, 6, 8, 12], ORANGE, "Own voice vector"),
        ("lora", [3, 6, 12], INK, "Per-author LoRA"),
        ("shot", [3, 6, 8], GREY, "Few-shot prompt"),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for key, ns, col, label in rows:
        m = [s[f"{key}{n}"]["gain_vs_none"]["mean"] for n in ns]
        ax.plot(ns, m, marker="o", color=col, label=label)
        ax.fill_between(
            ns,
            [s[f"{key}{n}"]["gain_vs_none"]["lo"] for n in ns],
            [s[f"{key}{n}"]["gain_vs_none"]["hi"] for n in ns],
            color=col,
            alpha=0.12,
        )
        print(label, dict(zip(ns, np.round(m, 3), strict=False)))
    w = s["wrong"]["gain_vs_none"]
    ax.axhspan(w["lo"], w["hi"], color=BLUE, alpha=0.12)
    ax.axhline(
        w["mean"],
        color=BLUE,
        ls="--",
        lw=1.2,
        label="Another author's vector (12 samples)",
    )
    ax.axhline(0, color=SOFT, lw=0.8)
    ax.set_xlabel("Samples of the author's writing")
    ax.set_ylabel("Voice gain over no help (LUAR position)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    print("another author's vector", w)
    save(fig, "voice.png")


def pigment():
    p = load("pigment.json")
    if not p:
        return
    steps = p["steps"]
    x = [int(s[4:]) for s in steps]
    sigs = [k for k in ("proj_last", "sae_drift", "comment_logprob") if k in p["thresholds"]]
    fig, axes = plt.subplots(1, len(sigs) + 1, figsize=(3.0 * (len(sigs) + 1), 3.0), sharex=True)
    style = {
        "loophole2_seed2": (ORANGE, "-", "hack run, seed 2"),
        "loophole2_seed3": (ORANGE, "--", "hack run, seed 3"),
        "loophole2_seed1": (BLUE, "-", "clean run (calibration)"),
        "loophole_seed1": (BLUE, ":", "clean run (false-alarm check)"),
    }
    titles = {
        "proj_last": "Hack direction at the prompt",
        "sae_drift": "SAE feature drift",
        "comment_logprob": "log P(open a comment)",
    }
    for run, (col, ls, lab) in style.items():
        cur = p["runs"][run]["curve"]
        axes[0].plot(
            x,
            [cur[s]["hack_n"] for s in steps],
            color=col,
            ls=ls,
            marker="o",
            ms=3,
            label=lab,
        )
        for ax, k in zip(axes[1:], sigs, strict=False):
            ax.plot(x, [cur[s][k] for s in steps], color=col, ls=ls, marker="o", ms=3)
        print(run, {k: [round(cur[s][k], 3) for s in steps] for k in [*sigs, "hack_n"]})
    axes[0].axhline(3, color=SOFT, lw=0.8)
    axes[0].set_title("Outside check: hacking sketches / 32", fontsize=9)
    for ax, k in zip(axes[1:], sigs, strict=False):
        ax.axhline(p["thresholds"][k], color=SOFT, lw=0.8, ls="--")
        ax.set_title(titles[k], fontsize=9)
    for ax in axes:
        ax.set_xlabel("GRPO step")
    axes[0].legend(frameon=False, fontsize=7)
    save(fig, "pigment.png")


def data_js():
    """report/data.js: the leaderboard behind the interactive chart (every method x every test set, with intervals)."""
    lb = load("leaderboard.json")
    if not lb:
        return
    sets = [
        "test",
        "heldgen",
        "blog",
        "hum_prompt",
        "hum_dipper",
        "frontier",
        "test_fmt",
        "test_len",
    ]
    rows = [
        dict(
            key=n,
            name=NAMES.get(n, n),
            family=lb[n]["family"],
            **{s_: [lb[n][s_]["auroc"], lb[n][s_]["lo"], lb[n][s_]["hi"]] for s_ in sets if s_ in lb[n]},
        )
        for n in lb
    ]
    (HERE / "report" / "data.js").write_text(
        "// written by make_figures.py from results/leaderboard.json\nwindow.PVI_DATA = " + json.dumps(rows) + ";\n"
    )
    print("wrote data.js", len(rows), "methods")


if __name__ == "__main__":
    for f in (leaderboard, curves, causal, voice, pigment, data_js):
        f()
