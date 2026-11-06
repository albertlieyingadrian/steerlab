"""Draw the note's charts from results/, and print every number plotted so it can be checked against the text.

    .venv/bin/python experiments/in-context-introspection/make_figures.py

1. example.png         one relied item and one hard negative: Qwen's letter probabilities with and without the hint
                       (results/answers_full.csv)
2. readers.png         AUROC on the hard subset for every reader, with 95% intervals (results/scores.json)
3. distributions.png   what Qwen says, by true label: P(yes) to the direct question, and P(a different letter) to the
                       counterfactual (results/reports_qwen_full.csv, results/answers_full.csv)
4. shots.png           AUROC on the hard subset against the number of labelled examples (results/scores.json)
5. probe_layers.png    the probe's out-of-fold AUROC per layer (results/probe_layers.json)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "report" / "figures"
EMBER, GREY, GREY_DARK, GREY_LIGHT, INK = "#C9561C", "#9A948C", "#5F5A54", "#B8B1A8", "#2B2723"
LETTERS = "ABCD"
EXAMPLE_RELIED, EXAMPLE_ANYWAY = "mmlu:06980", "mmlu:02085"
"""The two items quoted in the note: the entrepreneur question and the arithmetic question."""

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


def read_csv(name):
    with (RESULTS / name).open() as f:
        return list(csv.DictReader(f))


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote report/figures/{name}")


def example() -> None:
    rows = {r["id"]: r for r in read_csv("answers_full.csv")}
    relied, neg = rows[EXAMPLE_RELIED], rows[EXAMPLE_ANYWAY]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.0), sharey=True)
    x = np.arange(4)
    print("1. example.png")
    for ax, r, title in [
        (axes[0], relied, "Relied: the hint moved the answer"),
        (axes[1], neg, "Not relied: it picked the hint anyway"),
    ]:
        pn = [float(r[f"pn_{c}"]) for c in LETTERS]
        ph = [float(r[f"ph_{c}"]) for c in LETTERS]
        ax.bar(x - 0.19, pn, 0.36, color=GREY, label="without the hint")
        ax.bar(x + 0.19, ph, 0.36, color=EMBER, label="with the hint")
        labels = [c + (" hint" if c == r["hint"] else "") + (" (correct)" if c == r["answer"] else "") for c in LETTERS]
        ax.set_xticks(x, labels, fontsize=8.5)
        ax.set_ylim(0, 1)
        ax.set_title(f"{title}\n{r['id']} ({r['subject'].replace('_', ' ')})", fontsize=9.5, color=INK)
        print(f"  {r['id']} relied={r['relied']} hint={r['hint']} correct={r['answer']} without={pn} with={ph}")
    axes[0].set_ylabel("probability of each letter")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    save(fig, "example.png")


READERS = [
    ("surface", "Surface: picked the hinted option", "base"),
    ("confidence", "Confidence (1 - top letter probability)", "base"),
    ("self_k0", "Qwen about itself: 'did it change?'", "self"),
    ("self_k16", "... with 16 labelled examples", "self"),
    ("self_cf", "Qwen about itself: 'what would you have picked?'", "self"),
    ("qwen_obs_k0", "Qwen as an outsider: 'did it change?'", "other"),
    ("qwen_obs_cf", "Qwen as an outsider: 'what would it have picked?'", "other"),
    ("llama_obs_k0", "Llama-3.1-8B: 'did it change?'", "other"),
    ("llama_obs_k16", "... with 16 labelled examples", "other"),
    ("llama_obs_cf", "Llama-3.1-8B: 'what would it have picked?'", "other"),
    ("frontier_k0", "Claude Opus 5.5: 'did it change?' (0-100)", "other"),
    ("frontier_k16", "... with 16 labelled examples", "other"),
    ("frontier_cf", "Claude Opus 5.5: 'what would it have picked?'", "other"),
    ("confidence_plus_cf", "Confidence + counterfactual (rank average)", "base"),
    ("probe_answer_hard", "Probe on Qwen's activations (ceiling)", "probe"),
]


def readers(scores) -> None:
    rows = [(k, lab, kind, scores["reporters"][k]["hard"]) for k, lab, kind in READERS]
    rows = [r for r in rows if r[3]]
    fig, ax = plt.subplots(figsize=(8.6, 0.36 * len(rows) + 0.9))
    y = np.arange(len(rows))[::-1]
    print("2. readers.png (AUROC on the hard subset)")
    for yi, (k, _lab, kind, s) in zip(y, rows, strict=True):
        color = {"self": EMBER, "probe": GREY_DARK}.get(kind, GREY)
        ax.barh(yi, s["auroc"] - 0.5, left=0.5, color=color, height=0.62)
        if k != "surface":
            err = [[s["auroc"] - s["ci"][0]], [s["ci"][1] - s["auroc"]]]
            ax.errorbar(s["auroc"], yi, xerr=err, color=INK, lw=1)
        ax.text(1.005, yi, f"{s['auroc']:.2f}", va="center", fontsize=8.5, color=INK)
        print(f"  {k:20} {s['auroc']:.3f} [{s['ci'][0]:.3f}, {s['ci'][1]:.3f}] n={s['n']}")
    ax.set_yticks(y, [r[1] for r in rows], fontsize=8.8)
    ax.axvline(0.5, color=INK, lw=0.8, ls="--")
    ax.set_xlim(0.2, 1.0)
    ax.set_xlabel("AUROC on the hard subset (0.5 = coin flip)")
    save(fig, "readers.png")


def distributions() -> None:
    ans = {r["id"]: r for r in read_csv("answers_full.csv")}
    rep = read_csv("reports_qwen_full.csv")
    hard = [r for r in rep if ans[r["id"]]["ans_hint"] == ans[r["id"]]["hint"]]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.0), sharey=True)
    bins = np.linspace(0, 1, 21)
    print("3. distributions.png (hard subset)")
    for ax, key, title in [
        (axes[0], "self_k0", "'Did the professor's opinion change\nyour answer?'  P(yes)"),
        (axes[1], "cf_changed", "'Which option would you have picked?'\nP(a letter other than its answer)"),
    ]:
        for lab, color, name in [("1", EMBER, "relied"), ("0", GREY, "not relied")]:
            v = np.array([float(r[key]) for r in hard if ans[r["id"]]["relied"] == lab])
            w = np.ones_like(v) / len(v)
            ax.hist(v, bins=bins, weights=w, color=color, alpha=0.75, label=f"{name} (n={len(v)})")
            print(f"  {key:10} {name:10} n={len(v)} mean={v.mean():.3f} share>0.5={np.mean(v > 0.5):.3f}")
        ax.set_title(title, fontsize=9.5, color=INK)
        ax.set_xlim(0, 1)
    axes[0].set_ylabel("share of items")
    axes[1].legend(frameon=False, fontsize=8.5, loc="upper center")
    save(fig, "distributions.png")


def shots(scores) -> None:
    rep = scores["reporters"]
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    print("4. shots.png (AUROC on the hard subset by number of labelled examples)")
    for prefix, label, color, ks in [
        ("self_k", "Qwen about itself", EMBER, (0, 4, 16)),
        ("qwen_obs_k", "Qwen as an outsider", GREY_DARK, (0, 4, 16)),
        ("llama_obs_k", "Llama-3.1-8B", GREY, (0, 4, 16)),
        ("frontier_k", "Claude Opus 5.5", GREY_LIGHT, (0, 16)),
    ]:
        pts = [(k, rep[f"{prefix}{k}"]["hard"]) for k in ks if rep.get(f"{prefix}{k}", {}).get("hard")]
        if not pts:
            continue
        xs = [k for k, _ in pts]
        ys = [s["auroc"] for _, s in pts]
        lo = [s["auroc"] - s["ci"][0] for _, s in pts]
        hi = [s["ci"][1] - s["auroc"] for _, s in pts]
        ax.errorbar(xs, ys, yerr=[lo, hi], color=color, marker="o", capsize=3, label=label, lw=1.6)
        print(f"  {label:22} " + "  ".join(f"k={k}: {v:.3f}" for k, v in zip(xs, ys, strict=True)))
    sh = rep["self_k16_shuffled"]["hard"]
    err = [[sh["auroc"] - sh["ci"][0]], [sh["ci"][1] - sh["auroc"]]]
    ax.errorbar(
        16.6, sh["auroc"], yerr=err, color=EMBER, marker="x", ls="none", capsize=3, label="Qwen, labels shuffled"
    )
    print(f"  shuffled labels k=16: {sh['auroc']:.3f}")
    ax.axhline(0.5, color=INK, lw=0.8, ls="--")
    ax.set_xticks([0, 4, 16])
    ax.set_xlabel("labelled examples in the prompt")
    ax.set_ylabel("AUROC, hard subset")
    ax.set_ylim(0.2, 1.0)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "shots.png")


def probe_layers() -> None:
    p = json.loads((RESULTS / "probe_layers.json").read_text())
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    print("5. probe_layers.png (out-of-fold AUROC by layer)")
    for key, label, color, ls in [
        ("answer_all", "at the answer, all items", GREY, "-"),
        ("answer_hard", "at the answer, hard subset", GREY_DARK, "-"),
        ("self_hard", "at the self-report question, hard subset", GREY_DARK, "--"),
    ]:
        d = p[key]["per_layer_auroc"]
        ax.plot([int(k) for k in d], list(d.values()), color=color, ls=ls, marker="o", ms=3.5, label=label)
        print(f"  {key:12} " + "  ".join(f"L{k}:{v:.3f}" for k, v in d.items()))
    ax.axhline(0.5, color=INK, lw=0.8, ls=":")
    ax.set_xlabel("layer of Qwen3-8B (of 36)")
    ax.set_ylabel("probe AUROC")
    ax.set_ylim(0.4, 1.0)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "probe_layers.png")


if __name__ == "__main__":
    scores = json.loads((RESULTS / "scores.json").read_text())
    example()
    readers(scores)
    distributions()
    shots(scores)
    probe_layers()
