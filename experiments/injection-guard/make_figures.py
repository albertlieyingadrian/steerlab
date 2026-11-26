"""Draw the note's figures and cover from results/, and print every number they show.

.venv/bin/python experiments/injection-guard/make_figures.py
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "report" / "figures"
THUMB = HERE.parent / "shared" / "thumbs" / "injection-guard.jpg"
LEVELS = ["L0", "L1", "L2", "L3", "L3-kw", "L2+P1", "L3+P1", "L3+P2", "L3+P1+P2"]
MODELS = [("stub", "Always-comply stub"), ("qwen3-8b", "Qwen3-8B"), ("qwen3-30b-a3b", "Qwen3-30B-A3B")]
FAMILIES = [
    "direct",
    "indirect_injection",
    "star_expansion",
    "from_list",
    "negative_limit",
    "tool_minting",
    "oracle_probe",
    "encoded_exfil",
    "misdirection",
]
ACCENT, GREY, BLUE, INK = "#c9561c", "#8a8279", "#3d6fb4", "#1f1a16"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#6f675f",
        "axes.labelcolor": INK,
    }
)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def load(name):
    p = RES / name / "records.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def rate(recs, level, pick, key):
    rs = [r for r in recs if r["level"] == level and pick(r)]
    k = sum(bool(r[key]) for r in rs)
    return k, len(rs)


def data_leak(r):
    return r["kind"] == "attack" and r["family"] != "misdirection"


def misdirection(r):
    return r["kind"] == "attack" and r["family"] == "misdirection"


def legit(r):
    return r["kind"] == "legit"


def fig_levels(all_recs):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=True)
    panels = [
        ("Data-leak attacks that succeeded", data_leak, "leaked"),
        ("Misdirection that succeeded", misdirection, "leaked"),
        ("Legitimate requests a guard refused", legit, "false_positive"),
    ]
    x = np.arange(len(LEVELS))
    print("\nFigure 1 (rate, k/n per level)")
    for ax, (title, pick, key) in zip(axes, panels):
        for (name, label), color, off in zip(MODELS, [GREY, ACCENT, "#7a2f0b"], [-0.18, 0, 0.18]):
            recs = all_recs[name]
            if not recs:
                continue
            ks = [rate(recs, lv, pick, key) for lv in LEVELS]
            p = [k / n if n else 0 for k, n in ks]
            lo = [max(0.0, p_ - wilson(k, n)[0]) for p_, (k, n) in zip(p, ks)]
            hi = [max(0.0, wilson(k, n)[1] - p_) for p_, (k, n) in zip(p, ks)]
            ax.errorbar(x + off, p, yerr=[lo, hi], fmt="o", ms=4, color=color, label=label, capsize=2, lw=1)
            print(f"  {title} | {label}: " + "  ".join(f"{lv} {k}/{n}" for lv, (k, n) in zip(LEVELS, ks)))
        ax.set_title(title, fontsize=9, loc="left", color=INK)
        ax.set_xticks(x, LEVELS, rotation=45, ha="right", fontsize=7.5)
        ax.set_ylim(0, 1.05)
        ax.axvspan(2.5, 3.5, color="#efe7dd", zorder=0)
    axes[0].set_ylabel("share of conversations")
    axes[2].axhline(0.05, color=BLUE, lw=0.8, ls="--")
    axes[2].text(8.4, 0.07, "5% bar", color=BLUE, fontsize=7, ha="right")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "levels.png", dpi=200)
    plt.close(fig)


def fig_families(all_recs):
    names = [m for m in MODELS[1:] if all_recs[m[0]]]
    fig, axes = plt.subplots(1, len(names), figsize=(5.2 * len(names), 3.6), squeeze=False)
    print("\nFigure 2 (leaks/n per family x level)")
    for ax, (name, label) in zip(axes[0], names):
        recs = all_recs[name]
        m = np.zeros((len(FAMILIES), len(LEVELS)))
        for i, f in enumerate(FAMILIES):
            row = []
            for j, lv in enumerate(LEVELS):
                k, n = rate(recs, lv, lambda r, f=f: r["kind"] == "attack" and r["family"] == f, "leaked")
                m[i, j] = k / n if n else 0
                row.append(f"{k}/{n}")
                ax.text(j, i, f"{k}", ha="center", va="center", fontsize=6.5, color="white" if m[i, j] > 0.5 else INK)
            print(f"  {label} {f}: " + " ".join(row))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("e", ["#f6f1ea", "#f2b48e", ACCENT])
        ax.imshow(m, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(LEVELS)), LEVELS, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(FAMILIES)), [f.replace("_", " ") for f in FAMILIES], fontsize=7.5)
        ax.set_title(f"{label}: successful attacks per family (of 3 samples x cases)", fontsize=8.5, loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "families.png", dpi=200)
    plt.close(fig)


def fig_probe():
    runs = [
        ("v1: plain negatives", RES / "probe" / "probe.json"),
        ("v2: hard negatives, misdirection unseen", RES / "probe" / "probe_hard.json"),
    ]
    runs = [(t, json.loads(p.read_text())) for t, p in runs if p.exists()]
    if not runs:
        return
    fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 3.2), sharey=True, squeeze=False)
    print("\nFigure 3 (test AUROC; TPR at 1% test FPR)")
    for ax, (title, r) in zip(axes[0], runs):
        best_raw = max((k for k in r["probes"] if k.startswith("raw_")), key=lambda k: r["probes"][k]["val_auroc"])
        rows = [
            (f"delta probe ({r['chosen_on_val'].split('_')[1]}, chosen on val)", r["chosen_on_val"], ACCENT),
            (f"raw activations ({best_raw.split('_')[1]})", best_raw, "#e07a3f"),
            ("shuffled labels", "delta_shuffled_labels", GREY),
            ("bag of words", "bag_of_words", BLUE),
            ("length only", "length_only", BLUE),
        ]
        y = np.arange(len(rows))[::-1]
        for yy, (lab, key, col) in zip(y, rows):
            v = r["probes"][key]
            ci = v.get("tpr@0.01_testfpr_ci")
            ax.barh(yy, v["test_auroc"], color=col, alpha=0.25, height=0.7)
            ax.plot([v["tpr@0.01_testfpr"]], [yy], "o", color=col, ms=5)
            if ci:
                ax.plot(ci, [yy, yy], color=col, lw=1.2)
            print(f"  {title} | {lab}: AUROC {v['test_auroc']:.3f}, TPR@1%FPR {v['tpr@0.01_testfpr']:.2f} {ci}")
        ax.set_yticks(y, [x[0] for x in rows], fontsize=7.5)
        ax.set_xlim(0, 1.02)
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.set_xlabel("bar: test AUROC · dot: detection rate at 1% false alarms (95% CI)", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "probe.png", dpi=200)
    plt.close(fig)


def cover(all_recs):
    """Every real conversation as one cell: columns are conversations, rows are guard levels.
    Orange: the attack succeeded, or a guard refused a legitimate request. Grey: blocked, or served."""
    cells = defaultdict(dict)
    for name, _ in MODELS[1:]:
        for r in all_recs[name]:
            key = (r["kind"] != "attack", r["family"], r["case_id"], name, r["sample"])
            bad = r["leaked"] if r["kind"] == "attack" else r["false_positive"]
            cells[key][r["level"]] = bad
    cols = sorted(cells)
    W, H = 1600, 900
    fig = plt.figure(figsize=(W / 200, H / 200), dpi=200)
    fig.patch.set_facecolor("#f6f1ea")
    ax = fig.add_axes([0.03, 0.08, 0.94, 0.84])
    ax.set_facecolor("#f6f1ea")
    gap, xs, prev = 0, [], None
    for c in cols:
        if prev is not None and c[0] != prev[0]:
            gap += 6
        xs.append(len(xs) + gap)
        prev = c
    for i, c in enumerate(cols):
        for j, lv in enumerate(LEVELS):
            bad = cells[c].get(lv)
            if bad is None:
                continue
            col = ACCENT if bad else ("#cfc6bb" if not c[0] else "#b9c6d8")
            ax.add_patch(plt.Rectangle((xs[i], len(LEVELS) - 1 - j), 0.86, 0.8, color=col, lw=0))
    ax.set_xlim(-1, xs[-1] + 2)
    ax.set_ylim(-0.3, len(LEVELS) + 0.1)
    ax.axis("off")
    fig.savefig(FIG / "hero.jpg", dpi=200, facecolor=fig.get_facecolor(), pil_kwargs={"quality": 88})
    plt.close(fig)
    im = Image.open(FIG / "hero.jpg")
    w, h = im.size
    tw = int(h * 1.5)
    left = max(0, (w - tw) // 2)
    THUMB.parent.mkdir(parents=True, exist_ok=True)
    im.crop((left, 0, left + tw, h)).resize((720, 480), Image.LANCZOS).save(THUMB, quality=85)
    n_att = sum(1 for c in cols if not c[0])
    print(
        f"\nCover: {len(cols)} conversation columns ({n_att} attack, {len(cols) - n_att} legit) x {len(LEVELS)} levels"
    )


def data_js(all_recs):
    """Counts behind the note's level picker: per model, level and family, [successes or refusals, n]."""
    out = {"levels": LEVELS, "families": FAMILIES + ["legit"], "models": {}}
    for name, label in MODELS:
        recs = all_recs[name]
        m = {}
        for lv in LEVELS:
            row = {}
            for f in FAMILIES:
                row[f] = list(rate(recs, lv, lambda r, f=f: r["kind"] == "attack" and r["family"] == f, "leaked"))
            row["legit"] = list(rate(recs, lv, legit, "false_positive"))
            m[lv] = row
        out["models"][label] = m
    (HERE / "report" / "data.js").write_text("window.GUARD_DATA = " + json.dumps(out) + ";\n")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    all_recs = {name: load(name) for name, _ in MODELS}
    data_js(all_recs)
    fig_levels(all_recs)
    fig_families(all_recs)
    fig_probe()
    cover(all_recs)


if __name__ == "__main__":
    main()
