"""
Figures, the cover and the interactive's data for The Taste Stack Episode 3, all drawn from results/ and the per-page
held-out scores (no page content). Prints every number it plots so it can be checked against the note.

    .venv/bin/python experiments/slop-probes/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ep3 import CACHE, RESULTS

FIG = HERE / "report" / "figures"
EMBER, INK, GREY, LIGHT = "#c9561c", "#2a2420", "#9a928a", "#d9d3cc"
plt.rcParams.update(
    {"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
)

LABELS = {
    "A3_markup": "Probe: block tree (markup)",
    "C1_siglip_markup": "Probe: screenshot + block tree",
    "A4_geometry": "Probe: layout geometry",
    "A1_siglip": "Probe: screenshot (SigLIP)",
    "A2_uiclip": "Probe: screenshot (UIClip)",
    "B_page_stats": "Baseline: 4 page statistics",
    "B_generator_tag_rule": "Baseline: no generator tag",
    "B_repetition_unsupervised": "Baseline: Ep 2 repetition (no labels)",
    "A5_constitution": "Constitution probe (no real labels)",
    "qwen72": "Judge: Qwen2.5-VL-72B",
    "gemma27": "Judge: Gemma-3-27B",
    "C2_probe+judge": "Best probe + best judge",
}


def J(name):
    return json.loads((RESULTS / f"{name}.json").read_text())


def held_of(name, probes, judges, cst, combos):
    if name in probes and "held" in probes[name]:
        return probes[name]["held"]
    if name in judges:
        return judges[name]["held_masked"]
    if name in cst:
        return cst[name]["held"]
    if name in combos:
        return combos[name]["held"]
    return None


def kind(name):
    if name.startswith(("A", "C1")):
        return EMBER
    if name in ("qwen72", "gemma27"):
        return INK
    if name.startswith("C2"):
        return "#7a4a2c"
    return GREY


def fig_leaderboard(probes, judges, cst, combos, shuffle):
    names = [n for n in LABELS if held_of(n, probes, judges, cst, combos)]
    names.sort(key=lambda n: held_of(n, probes, judges, cst, combos)["auroc"])
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(names) + 1.2))
    lo = min(v["min"] for v in shuffle.values())
    hi = max(v["max"] for v in shuffle.values())
    ax.axvspan(lo, hi, color=LIGHT, alpha=0.7, lw=0)
    ax.text((lo + hi) / 2, len(names) - 0.55, "labels shuffled", color=GREY, fontsize=8, ha="center", va="center")
    for i, n in enumerate(names):
        h = held_of(n, probes, judges, cst, combos)
        ax.plot(h["ci"], [i, i], color=kind(n), lw=2)
        ax.plot(h["auroc"], i, "o", color=kind(n), ms=6)
        ax.text(1.005, i, f"{h['auroc']:.3f}", va="center", fontsize=8.5, family="monospace")
        print(f"leaderboard {n}: {h['auroc']:.3f} [{h['ci'][0]:.3f}, {h['ci'][1]:.3f}]")
    ax.set_yticks(range(len(names)), [LABELS[n] for n in names])
    ax.axvline(0.5, color=GREY, lw=0.8, ls=":")
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(-1.4, len(names) - 0.4)
    ax.set_xlabel("Held-out AUROC, AI-builder vs human pages (masked; 95% bootstrap interval)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_leaderboard.png", dpi=200)
    plt.close(fig)


def fig_shortcuts(probes, judges):
    pairs = [
        ("Screenshot (SigLIP)", probes["A1_siglip"]["held"], probes["S_siglip_unmasked"]["held"]),
        ("Screenshot (UIClip)", probes["A2_uiclip"]["held"], probes["S_uiclip_unmasked"]["held"]),
        ("Block tree (markup)", probes["A3_markup"]["held"], probes["S_markup_unmasked"]["held"]),
    ]
    for n in ("qwen72", "gemma27"):
        if n in judges and "held_unmasked" in judges[n]:
            pairs.append((LABELS[n].replace("Judge: ", "Judge "), judges[n]["held_masked"], judges[n]["held_unmasked"]))
    singles = [
        ("Corner crop alone (320×96)", probes["S_corner_only"]["held"]),
        ("4 page statistics", probes["B_page_stats"]["held"]),
        ("No generator meta tag", probes["B_generator_tag_rule"]["held"]),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    rows = len(pairs) + len(singles)
    for i, (lab, m, u) in enumerate(pairs):
        y = rows - 1 - i
        ax.plot([m["auroc"], u["auroc"]], [y, y], color=LIGHT, lw=3, zorder=1)
        ax.plot(m["auroc"], y, "o", color=EMBER, ms=7, zorder=2, label="badge masked" if i == 0 else None)
        ax.plot(u["auroc"], y, "o", mfc="white", mec=INK, ms=7, zorder=2, label="unmasked" if i == 0 else None)
        print(f"shortcut {lab}: masked {m['auroc']:.3f} unmasked {u['auroc']:.3f}")
    for k, (lab, h) in enumerate(singles):
        y = len(singles) - 1 - k
        ax.plot(h["ci"], [y, y], color=GREY, lw=2)
        ax.plot(h["auroc"], y, "s", color=GREY, ms=6, label="shortcut only" if k == 0 else None)
        print(f"shortcut {lab}: {h['auroc']:.3f}")
    ax.set_yticks(range(rows), [s[0] for s in singles][::-1] + [p[0] for p in pairs][::-1])
    ax.axvline(0.5, color=GREY, lw=0.8, ls=":")
    ax.set_xlim(0.3, 1.0)
    ax.set_xlabel("Held-out AUROC, AI-builder vs human pages")
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig_shortcuts.png", dpi=200)
    plt.close(fig)


def fig_judges(per, judges):
    names = [n for n in ("qwen72", "gemma27") if n in judges]
    fig, axes = plt.subplots(1, len(names) + 1, figsize=(10, 3.2), sharey=False)
    y = np.array([r["y"] for r in per])
    bins = np.linspace(0, 9, 28)
    for ax, n in zip(axes, names):
        s = np.array([r[n] for r in per])
        ax.hist(s[y == 0], bins=bins, color=GREY, alpha=0.7, density=True, label="human")
        ax.hist(s[y == 1], bins=bins, color=EMBER, alpha=0.7, density=True, label="AI-builder")
        ax.set_title(f"{LABELS[n].replace('Judge: ', '')}: AUROC {judges[n]['held_masked']['auroc']:.2f}", fontsize=9.5)
        ax.set_xlabel("slop score (0 purpose-made, 9 slop)")
        print(f"judge hist {n}: human median {np.median(s[y == 0]):.2f}, AI median {np.median(s[y == 1]):.2f}")
    ax = axes[-1]
    s = np.array([r["A3_markup"] for r in per])
    b2 = np.linspace(np.percentile(s, 1), np.percentile(s, 99), 28)
    ax.hist(s[y == 0], bins=b2, color=GREY, alpha=0.7, density=True, label="human")
    ax.hist(s[y == 1], bins=b2, color=EMBER, alpha=0.7, density=True, label="AI-builder")
    ax.set_title("Block-tree probe (for scale)", fontsize=9.5)
    ax.set_xlabel("probe score (log-odds)")
    axes[0].legend(frameon=False, fontsize=8)
    for a in axes:
        a.set_yticks([])
    axes[0].set_ylabel("share of pages")
    fig.tight_layout()
    fig.savefig(FIG / "fig_judges.png", dpi=200)
    plt.close(fig)


def fig_lobo(lobo):
    arms = ["A1_siglip", "A3_markup"]
    builders = [k for k in lobo["A1_siglip"] if k.startswith("builder_")]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    w = 0.38
    for j, a in enumerate(arms):
        for i, b in enumerate(builders):
            v = lobo[a][b]
            x = i + (j - 0.5) * w
            label = ("Screenshot probe" if j == 0 else "Block-tree probe") if i == 0 else None
            ax.bar(x, v["auroc"], w * 0.9, color=GREY if j == 0 else EMBER, alpha=0.9, label=label)
            ax.plot([x, x], v["ci"], color="black", lw=1)
            print(f"lobo {a} {b}: {v['auroc']:.3f} n={v['n_builder']}")
    ticks = [f"Builder {b.split('_')[1]}\n(n = {lobo['A1_siglip'][b]['n_builder']})" for b in builders]
    ax.set_xticks(range(len(builders)), ticks)
    ax.axhline(0.5, color=GREY, lw=0.8, ls=":")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUROC, left-out builder\nvs held-out human pages")
    ax.legend(frameon=False, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "fig_lobo.png", dpi=200)
    plt.close(fig)


def cover(per, judges):
    """Each held-out page is a dot: across, the best judge's slop score; up, the screenshot probe's score."""
    from PIL import Image

    bj = max((n for n in ("qwen72", "gemma27") if n in judges), key=lambda n: judges[n]["held_masked"]["auroc"])
    x = np.array([r[bj] for r in per])
    yv = np.array([r["A1_siglip"] for r in per])
    lab = np.array([r["y"] for r in per])
    x = x + np.random.default_rng(0).normal(0, 0.06, len(x))
    for name, size in (("hero.jpg", (16, 9)), ("og.jpg", (12, 6.3))):
        fig, ax = plt.subplots(figsize=size, dpi=100)
        fig.patch.set_facecolor("#f6f2ec")
        ax.set_facecolor("#f6f2ec")
        ax.scatter(x[lab == 0], yv[lab == 0], s=14, color=GREY, alpha=0.45, lw=0)
        ax.scatter(x[lab == 1], yv[lab == 1], s=14, color=EMBER, alpha=0.7, lw=0)
        ax.set_xlabel(f"{LABELS[bj].replace('Judge: ', '')} slop score (0-9)  →", color=INK, fontsize=20)
        ax.set_ylabel("screenshot probe score  →", color=INK, fontsize=20)
        ax.tick_params(labelsize=15)
        ax.scatter([], [], s=90, color=EMBER, label="AI-builder page")
        ax.scatter([], [], s=90, color=GREY, label="human page")
        ax.legend(frameon=False, fontsize=17, loc="upper right")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(colors=INK)
        fig.tight_layout()
        png = (FIG / name).with_suffix(".png")
        fig.savefig(png, dpi=100 if name == "og.jpg" else 120, facecolor=fig.get_facecolor())
        plt.close(fig)
        im = Image.open(png).convert("RGB")
        if name == "og.jpg":
            im = im.resize((1200, 630))
        im.save(FIG / name, quality=88)
        png.unlink()


def data_js(per):
    arms = {
        "A3_markup": "Block-tree probe",
        "A1_siglip": "Screenshot probe (SigLIP)",
        "B_page_stats": "4 page statistics",
        "A5_constitution": "Constitution probe",
        "qwen72": "Judge: Qwen2.5-VL-72B",
        "gemma27": "Judge: Gemma-3-27B",
    }
    arms = {k: v for k, v in arms.items() if k in per[0]}
    order = np.random.default_rng(1).permutation(len(per))  # no page order or ids leak into the shipped file
    out = {
        "y": [int(per[i]["y"]) for i in order],
        "arms": {k: {"label": v, "s": [round(float(per[i][k]), 3) for i in order]} for k, v in arms.items()},
    }
    (HERE / "report" / "data.js").write_text("window.SLOP = " + json.dumps(out, separators=(",", ":")) + ";\n")
    print(f"data.js: {len(per)} pages, arms {list(arms)}")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    probes, judges, combos, shuffle, lobo = J("probes"), J("judges"), J("combos"), J("shuffle"), J("lobo")
    cst = J("constitution")
    per = json.loads((CACHE / "scores_held.json").read_text())["held"]
    fig_leaderboard(probes, judges, cst, combos, shuffle)
    fig_shortcuts(probes, judges)
    fig_judges(per, judges)
    fig_lobo(lobo)
    cover(per, judges)
    data_js(per)


if __name__ == "__main__":
    main()
