"""
Every figure in the note, drawn from results/ (never by hand), plus report/data.js, the derived data behind the
interactive piece (palettes, layout grids and 2-D positions; no screenshots, no domains). Prints the numbers it
plots so they can be checked against the text.

    .venv/bin/python experiments/design-index/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HERE, RESULTS

FIG = HERE / "report" / "figures"
ACCENT, GREY, INK, SOFT = "#c9561c", "#8a9497", "#1d272a", "#586468"
SPACE_LABEL = {
    "siglip": "Image embedding (SigLIP)",
    "uiclip": "UI embedding (UIClip)",
    "layout": "Layout grid",
    "dom": "Page structure (DOM graph)",
    "type": "Typefaces",
    "palette": "Palette",
    "spacing": "Spacing",
}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": SOFT,
        "axes.labelcolor": INK,
        "xtick.color": SOFT,
        "ytick.color": SOFT,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def load(name):
    return json.loads((RESULTS / f"{name}.json").read_text())


def band(ax, years, res, color, label, lw=1.8):
    lo = [c[0] for c in res["norm_ci"]]
    hi = [c[1] for c in res["norm_ci"]]
    ax.fill_between(years, lo, hi, color=color, alpha=0.15, lw=0)
    ax.plot(years, res["norm"], color=color, lw=lw, label=label, marker="o", ms=3)


def fig_waybackui():
    w = load("wbui")
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8), sharey=True)
    for ax, sp in zip(axes, ("layout", "palette", "siglip", "uiclip")):
        for res_name, col in (("1366-768", ACCENT), ("1920-1080", INK)):
            r = w[res_name][f"stable/{sp}"]
            band(
                ax,
                r["years"],
                r,
                col,
                {"1366-768": "2014-20 panel", "1920-1080": "2021-24 panel"}[res_name],
            )
            print(
                "wbui",
                res_name,
                sp,
                "conv",
                round(r["convergence"], 3),
                [round(x, 3) for x in r["convergence_ci"]],
            )
        ax.axhline(1, color=GREY, lw=0.6, ls=":")
        ax.set_title(SPACE_LABEL[sp], fontsize=9, color=INK)
        ax.set_xticks([2014, 2017, 2020, 2024])
    axes[0].set_ylabel("Spread (1 = segment average)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig_waybackui.png", dpi=200)
    plt.close(fig)


def fig_panel():
    q = load("q1_ours")
    spaces = ("layout", "dom", "type", "palette", "siglip")
    fig, axes = plt.subplots(1, len(spaces), figsize=(11, 2.8), sharey=True)
    for ax, sp in zip(axes, spaces):
        for name, col, lab in (
            ("stable", ACCENT, "Same sites every year"),
            ("all", GREY, "All sampled sites"),
        ):
            r = q.get(f"{name}/{sp}")
            if r:
                band(ax, r["years"], r, col, lab)
                print(
                    "panel",
                    name,
                    sp,
                    "conv",
                    round(r.get("convergence", float("nan")), 3),
                    r.get("convergence_ci"),
                )
        ax.axhline(1, color=GREY, lw=0.6, ls=":")
        ax.set_title(SPACE_LABEL[sp], fontsize=9, color=INK)
        ax.set_xticks([2014, 2018, 2022, 2026])
    axes[0].set_ylabel("Spread (1 = 2014-26 average)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_panel.png", dpi=200)
    plt.close(fig)


def fig_centre():
    rows = []
    for sp in ("siglip", "uiclip", "layout"):
        q = load(f"q2_{sp}")
        for g, lab in (
            ("ai_host", "AI-builder sites"),
            ("generated", "Open-model pages"),
            ("curated", "Curated library sites"),
            ("human_new", "New human sites (2023+)"),
        ):
            if g in q:
                rows.append(
                    (SPACE_LABEL[sp], lab, q[g]["ratio"], q[g]["ci"], q[g]["n"])
                )
                print(
                    "centre",
                    sp,
                    g,
                    round(q[g]["ratio"], 3),
                    [round(x, 3) for x in q[g]["ci"]],
                    "n",
                    q[g]["n"],
                )
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for i, (sp, lab, r, c, n) in enumerate(rows):
        col = ACCENT if lab.startswith("AI") else GREY if lab.startswith("New") else INK
        ax.plot([c[0], c[1]], [i, i], color=col, lw=2)
        ax.plot([r], [i], "o", color=col, ms=5)
    ax.axvline(1, color=SOFT, lw=0.8, ls=":")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{lab} · {sp}" for sp, lab, *_ in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Distance to the category centre, ÷ human sites' (1 = same as human)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_centre.png", dpi=200)
    plt.close(fig)


def wire(layout) -> str:
    """15 x 16 wireframe from the 3 x 30 x 32 occupancy grid: per cell the dominant channel (t text, m media,
    c control) when covered at least 15%, else '.'."""
    import numpy as np

    g = (
        np.asarray(layout, np.float32)
        .reshape(3, 30, 32)
        .reshape(3, 15, 2, 16, 2)
        .mean((2, 4))
    )
    best = g.argmax(0)
    out = []
    for i in range(15):
        for j in range(16):
            out.append("tmc"[best[i, j]] if g[best[i, j], i, j] >= 0.15 else ".")
    return "".join(out)


def pal(r):
    from features import lab_to_hex

    return [[lab_to_hex(c), round(float(w), 3)] for c, w in zip(r["pal_c"], r["pal_w"])]


def export_data():
    """report/data.js: derived numbers only (palettes, wireframes, similarities), no domains or screenshots."""
    import random

    import numpy as np
    from analyse import _unit, slop_scores
    from dataset import load

    rs = load()
    data = {}
    # 1. year slider: our held-out panel, stable sites, even years
    panel = [
        r
        for r in rs
        if r["source"] == "wayback" and r["split"] == "held" and r["stable"]
    ]
    q = load_json("q1_ours")
    years = sorted({r["year"] for r in panel})
    rng = random.Random(0)
    data["years"] = []
    for y in years:
        ys = sorted((r for r in panel if r["year"] == y), key=lambda r: r["pid"])
        pick = rng.sample(ys, min(40, len(ys)))
        mean = np.mean([np.asarray(r["layout"], np.float32) for r in ys], 0)
        entry = {
            "year": y,
            "n": len(ys),
            "palettes": [pal(r) for r in pick],
            "heat": [
                round(float(v), 3)
                for v in mean.reshape(3, 30, 32)
                .sum(0)
                .clip(0, 1)
                .reshape(15, 2, 16, 2)
                .mean((1, 3))
                .ravel()
            ],
        }
        for sp in ("layout", "palette", "siglip", "dom"):
            res = q.get(f"stable/{sp}") if q else None
            if res and y in res["years"]:
                i = res["years"].index(y)
                entry[sp] = [
                    round(res["norm"][i], 4),
                    [round(v, 4) for v in res["norm_ci"][i]],
                ]
        data["years"].append(entry)
    # 2. nearest-neighbour explorer: pages and their nearest pages from other categories (DOM graph = structure)
    pool = [
        r for r in rs if r["source"] in ("human", "ai_host", "generated", "template")
    ]
    _, rep, _ = slop_scores(pool, "dom")
    X = _unit(np.stack([r["dom"] for r in pool]))
    cats = np.array([r["category"] for r in pool])
    held = [i for i, r in enumerate(pool) if r["split"] == "held"]
    picks = []
    for src, n in (("human", 30), ("ai_host", 30), ("generated", 15), ("template", 15)):
        ids = sorted(
            (i for i in held if pool[i]["source"] == src), key=lambda i: pool[i]["pid"]
        )
        picks += rng.sample(ids, min(n, len(ids)))
    zrep = rep  # already z-scored against dev human pages by slop_scores
    items = []
    for i in picks:
        sims = X @ X[i]
        mask = cats != cats[i]
        order = [
            j
            for j in np.argsort(-sims)
            if mask[j] and pool[j]["domain"] != pool[i]["domain"]
        ][:5]
        items.append(
            {
                "cls": pool[i]["source"],
                "cat": pool[i]["category"],
                "rep": round(float(zrep[i]), 2),
                "wire": wire(pool[i]["layout"]),
                "pal": pal(pool[i]),
                "nn": [
                    {
                        "cls": pool[j]["source"],
                        "cat": pool[j]["category"],
                        "sim": round(float(sims[j]), 3),
                        "wire": wire(pool[j]["layout"]),
                        "pal": pal(pool[j]),
                    }
                    for j in order
                ],
            }
        )
    data["explorer"] = items
    (HERE / "report" / "data.js").write_text(
        "window.DI_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n"
    )
    print("data.js:", len(data["years"]), "years;", len(items), "explorer pages")


def load_json(name):
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    fig_waybackui()
    if (RESULTS / "q1_ours.json").exists():
        fig_panel()
    fig_centre()
    export_data()


if __name__ == "__main__":
    main()
