"""Every figure in the note, drawn from results/ (prints the numbers it plots).

python make_figures.py   # -> report/figures/fig1-fig6.png, hero.jpg, og.jpg
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import HERE, RESULTS

FIG = HERE / "report" / "figures"
EMBER, EMBER_SOFT, GREY, GREY_SOFT, INK = "#c9561c", "#f2b48e", "#77706a", "#c9c3bc", "#2a2420"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#8a837c",
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "figure.dpi": 150,
    }
)


def save(fig: plt.Figure, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig1_scatter() -> None:
    """Every v1 rule: share of human windows it fires on vs share of open-model windows."""
    d = pd.read_csv(RESULTS / "rules_v1_all.csv")
    d = d[d.kind.isin(["word", "habit", "signpost", "stock", "phrase", "pattern"])]
    floor = 1e-4
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    lim = [floor * 0.8, 1]
    ax.plot(lim, lim, color=GREY, lw=1, ls="--")
    ax.plot(lim, [2 * x for x in lim], color=GREY_SOFT, lw=1, ls=":")
    ax.text(0.5, 0.42, "people = models", color=GREY, fontsize=8, rotation=38, ha="right")
    ax.text(0.2, 0.47, "models 2x", color=GREY, fontsize=8, rotation=38, ha="right")
    style = {
        "word": ("o", 18),
        "habit": ("s", 46),
        "signpost": ("^", 22),
        "stock": ("^", 22),
        "phrase": ("D", 22),
        "pattern": ("P", 46),
    }
    for kind, (m, sz) in style.items():
        k = d[d.kind == kind]
        x, y = np.maximum(k.h, floor), np.maximum(k.a, floor)
        keep = k.bp >= 2 / 3
        ax.scatter(x[keep], y[keep], marker=m, s=sz, color=EMBER, alpha=0.8, lw=0, label=kind)
        ax.scatter(x[~keep], y[~keep], marker=m, s=sz, color=GREY, alpha=0.7, lw=0)
    for r in d[d.kind == "habit"].itertuples():
        ax.annotate(
            r.label,
            (max(r.h, floor), max(r.a, floor)),
            fontsize=7.5,
            color=INK,
            xytext=(4, -3),
            textcoords="offset points",
        )
    ax.set_xscale("log"), ax.set_yscale("log"), ax.set_xlim(lim), ax.set_ylim(lim)
    ax.set_xlabel("share of human windows the rule fires on (5,264 windows)")
    ax.set_ylabel("share of open-model windows it fires on (4,802)")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right", title="orange: models 2x or more", title_fontsize=7.5)
    ax.text(floor * 0.9, floor * 1.25, "never seen", fontsize=7, color=GREY)
    save(fig, "fig1_rules.png")
    print("fig1:", len(d), "rules;", int((d.bp >= 2 / 3).sum()), "at 2x or more")


def fig2_words() -> None:
    d = pd.read_csv(RESULTS / "rules_v1_all.csv")
    w = d[d.kind == "word"].copy()
    w["bp_plot"] = w.bp.fillna(-1)
    w = w.sort_values("bp_plot")
    n_h, n_a = int(w.n_h.iloc[0]), int(w.n_a.iloc[0])
    rl = json.loads((RESULTS / "random_lists.json").read_text())
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 12.5), gridspec_kw={"width_ratios": [3, 1.3]})
    y = np.arange(len(w))
    for yi, r in zip(y, w.itertuples(), strict=True):
        c = EMBER if (np.isfinite(r.bp) and r.bp >= 2 / 3) else GREY
        if np.isfinite(r.bp):
            ax.plot([r.bp_lo, r.bp_hi], [yi, yi], color=c, lw=1.2, alpha=0.6)
            ax.scatter([r.bp], [yi], color=c, s=14, zorder=3)
        else:
            ax.text(0.5, yi, "never used by models or people", fontsize=6.5, va="center", ha="center", color=GREY)
    ax.axvline(0.5, color=GREY, lw=1, ls="--")
    ax.axvline(2 / 3, color=GREY_SOFT, lw=1, ls=":")
    ax.set_yticks(y, [f"{r.label}  ({round(r.h * n_h)}/{round(r.a * n_a)})" for r in w.itertuples()], fontsize=6.6)
    ax.set_xlim(-0.02, 1.02), ax.set_ylim(-1, len(w))
    ax.set_xlabel("balanced precision: P(AI | the word appears) at 50/50, 95% CI")
    ax.set_title(
        f"The 89 v1 ban words (human / open-model windows using it, of {n_h:,} / {n_a:,})", fontsize=10, loc="left"
    )
    rand = np.array([x for x in rl["random_bp"] if np.isfinite(x)])
    ax2.hist(rand, bins=24, color=GREY_SOFT, edgecolor="white")
    ax2.axvline(rl["real_bp"], color=EMBER, lw=2)
    ax2.text(rl["real_bp"], ax2.get_ylim()[1] * 0.92, f" ban list\n {rl['real_bp']:.2f}", color=EMBER, fontsize=9)
    ax2.set_xlabel("BP of 'any list word appears'")
    ax2.set_title("vs 200 random lists\nmatched on word frequency", fontsize=10, loc="left")
    ax2.set_yticks([])
    save(fig, "fig2_words.png")
    print(
        "fig2: list BP",
        round(rl["real_bp"], 3),
        "random max",
        round(float(rand.max()), 3),
        "p95",
        round(float(np.percentile(rand, 95)), 3),
    )


def fig3_habits_by_model() -> None:
    d = pd.read_csv(RESULTS / "rules_v1_by_model.csv")
    d = d[d.rule.str.startswith("habit:")]
    order = [
        "formatting",
        "signpost",
        "stock_phrase",
        "slop_word",
        "em_dash",
        "which_chain",
        "not_x_but_y",
        "triad",
        "rhetorical_qa",
        "invented_claim",
        "fragment",
    ]
    groups = list(d.group.unique())
    models = [m for m in groups if not m.startswith("closed:")] + sorted(m for m in groups if m.startswith("closed:"))
    M = np.full((len(order), len(models)), np.nan)
    for i, h in enumerate(order):
        for j, m in enumerate(models):
            v = d[(d.rule == f"habit:{h}") & (d.group == m)].bp
            if len(v):
                M[i, j] = v.iloc[0]
    from matplotlib.colors import LinearSegmentedColormap

    fig, ax = plt.subplots(figsize=(10, 5.2))
    cmap = LinearSegmentedColormap.from_list("bp", ["#5f5a55", "#f4f1ec", EMBER])
    im = ax.imshow(M, cmap=cmap, vmin=0.2, vmax=1.0, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color=INK)
    ax.set_yticks(range(len(order)), [o.replace("_", " ") for o in order])
    ax.set_xticks(range(len(models)), [m.replace("closed:", "") for m in models], rotation=40, ha="right", fontsize=8)
    n_open = sum(1 for m in models if not m.startswith("closed:"))
    ax.axvline(n_open - 0.5, color=INK, lw=1.5)
    ax.text(n_open / 2 - 0.5, -0.9, "8 open-weights models (decision set)", ha="center", fontsize=8.5)
    ax.text(n_open + (len(models) - n_open) / 2 - 0.5, -0.9, "closed models (transfer only)", ha="center", fontsize=8.5)
    fig.colorbar(im, ax=ax, fraction=0.025, label="BP (0.5 = coin)")
    ax.spines[:].set_visible(False)
    save(fig, "fig3_habits_by_model.png")
    print("fig3: em dash BP by model", dict(zip(models, np.round(M[order.index("em_dash")], 2), strict=True)))


def fig4_scores() -> None:
    from sklearn.metrics import roc_curve

    s = pd.read_csv(RESULTS / "window_scores.csv")
    cmp_ = json.loads((RESULTS / "compare.json").read_text())
    t = s[(s.split == "test") & (s.set == "matched")]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
    ax = axes[0]
    for col, lab, c in [("score_v1", "v1 (hand-set)", GREY), ("score_v2", "v2 (data-backed)", EMBER)]:
        fpr, tpr, _ = roc_curve(t.side == "ai", t[col])
        au = cmp_["auroc"]["v1" if col == "score_v1" else "v2"]["open_test"]
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{lab}: {au[0]:.2f} [{au[1]:.2f}, {au[2]:.2f}]")
    ax.plot([0, 1], [0, 1], color=GREY_SOFT, ls="--", lw=1)
    bow = cmp_["baselines"]["bag_of_words"]["open_test"][0]
    ax.set_title(f"Held-out ROC (bag of words: {bow:.2f})", fontsize=10, loc="left")
    ax.set_xlabel("human windows flagged"), ax.set_ylabel("open-model windows flagged")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    panels = [
        (axes[1], "score_v1", 33.63, "v1 score, band 33.6 (classic)"),
        (axes[2], "score_v2", cmp_["limits"]["v2"], f"v2 score, band {cmp_['limits']['v2']:.1f} (modern)"),
    ]
    for ax, col, lim, title in panels:
        bins = np.linspace(0, 120, 41)
        hum = s[(s.split == "test") & (s.side == "human") & s.set.isin(["matched", "raid"])][col]
        ai = t[t.side == "ai"][col]
        ax.hist(np.clip(hum, 0, 119.9), bins=bins, color=GREY_SOFT, density=True, label=f"modern human, n={len(hum):,}")
        ax.hist(
            np.clip(ai, 0, 119.9),
            bins=bins,
            color=EMBER if col == "score_v2" else GREY,
            alpha=0.55,
            density=True,
            label=f"open models, n={len(ai):,}",
        )
        ax.axvline(lim, color=INK, lw=1.2, ls="--")
        ax.set_title(title, fontsize=10, loc="left"), ax.set_xlabel("points per 1,000 words (clipped at 120)")
        ax.set_yticks([])
        ax.legend(fontsize=7.5, frameon=False)
    save(fig, "fig4_scores.png")
    print("fig4:", cmp_["auroc"]["v1"]["open_test"], cmp_["auroc"]["v2"]["open_test"])


def fig5_sources() -> None:
    d = pd.read_csv(RESULTS / "compare_by_source.csv")
    d = d[d.n_ai > 0].sort_values("auroc_v2")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    y = np.arange(len(d))
    for yi, r in zip(y, d.itertuples(), strict=True):
        ax.plot([r.auroc_v1, r.auroc_v2], [yi, yi], color=GREY_SOFT, lw=2)
    ax.scatter(d.auroc_v1, y, color=GREY, s=30, label="v1", zorder=3)
    ax.scatter(d.auroc_v2, y, color=EMBER, s=30, label="v2", zorder=3)
    ax.set_yticks(y, [s.replace("_", " ") for s in d.source])
    ax.axvline(0.5, color=GREY, ls="--", lw=1)
    ax.set_xlim(0.45, 1.0), ax.set_xlabel("held-out AUROC, human vs open models, same source")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "fig5_sources.png")
    print("fig5:", d[["source", "auroc_v1", "auroc_v2"]].round(3).to_dict("records"))


def fig6_discovered() -> None:
    c = pd.read_csv(RESULTS / "candidates.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for kind, m in {"word": "o", "ngram": "D", "frame": "^"}.items():
        k = c[c.kind == kind]
        ax.scatter(
            k.df_a * 100,
            k.test_bp,
            marker=m,
            s=28,
            color=[EMBER if p else GREY for p in k.passes],
            alpha=0.85,
            lw=0,
            label=kind,
        )
    for r in c[(c.kind != "frame") & (c.df_a > 0.045)].itertuples():
        ax.annotate(r.item, (r.df_a * 100, r.test_bp), fontsize=7, xytext=(3, 2), textcoords="offset points")
    ax.axhline(0.5, color=GREY, ls="--", lw=1)
    ax.set_xlabel("share of open-model train windows using it (%)")
    ax.set_ylabel("held-out BP")
    ax.set_ylim(0.3, 1.03)
    ax.legend(frameon=False, fontsize=8, title="orange: passed held-out (lower bound >= 0.70)", title_fontsize=7.5)
    save(fig, "fig6_discovered.png")
    print("fig6:", int(c.passes.sum()), "of", len(c), "passed")


def cover() -> None:
    """Hero: every v1 rule (and each named-pattern alternative) at its human vs AI firing rate."""
    d = pd.read_csv(RESULTS / "rules_v1_all.csv")
    d = d[d.kind.isin(["word", "habit", "signpost", "stock", "phrase", "pattern", "patalt"])]
    floor = 1.5e-4
    keep = d.bp >= 2 / 3
    for name, size in [("hero.jpg", (16, 9))]:  # build_meta.py crops og.jpg from this
        fig, ax = plt.subplots(figsize=size)
        fig.patch.set_facecolor("#15110e"), ax.set_facecolor("#15110e")
        lim = [floor * 0.7, 1.2]
        ax.plot(lim, lim, color="#6f675f", lw=1.5, ls="--")
        x, y = np.maximum(d.h, floor), np.maximum(d.a, floor)
        rng = np.random.default_rng(0)
        jx, jy = np.exp(rng.normal(0, 0.06, len(d))), np.exp(rng.normal(0, 0.06, len(d)))
        ax.scatter(x[~keep] * jx[~keep], y[~keep] * jy[~keep], s=60, color="#a39a90", alpha=0.75, lw=0)
        ax.scatter(x[keep] * jx[keep], y[keep] * jy[keep], s=60, color="#e07a3f", alpha=0.85, lw=0)
        ax.set_xscale("log"), ax.set_yscale("log"), ax.set_xlim(lim), ax.set_ylim(lim)
        ax.axis("off")
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.97)
        fig.savefig(FIG / name, dpi=100, facecolor="#15110e")  # exactly 1600x900
        plt.close(fig)
    print("cover:", len(d), "rules,", int(keep.sum()), "orange")


def data_js() -> None:
    """report/data.js for the rule explorer: every v1 rule and every discovered candidate, with its v2 decision."""
    d = pd.read_csv(RESULTS / "rules_v1_all.csv")
    dec = pd.read_csv(RESULTS / "list_decisions.csv").set_index("rule")
    kinds = {
        "word": "ban word",
        "habit": "habit",
        "signpost": "signpost",
        "stock": "stock phrase",
        "phrase": "tool phrase",
        "pattern": "named pattern",
        "patalt": "pattern part",
        "stockword": "tool word",
    }
    rows = []
    for r in d[d.kind.isin(list(kinds))].itertuples():
        action = dec.action.get(r.rule, "kept (habit)" if r.kind == "habit" else "")
        rows.append(
            [
                kinds[r.kind],
                r.label,
                round(r.h, 5),
                round(r.a, 5),
                None if pd.isna(r.bp) else round(r.bp, 3),
                None if pd.isna(r.bp_lo) else round(r.bp_lo, 3),
                None if pd.isna(r.bp_hi) else round(r.bp_hi, 3),
                r.verdict,
                action,
            ]
        )
    c = pd.read_csv(RESULTS / "candidates.csv")
    for r in c.itertuples():
        key = f"{r.kind}:{r.item}"
        action = dec.action.get(key, "report only") if r.kind != "frame" else "report only"
        rows.append(
            [
                f"new {r.kind}",
                r.item,
                round(r.test_h, 5),
                round(r.test_a, 5),
                round(r.test_bp, 3),
                round(r.test_bp_lo, 3),
                round(r.test_bp_hi, 3),
                "passes" if r.passes else "fails held-out",
                action,
            ]
        )
    cols = ["kind", "rule", "human", "ai", "bp", "lo", "hi", "verdict", "v2"]
    (HERE / "report" / "data.js").write_text("window.RULES = " + json.dumps({"cols": cols, "rows": rows}) + ";\n")
    print("data.js:", len(rows), "rows")


if __name__ == "__main__":
    fig1_scatter(), fig2_words(), fig3_habits_by_model(), fig4_scores(), fig5_sources(), fig6_discovered(), cover()
    data_js()
