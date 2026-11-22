"""
Figures, the cover and the interactive's data for The Taste Stack Episode 7, drawn only from results/ (derived
numbers; no images or brief text). Prints every number it plots so it can be checked against the note.

    .venv/bin/python experiments/designer-taste/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "report" / "figures"
EMBER, INK, GREY, LIGHT, CREAM = "#c9561c", "#2a2420", "#9a928a", "#d9d3cc", "#f6f2ec"
plt.rcParams.update(
    {"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
)
TIER_LABEL = {
    "formal": "Formal check",
    "probe": "Probe (shrunk head)",
    "small": "Qwen2.5-VL-7B",
    "mid": "Gemma-3-27B",
    "large": "Qwen2.5-VL-72B",
}


def J(name):
    return json.loads((RES / f"{name}.json").read_text())


def leaderboard(H, Jd, R):
    rows = [
        ("Human reference: the other 4 designers' majority", H["human_loo_ceiling"], "ceiling"),
        ("Router (cheapest good-enough judge per part)", R["router"], "grey"),
        ("Per-designer head", H["own"], "ember"),
        ("Shrunk head (shared + designer offset)", H["shrunk"], "grey"),
        ("Pooled head (whole cohort)", H["pooled"], "grey"),
        ("Pooled head, one designer's amount of data", H["pooled_fifth"], "grey"),
        ("Another designer's head", H["other"], "grey"),
        ("Per-designer generator prior (4 numbers)", H["gen_prior"], "ember"),
        ("Image statistics only (6 numbers)", H["stats"], "grey"),
        ("Shrunk head, generator removed", H["resid_shrunk"], "grey"),
        ("Judge: Qwen2.5-VL-72B", Jd["large"], "grey"),
        ("Judge: Gemma-3-27B", Jd["mid"], "grey"),
        ("Judge: Qwen2.5-VL-7B", Jd["small"], "grey"),
        ("Formal checks (0.5 where none applies)", Jd["formal"], "grey"),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, s, kind) in zip(y, rows):
        c = EMBER if kind == "ember" else (INK if kind == "ceiling" else GREY)
        ax.plot(s["ci"], [yi, yi], color=c, lw=2.2, solid_capstyle="round")
        ax.plot(s["mean"], yi, "o", color=c, ms=7, mfc="white" if kind == "ceiling" else c)
        ax.text(s["ci"][1] + 0.003, yi, f"{s['mean']:.3f}", va="center", fontsize=8.5, color=INK)
        print(f"leaderboard {lab}: {s['mean']} {s['ci']}")
    sh = H["shuffled_own"]
    ax.axvspan(sh["range"][0], sh["range"][1], color=LIGHT, alpha=0.6, lw=0)
    ax.axvline(0.5, color=GREY, lw=0.8, ls=":")
    ax.text(0.5, -1.45, "coin flip", fontsize=7.5, color=GREY, ha="center")
    ax.text(
        (sh["range"][0] + sh["range"][1]) / 2,
        len(rows) - 0.45,
        "shaded: heads on shuffled\nrankings (10 runs)",
        fontsize=7.5,
        color=GREY,
        ha="center",
    )
    ax.set_yticks(y, [r[0] for r in rows], fontsize=9)
    bound = json.loads((RES / "post_hoc.json").read_text())["designer_blind_bound_in_sample"]["mean"]
    ax.annotate(
        f"in-sample bound for any designer-blind judge: {bound:.3f} →",
        (0.649, len(rows) - 1),
        fontsize=7.5,
        color=INK,
        ha="right",
        va="center",
        xytext=(0.649, len(rows) - 1.55),
    )
    ax.set_xlim(0.47, 0.65)
    ax.set_ylim(-1.6, len(rows) - 0.1)
    ax.set_xlabel("agreement with each designer, held-out pairs\n(mean of 9 parts; 95% scene-bootstrap CI)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_leaderboard.png", dpi=200)
    plt.close(fig)


def parts(R, H):
    tab = R["table"]
    partsl = []
    for r in tab:
        if r["part"] not in partsl:
            partsl.append(r["part"])
    tiers = ["formal", "probe", "small", "mid", "large"]
    acc = {p: {} for p in partsl}
    for p in partsl:
        rows = [r for r in tab if r["part"] == p]
        n = sum(r["n_test"] for r in rows)
        for t in tiers:
            if t in rows[0]["test_hits"]:
                acc[p][t] = sum(r["test_hits"][t] for r in rows) / n
    ceil = H["human_loo_ceiling"]["per_part"]
    gp = H["gen_prior"]["per_part"]
    chosen = {}
    for ch in R["choices"].values():
        for key, t in ch.items():
            lab = next(r["part"] for r in tab if r["key"] == key)
            chosen.setdefault(lab, []).append(t)
    fig, ax = plt.subplots(figsize=(9, 6.0))
    marks = {"formal": "s", "probe": "o", "small": "v", "mid": "D", "large": "^"}
    off = {t: (i - 2) * 0.12 for i, t in enumerate(tiers)}
    for i, p in enumerate(partsl[::-1]):
        yi = i
        ax.plot([0.4, 0.72], [yi, yi], color=LIGHT, lw=0.6, zorder=0)
        ax.plot(ceil[p], yi, "|", color=INK, ms=16, mew=2)
        ax.plot(gp[p], yi, "x", color=GREY, ms=6)
        for t in tiers:
            if t in acc[p]:
                c = EMBER if t in chosen.get(p, []) else GREY
                ax.plot(acc[p][t], yi + off[t], marks[t], color=c, ms=6.5, mfc=c if t in chosen.get(p, []) else "white")
        print(
            f"parts {p}: "
            + ", ".join(f"{t} {acc[p][t]:.3f}" for t in tiers if t in acc[p])
            + f"; ceiling {ceil[p]}; router {chosen.get(p)}"
        )
    ax.set_yticks(range(len(partsl)), partsl[::-1])
    ax.axvline(0.5, color=GREY, lw=0.8, ls=":")
    for t in tiers:
        ax.plot([], [], marks[t], color=GREY, mfc="white", label=TIER_LABEL[t])
    ax.plot([], [], "o", color=EMBER, label="chosen by the router (filled)")
    ax.plot([], [], "x", color=GREY, label="generator prior")
    ax.plot([], [], "|", color=INK, ms=12, mew=2, label="human ceiling")
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=4, bbox_to_anchor=(0.45, -0.13))
    ax.set_xlim(0.4, 0.72)
    ax.set_xlabel("agreement with the individual designer, held-out pairs (both folds pooled)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_parts.png", dpi=200)
    plt.close(fig)


def designers(H):
    per = H["per_designer"]
    ds = sorted(per, key=lambda d: int(d[1:]))
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    x = np.arange(len(ds))
    for k, (col, c, m) in enumerate(
        [("h_own", EMBER, "o"), ("h_pooled", GREY, "s"), ("h_other", GREY, "D"), ("h_loo", INK, "_")]
    ):
        vals = [per[d][col] for d in ds]
        ax.plot(
            x + (k - 1.5) * 0.12,
            vals,
            m,
            color=c,
            ms=7 if m != "_" else 16,
            mew=2,
            label={
                "h_own": "own head",
                "h_pooled": "pooled head",
                "h_other": "another designer's head",
                "h_loo": "other designers' majority",
            }[col],
        )
    for d in ds:
        print(
            f"designer {d}: own {per[d]['h_own']}, pooled {per[d]['h_pooled']}, other {per[d]['h_other']}, ceiling {per[d]['h_loo']}"
        )
    ax.axhline(0.5, color=GREY, lw=0.8, ls=":")
    ax.axvline(4.5, color=LIGHT, lw=1)
    ax.text(2, 0.68, "aesthetics cohort", ha="center", fontsize=8.5, color=GREY)
    ax.text(7, 0.68, "descriptions cohort", ha="center", fontsize=8.5, color=GREY)
    ax.set_xticks(x, ds)
    ax.set_ylim(0.44, 0.7)
    ax.set_ylabel("agreement, all criteria of the track")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    fig.tight_layout()
    fig.savefig(FIG / "fig_designers.png", dpi=200)
    plt.close(fig)


def constitution(C):
    arms = [
        ("none", "no constitution", INK, "o"),
        ("own", "the designer's own", EMBER, "o"),
        ("other", "another designer's", GREY, "D"),
        ("pooled", "the cohort's", GREY, "s"),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    labs = list(C["none"]["per_part"])
    w = 0.16
    for k, (a, lab, c, m) in enumerate(arms):
        for j, p in enumerate(labs):
            v = C[a]["per_part"][p]
            lo, hi = C[a]["per_part_ci"][p]
            x = j + (k - 1.5) * w
            ax.plot([x, x], [lo, hi], color=c, lw=1.6)
            ax.plot(x, v, m, color=c, ms=7, mfc=c if a != "pooled" else "white", label=lab if j == 0 else None)
            print(f"constitution {p} {a}: {v} [{lo}, {hi}]")
    if "R7_bright" in C:
        j = len(labs)
        b = C["R7_bright"]
        ax.plot(j - 0.5 * w, b["no_constitution"], "o", color=INK, ms=7)
        ax.plot(j + 0.5 * w, b["own_constitution"], "o", color=EMBER, ms=7)
        ax.plot([j - 0.4, j + 0.4], [0.8, 0.8], color=INK, lw=1, ls="--")
        ax.text(j, 0.81, "R7 bar 0.80", ha="center", fontsize=8, color=INK)
        labs = labs + ["Positive control:\nbrightness designer"]
        print(f"constitution R7 bright: none {b['no_constitution']}, own {b['own_constitution']}")
    ax.axhline(0.5, color=GREY, lw=0.8, ls=":")
    ax.set_xticks(range(len(labs)), labs, fontsize=9)
    ax.set_xlim(-0.6, len(labs) - 0.4)
    ax.set_ylim(0.4, 0.86)
    ax.set_ylabel("agreement, Qwen2.5-VL-72B judge")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_constitution.png", dpi=200)
    plt.close(fig)


def router(R):
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for t, s in R["single"].items():
        ax.errorbar(
            s["cost_per_1k"],
            s["mean"],
            yerr=[[s["mean"] - s["ci"][0]], [s["ci"][1] - s["mean"]]],
            fmt="o",
            color=GREY,
            ms=6,
            capsize=0,
        )
        off = {"probe": (-8, 8), "mid": (-8, -12), "small": (6, 4)}.get(t, (6, -11))
        ax.annotate(
            TIER_LABEL[t],
            (s["cost_per_1k"], s["mean"]),
            textcoords="offset points",
            xytext=off,
            fontsize=8.5,
            color=INK,
            ha="right" if t in ("probe", "mid") else "left",
        )
        print(f"router single {t}: {s['mean']} {s['ci']} at ${s['cost_per_1k']}/1k")
    for key, lab, c in (("router", "Router", EMBER), ("oracle_router", "Oracle router (chosen on test)", INK)):
        s = R[key]
        ax.errorbar(
            s["cost_per_1k"],
            s["mean"],
            yerr=[[s["mean"] - s["ci"][0]], [s["ci"][1] - s["mean"]]],
            fmt="o",
            color=c,
            ms=8,
            mfc=c if key == "router" else "white",
            capsize=0,
        )
        ax.annotate(
            lab,
            (s["cost_per_1k"], s["mean"]),
            textcoords="offset points",
            xytext=(8, -14) if key == "router" else (6, 6),
            fontsize=9,
            color=c,
        )
        print(f"router {key}: {s['mean']} {s['ci']} at ${s['cost_per_1k']}/1k")
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.2, 0.5, 1.0], ["$0.10", "$0.20", "$0.50", "$1"])
    ax.axhline(0.5, color=GREY, lw=0.8, ls=":")
    ax.set_xlabel("compute cost per 1,000 pair decisions (US$, log scale)")
    ax.set_ylabel("agreement (mean of 9 parts)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_router.png", dpi=200)
    plt.close(fig)


def cover(R, H):
    """The 9 parts x 5 tiers of held-out agreement as a grid; the router's picks outlined."""
    from PIL import Image

    tab = R["table"]
    partsl = list(dict.fromkeys(r["part"] for r in tab))
    tiers = ["formal", "probe", "small", "mid", "large"]
    M = np.full((len(partsl), len(tiers)), np.nan)
    for i, p in enumerate(partsl):
        rows = [r for r in tab if r["part"] == p]
        n = sum(r["n_test"] for r in rows)
        for j, t in enumerate(tiers):
            if t in rows[0]["test_hits"]:
                M[i, j] = sum(r["test_hits"][t] for r in rows) / n
    picks = {(next(r["part"] for r in tab if r["key"] == k), t) for ch in R["choices"].values() for k, t in ch.items()}
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("e", ["#efe7dd", "#f2b48e", "#e07a3f", "#c9561c"])
    for name, size in (("hero.jpg", (16, 9)), ("og.jpg", (12, 6.3))):
        fig, ax = plt.subplots(figsize=size, dpi=100)
        fig.patch.set_facecolor(CREAM)
        ax.set_facecolor(CREAM)
        ax.imshow(np.where(np.isnan(M), 0.5, M), cmap=cmap, vmin=0.48, vmax=0.66, aspect="auto")
        for i in range(len(partsl)):
            for j in range(len(tiers)):
                if np.isnan(M[i, j]):
                    ax.add_patch(
                        plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=CREAM, edgecolor=LIGHT, hatch="///", lw=0)
                    )
                    ax.text(j, i, "no check", ha="center", va="center", fontsize=13, color=GREY)
                    continue
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=17, color=INK)
                if (partsl[i], tiers[j]) in picks:
                    ax.add_patch(plt.Rectangle((j - 0.47, i - 0.45), 0.94, 0.9, fill=False, ec=INK, lw=3))
        ax.set_xticks(range(len(tiers)), [TIER_LABEL[t] for t in tiers], fontsize=16, color=INK)
        ax.set_yticks(range(len(partsl)), partsl, fontsize=16, color=INK)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.xaxis.tick_top()
        fig.tight_layout()
        png = (FIG / name).with_suffix(".png")
        fig.savefig(png, dpi=100 if name == "og.jpg" else 120, facecolor=fig.get_facecolor())
        plt.close(fig)
        im = Image.open(png).convert("RGB")
        if name == "og.jpg":
            im = im.resize((1200, 630))
        im.save(FIG / name, quality=88)
        png.unlink()
    thumbs = HERE.parent / "shared" / "thumbs"
    if thumbs.exists():  # series cards use 720x480 (3:2), cropped from the centre of the 16:9 hero
        im = Image.open(FIG / "hero.jpg")
        w, h = im.size
        cw = int(h * 1.5)
        im.crop(((w - cw) // 2, 0, (w - cw) // 2 + cw, h)).resize((720, 480)).save(
            thumbs / "designer-taste.jpg", quality=85
        )


def closed(Cj, G):
    """Exploratory: (a) the closed judge beside every arm on its 539 items; (b) each generator's win rate."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.1), gridspec_kw={"width_ratios": [1.25, 1]})
    rows = [
        ("other_designers_majority", "Other designers' majority", INK),
        ("generator_prior", "Generator prior (4 numbers)", EMBER),
        ("closed_strict", "Claude Sonnet 5.5 (cut-offs = tie)", EMBER),
        ("own_head", "Own head", GREY),
        ("qwen72_both_orders", "Qwen2.5-VL-72B, both orders", GREY),
        ("probe_shrunk", "Probe (shrunk head)", GREY),
        ("qwen72_same_order", "Qwen2.5-VL-72B, same order", GREY),
        ("qwen7_same_order", "Qwen2.5-VL-7B, same order", GREY),
        ("gemma27_same_order", "Gemma-3-27B, same order", GREY),
    ]
    y = np.arange(len(rows))[::-1]
    for yi, (k, lab, c) in zip(y, rows):
        s = Cj["arms"][k]
        a1.plot(s["ci"], [yi, yi], color=c, lw=2)
        a1.plot(s["mean"], yi, "o", color=c, ms=6, mfc="white" if c == INK else c)
        a1.text(s["ci"][1] + 0.003, yi, f"{s['mean']:.3f}", va="center", fontsize=8)
        print(f"closed {k}: {s['mean']} {s['ci']}")
    a1.axvline(0.5, color=GREY, lw=0.8, ls=":")
    a1.set_yticks(y, [r[1] for r in rows], fontsize=8.5)
    a1.set_xlim(0.48, 0.67)
    a1.set_xlabel(f"agreement on the closed judge's {Cj['n_items']} items")
    a1.set_title("(a) same items, every arm", fontsize=9.5, loc="left")
    src = [
        ("designers", "Designers", INK),
        ("Qwen2.5-VL-7B", "Qwen-7B", GREY),
        ("Gemma-3-27B", "Gemma-27B", GREY),
        ("Qwen2.5-VL-72B", "Qwen-72B", GREY),
        ("closed", "Sonnet 5.5", EMBER),
    ]
    gs = ["G1", "G2", "G3", "G4"]
    w = 0.16
    for k, (key, lab, c) in enumerate(src):
        vals = Cj["generator_win_rate_answered"] if key == "closed" else G[key]
        a2.bar(
            np.arange(4) + (k - 2) * w,
            [vals[g] for g in gs],
            w * 0.92,
            color=c,
            alpha=1 if c != GREY else 0.35 + 0.2 * k,
            label=lab,
        )
        print(f"generator win {lab}: " + ", ".join(f"{g} {vals[g]}" for g in gs))
    a2.axhline(0.5, color=INK, lw=0.8, ls=":")
    a2.set_xticks(range(4), gs)
    a2.set_ylim(0, 0.75)
    a2.set_ylabel("share of its pairs the generator wins")
    a2.set_title("(b) which generator each judge favours", fontsize=9.5, loc="left")
    a2.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center")
    fig.tight_layout()
    fig.savefig(FIG / "fig_closed.png", dpi=200)
    plt.close(fig)


def data_js(R):
    out = {"cost": R["cost_per_decision"], "table": R["table"], "eps": R["eps"], "labels": TIER_LABEL}
    (HERE / "report" / "data.js").write_text("window.ROUTER = " + json.dumps(out, separators=(",", ":")) + ";\n")
    print(f"data.js: {len(R['table'])} part-fold rows")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    H, Jd, R = J("heads"), J("judges"), J("router")
    leaderboard(H, Jd, R)
    parts(R, H)
    designers(H)
    router(R)
    if (RES / "constitution.json").exists():
        constitution(J("constitution"))
    if (RES / "closed_judge.json").exists():
        closed(J("closed_judge"), J("generators"))
    cover(R, H)
    data_js(R)


if __name__ == "__main__":
    main()
