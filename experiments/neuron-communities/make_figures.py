"""Write results/summary.json and draw the report's figures from results/, printing every number plotted.

python experiments/neuron-communities/make_figures.py
"""

# ruff: noqa: E501, B905  (long data lines and pairwise zips read better as they are)

from __future__ import annotations

import collections
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nc
import numpy as np

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "report" / "figures"
EMBER, EMBER_L, GREY, GREY_L, INK = "#d26f2a", "#f2b48e", "#6f675f", "#c9c0b5", "#1d1a17"
plt.rcParams.update(
    {"font.family": "sans-serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
)


def load(name: str) -> dict:
    return json.loads((RES / name).read_text())


def summary() -> dict:
    b, i = load("e1_base.json"), load("e1_instruct.json")
    bi = {r["id"]: r for r in i["rows"]}
    out: dict = {"e1": {}}
    for tag, d in (("base", b), ("instruct", i)):
        by: dict = collections.defaultdict(lambda: [0, 0])
        for r in d["rows"]:
            by[r["class"]][0] += r["passes"]
            by[r["class"]][1] += 1
        out["e1"][tag] = {
            "passes": d["passes"],
            "n": d["n"],
            "ci": d["pass_ci"],
            "by_class": {k: f"{v[0]}/{v[1]}" for k, v in by.items()},
            "max_random_median": max(max(r["random_medians"]) for r in d["rows"]),
            "median_of_community_medians": round(statistics.median(r["median_kl"] for r in d["rows"]), 3),
        }
    rho = [r["spearman_vs_theirs"] for r in b["rows"] if r["spearman_vs_theirs"] == r["spearman_vs_theirs"]]
    out["e1"]["replication"] = {
        "spearman_median": round(statistics.median(rho), 3),
        "spearman_range": [min(rho), max(rho)],
        "their_median_of_medians": round(statistics.median(r["their_median_n"] for r in b["rows"]), 3),
    }
    tb = sum(1 for r in b["rows"] if r["passes"])
    both = sum(1 for r in b["rows"] if r["passes"] and bi[r["id"]]["passes"])
    only_i = sum(1 for r in b["rows"] if not r["passes"] and bi[r["id"]]["passes"])
    out["e1"]["transfer"] = {
        "pass_base": tb,
        "pass_both": both,
        "pass_instruct_only": only_i,
        "ci": nc.wilson(both, tb),
    }
    an_b = next(r for r in b["rows"] if r["id"] == nc.AN)
    an_i = bi[nc.AN]
    out["e1"]["an"] = {
        "base": [an_b["median_kl"], max(an_b["random_medians"])],
        "instruct": [an_i["median_kl"], max(an_i["random_medians"])],
    }

    a = load("e2a.json")
    rows = a["rows"]
    steer_any = sum(1 for r in rows if r["best_good_hits"] > 0)
    per_c = {}
    for s in nc.STRENGTHS:
        k = str(s)
        per_c[k] = {
            "community_good": sum(r["community"][k]["good_hits"] for r in rows),
            "community_fluent": sum(r["community"][k]["fluent"] for r in rows),
            "random_good": sum(r["random"][k]["good_hits"] for r in rows),
            "random_fluent": sum(r["random"][k]["fluent"] for r in rows),
            "n": 4 * len(rows),
        }
    out["e2a"] = {
        "communities": len(rows),
        "excluded_no_words": 48 - len(rows),
        "with_any_good_hit": steer_any,
        "unsteered_good_hits": sum(r["none_hits"] for r in rows),
        "by_strength": per_c,
        "finalists": a["finalists"],
    }
    if (RES / "e2b.json").exists():
        e = load("e2b.json")["rows"]
        out["e2b"] = [
            {
                "id": r["id"],
                "label": r["label"],
                "c": r["c"],
                "community": r["community"]["good_hits"],
                "community_ci": r["community"]["ci"],
                "community_loops": r["community"]["loops"],
                "direction": r["direction"]["good_hits"],
                "direction_ci": r["direction"]["ci"],
                "random_max": max(r[f"random{k}"]["good_hits"] for k in range(3)),
                "none": r["none"]["good_hits"],
                "steers": r["steers"],
            }
            for r in e
        ]
    e3 = load("e3.json")
    out["e3"] = {
        "collateral": [
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in e3["collateral"]
        ],
        "an_switch": e3["an_switch"],
    }
    (RES / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return out


def fig_e1() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, tag in zip(axes, ("base", "instruct")):
        d = load(f"e1_{tag}.json")
        rows = sorted(d["rows"], key=lambda r: r["median_kl"])
        x = np.arange(len(rows))
        comm = np.array([max(r["median_kl"], 1e-4) for r in rows])
        rnd = np.array([max(max(r["random_medians"]), 1e-4) for r in rows])
        ax.scatter(x, rnd, s=10, color=GREY_L, label="largest of 10 random sets, same layers")
        ax.scatter(
            x, comm, s=16, color=[EMBER if r["passes"] else GREY for r in rows], label="community (orange = passes)"
        )
        ax.axhline(nc.NEC_TAU, color=INK, lw=0.8, ls="--")
        ax.text(1, nc.NEC_TAU * 1.25, "τ = 0.7", fontsize=8)
        ax.set_yscale("log")
        ax.set_title(f"Llama-3.1-8B{'' if tag == 'base' else '-Instruct'}: {d['passes']}/{d['n']} pass", fontsize=10)
        ax.set_xlabel("61 communities, sorted by effect")
        ax.set_xticks([])
        print(
            f"fig1 {tag}: passes {d['passes']}/{d['n']}, community medians {comm.min():.4f}-{comm.max():.2f}, random max {rnd.max():.4f}"
        )
    axes[0].set_ylabel("median KL when zeroed (nats)")
    axes[0].legend(fontsize=8, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "e1_necessity.png", dpi=160, bbox_inches="tight")


def fig_e2a() -> None:
    a = load("e2a.json")
    rows = a["rows"]
    xs = list(nc.STRENGTHS)
    n = 4 * len(rows)
    cg = [sum(r["community"][str(s)]["good_hits"] for r in rows) for s in xs]
    cf = [sum(r["community"][str(s)]["fluent"] for r in rows) for s in xs]
    rg = [sum(r["random"][str(s)]["good_hits"] for r in rows) for s in xs]
    rf = [sum(r["random"][str(s)]["fluent"] for r in rows) for s in xs]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    w = 0.38
    xi = np.arange(len(xs))
    for ax, (c, r, title) in zip(
        axes, ((cg, rg, "fluent replies on the community's topic"), (cf, rf, "fluent replies, any topic"))
    ):
        ax.bar(xi - w / 2, c, w, color=EMBER, label="community clamped")
        ax.bar(xi + w / 2, r, w, color=GREY_L, label="random neurons, same layers, clamped")
        for k in range(len(xs)):
            ax.text(xi[k] - w / 2, c[k] + 1, str(c[k]), ha="center", fontsize=8)
            ax.text(xi[k] + w / 2, r[k] + 1, str(r[k]), ha="center", fontsize=8)
        ax.set_xticks(xi, [f"×{s:g}" for s in xs])
        ax.set_xlabel("clamp strength (× the neuron's mean activation on its own prompts)")
        ax.set_title(f"{title} (of {n})", fontsize=10)
        ax.set_ylim(0, n * 1.08 if r is rf else max(c) * 1.3)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "e2a_screen.png", dpi=160, bbox_inches="tight")
    print(f"fig2: on-topic {cg} vs random {rg}; fluent {cf} vs random {rf} (of {n})")


def fig_e2b_e3() -> None:
    e = {r["id"]: r for r in load("e2b.json")["rows"]}
    col = {r["id"]: r for r in load("e3.json")["collateral"]}
    ids = [f["id"] for f in load("e2a.json")["finalists"]]
    labels = [e[i]["label"] for i in ids]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    y = np.arange(len(ids))[::-1]
    h = 0.26
    for off, arm, color, name in (
        (h, "community", EMBER, "community clamp"),
        (0, "direction", GREY, "direction, same size"),
        (-h, "random", GREY_L, "best of 3 random sets"),
    ):
        vals = [
            max(e[i][f"random{k}"]["good_hits"] for k in range(3)) if arm == "random" else e[i][arm]["good_hits"]
            for i in ids
        ]
        ax.barh(y + off, vals, h, color=color, label=name)
        if arm != "random":
            lo = [15 * e[i][arm]["ci"][0] for i in ids]
            hi = [15 * e[i][arm]["ci"][1] for i in ids]
            ax.errorbar(vals, y + off, xerr=[np.array(vals) - lo, np.array(hi) - vals], fmt="none", ecolor=INK, lw=0.7)
        print(f"fig3 {arm}: {dict(zip(labels, vals))}")
    ax.axvline(8, color=INK, lw=0.8, ls="--")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 15)
    ax.set_xlabel("held-out questions answered fluently on the topic (of 15)")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax = axes[1]
    for i in ids:
        ax.annotate(
            "",
            xy=(col[i]["direction"], e[i]["direction"]["good_hits"]),
            xytext=(col[i]["community"], e[i]["community"]["good_hits"]),
            arrowprops={"arrowstyle": "-", "color": GREY_L, "lw": 0.8},
        )
    ax.scatter(
        [col[i]["community"] for i in ids],
        [e[i]["community"]["good_hits"] for i in ids],
        color=EMBER,
        s=28,
        label="community clamp",
        zorder=3,
    )
    ax.scatter(
        [col[i]["direction"] for i in ids],
        [e[i]["direction"]["good_hits"] for i in ids],
        color=GREY,
        s=28,
        label="direction, same size",
        zorder=3,
    )
    ax.scatter(
        [col[i]["random0"] for i in ids],
        [max(e[i][f"random{k}"]["good_hits"] for k in range(3)) for i in ids],
        color=GREY_L,
        s=20,
        label="random set",
        zorder=3,
    )
    ax.set_xscale("log")
    ax.set_xlabel("collateral: mean KL on 200 unrelated web prompts (nats)")
    ax.set_ylabel("on-topic replies (of 15)")
    ax.legend(fontsize=8, frameon=False)
    for i in ids:
        print(
            f"fig3b {e[i]['label']}: community KL {col[i]['community']:.3f} hits {e[i]['community']['good_hits']}; direction KL {col[i]['direction']:.3f} hits {e[i]['direction']['good_hits']}; random KL {col[i]['random0']:.4f}"
        )
    fig.tight_layout()
    fig.savefig(FIG / "e2b_e3_steer.png", dpi=160, bbox_inches="tight")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    summary()
    fig_e1()
    fig_e2a()
    if (RES / "e2b.json").exists():
        fig_e2b_e3()


if __name__ == "__main__":
    main()
