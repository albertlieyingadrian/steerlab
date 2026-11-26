"""Figures 1-5, the slider's data (report/data.js) and results/summary.json, from results/seed*/.

Prints every number it plots, so the note's text can be checked against it.

    .venv/bin/python experiments/tiny-moe/make_figures.py
"""

import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "report" / "figures"
SEEDS = sorted(RES.glob("seed*"), key=lambda p: int(p.name[4:]))
EMBER, GREY, BLUE, INK = "#d9622b", "#8a8a8a", "#3b6fb6", "#222222"
VARIANTS = ["dense", "dense-wide", "moe-none", "moe-aux", "moe-bias", "moe-bias-fine"]
LABEL = {
    "dense": "dense\nsame active",
    "dense-wide": "dense\nsame total",
    "moe-none": "MoE\nnone",
    "moe-aux": "MoE\naux loss",
    "moe-bias": "MoE\nbias",
    "moe-bias-fine": "MoE bias\n32 small",
}
CLASSES = ["space", "newline", "vowel", "consonant", "upper", "punct"]


def load(seed_dir, name):
    p = seed_dir / name.format(s=seed_dir.name[4:])
    return json.loads(p.read_text()) if p.exists() else None


def mean_sd(xs):
    m = statistics.mean(xs)
    return m, (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)


def paired_gaps(comp, v):
    """Per eval step: variant minus dense on the same seed and validation windows, one list per step (over seeds)."""
    n = len(comp[0]["dense"]["history"])
    return [[c[v]["history"][i]["val"] - c["dense"]["history"][i]["val"] for c in comp] for i in range(n)]


def fig_compare(comp, out):
    """Figure 1: (a) each variant's paired gap to dense over training (mean of seeds, band = seed range);
    (b) the final paired difference per seed."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.5), gridspec_kw={"width_ratios": [1.35, 1]})
    steps = [r["step"] for r in comp[0]["dense"]["history"]]
    shades = {
        "moe-none": "#f0a77c",
        "moe-aux": "#e07a3f",
        "moe-bias": EMBER,
        "moe-bias-fine": "#8f3a12",
        "dense-wide": GREY,
    }
    for v, col in shades.items():
        g = paired_gaps(comp, v)[1:]
        xs = steps[1:]
        a.fill_between(xs, [min(x) for x in g], [max(x) for x in g], color=col, alpha=0.12, lw=0)
        a.plot(xs, [statistics.mean(x) for x in g], "-o", ms=2.5, lw=1.4, color=col, label=v)
    a.axhline(0, color=INK, lw=0.8)
    a.set_xlabel("training step", fontsize=8)
    a.set_ylabel("validation loss minus dense\n(same seed and windows; below 0 = better)", fontsize=8)
    a.set_title("a  the MoEs lead early, then the gap closes", fontsize=9, loc="left")
    a.legend(fontsize=7, frameon=False, ncol=2, loc="lower right")
    style(a)
    for i, v in enumerate(VARIANTS[1:]):
        d = [c[v]["val"] - c["dense"]["val"] for c in comp]
        col = EMBER if v.startswith("moe") else GREY
        b.scatter(d, [i] * len(d), color=col, s=16, zorder=3)
        m, _ = mean_sd(d)
        b.plot([m, m], [i - 0.25, i + 0.25], color=col, lw=2)
    b.axvline(0, color=INK, lw=0.8)
    b.set_yticks(range(len(VARIANTS) - 1), [LABEL[v].replace("\n", " ") for v in VARIANTS[1:]], fontsize=7)
    b.invert_yaxis()
    b.set_xlabel("final validation loss minus dense, step 3,000", fontsize=8)
    b.set_title("b  at the end: a tie; bar = mean of seeds", fontsize=9, loc="left")
    style(b)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def mean_load(row):
    return statistics.mean(la["max_load"] for la in row["layers"])


def fig_balance(comp, out):
    """Figure 2: mean-of-layers max load over training for the three balancing choices."""
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    cols = {"moe-none": GREY, "moe-aux": BLUE, "moe-bias": EMBER}
    for v, col in cols.items():
        for j, c in enumerate(comp):
            h = c[v]["history"]
            ax.plot(
                [r["step"] for r in h],
                [mean_load(r) for r in h],
                "-o",
                ms=2.4,
                lw=1.2,
                color=col,
                alpha=0.85,
                label=v.replace("moe-", "") if j == 0 else None,
            )
    ax.axhline(1, color=INK, lw=0.6, ls=":")
    ax.set_ylim(0, 4.1)
    ax.set_xlabel("training step", fontsize=8)
    ax.set_ylabel("max load, mean of 4 layers\n(1 = even, 4 = worst for top-2)", fontsize=8)
    ax.legend(fontsize=8, frameon=False, title="balancing", title_fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_routing(routes, out):
    """Figure 3: (a) normalised MI(token; expert) per layer: trained, untrained, shuffled; (b, c) class x expert tables
    for layers 0 and 3, first seed."""
    fig = plt.figure(figsize=(10, 3.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1, 1])
    a = fig.add_subplot(gs[0])
    L = len(routes[0])
    for key, col, lab in (
        ("normalised", EMBER, "trained"),
        ("untrained_model", GREY, "untrained, same shape"),
        ("shuffled_control", INK, "shuffled labels"),
    ):
        for j, r in enumerate(routes):
            a.plot(
                range(L),
                [la["specialisation"][key] for la in r],
                "-o",
                ms=3,
                color=col,
                alpha=0.85,
                label=lab if j == 0 else None,
            )
    a.set_xticks(range(L))
    a.set_xlabel("layer", fontsize=8)
    a.set_ylabel("MI(token; expert) / H(expert)", fontsize=8)
    a.set_ylim(0, 1.02)
    a.legend(fontsize=7, frameon=False)
    a.set_title("a  how much the character decides the expert", fontsize=9, loc="left")
    style(a)
    for k, li in enumerate((0, L - 1)):
        ax = fig.add_subplot(gs[k + 1])
        t = routes[0][li]["class_table"]
        im = ax.imshow(t, cmap="Oranges", vmin=0, vmax=0.5, aspect="auto")
        ax.set_yticks(range(len(CLASSES)), CLASSES if k == 0 else [""] * len(CLASSES), fontsize=7)
        ax.set_xticks(range(len(t[0])))
        ax.tick_params(labelsize=7)
        ax.set_xlabel("expert", fontsize=8)
        ax.set_title(f"{'bc'[k]}  layer {li}: share of each class's slots", fontsize=9, loc="left")
        if k == 1:
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_ablate(abls, out):
    """Figure 4: loss change on an expert's own tokens vs on the rest, every routed expert of every layer and seed."""
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    mx = 0
    for rows in abls:
        for r in rows:
            if r["expert"] == "shared":
                continue
            col = EMBER if r["layer"] == 0 else "#f0a77c"
            ax.scatter(r["other_tokens"], r["own_tokens"], s=14, color=col, alpha=0.85)
            mx = max(mx, r["own_tokens"], r["other_tokens"])
    ax.plot([0, mx], [0, mx], color=INK, lw=0.7, ls=":")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_xlabel("loss change on all other positions", fontsize=8)
    ax.set_ylabel("loss change on the positions\nthe expert served", fontsize=8)
    ax.scatter([], [], color=EMBER, label="layer 0")
    ax.scatter([], [], color="#f0a77c", label="layers 1-3")
    ax.legend(fontsize=8, frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_collapse(cols, out):
    """Figure 5: worst-layer max load over training from a fixed head start for expert 0."""
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    colors = {"none": GREY, "none-skew": INK, "aux-skew": BLUE, "bias-skew": EMBER}
    labels = {
        "none": "no head start, no balancing",
        "none-skew": "head start, no balancing",
        "aux-skew": "head start, aux loss",
        "bias-skew": "head start, bias",
    }
    for name, col in colors.items():
        for j, c in enumerate(cols):
            h = c[name]["history"]
            ax.plot(
                [r["step"] for r in h],
                [max(la["max_load"] for la in r["layers"]) for r in h],
                "-o",
                ms=2.4,
                lw=1.2,
                color=col,
                alpha=0.85,
                label=labels[name] if j == 0 else None,
            )
    ax.axhline(1, color=INK, lw=0.6, ls=":")
    ax.set_ylim(0, 4.3)
    ax.set_xlabel("training step", fontsize=8)
    ax.set_ylabel("max load, worst of 4 layers", fontsize=8)
    ax.legend(fontsize=7, frameon=False, loc="center right")
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    comp = [load(d, "compare_s{s}.json") for d in SEEDS]
    routes = [load(d, "route_lab.json") for d in SEEDS]
    none_routes = [load(d, "none/route_lab.json") for d in SEEDS]
    abls = [load(d, "ablate.json") for d in SEEDS]
    cols = [load(d, "collapse_s{s}.json") for d in SEEDS]
    mtps = [load(d, "mtp_s{s}.json") for d in SEEDS]
    seeds = [d.name[4:] for d in SEEDS]
    S = {"seeds": seeds}
    print(f"seeds: {seeds}")

    print("\nFigure 1 / Table: variant, params, active, final val per seed, mean ± sd, best (step), max load")
    S["compare"] = {}
    for v in VARIANTS:
        vals = [c[v]["val"] for c in comp]
        best = [min(c[v]["history"], key=lambda r: r["val"]) for c in comp]
        loads = [c[v]["max_load"] for c in comp]
        diff = [c[v]["val"] - c["dense"]["val"] for c in comp]
        m, sd = mean_sd(vals)
        dm, dsd = mean_sd(diff)
        S["compare"][v] = {
            "params": comp[0][v]["params"],
            "active": comp[0][v]["active"],
            "val": vals,
            "mean": m,
            "sd": sd,
            "diff": diff,
            "diff_mean": dm,
            "diff_sd": dsd,
            "best": [b["val"] for b in best],
            "best_step": [b["step"] for b in best],
            "max_load": loads,
            "seconds": [c[v]["seconds"] for c in comp],
        }
        print(
            f"  {v:14s} {comp[0][v]['params']:>9,} {comp[0][v]['active']:>9,}  "
            + " ".join(f"{x:.3f}" for x in vals)
            + f"  {m:.3f} ± {sd:.3f}  diff {dm:+.3f} ± {dsd:.3f}  best "
            + " ".join(f"{b['val']:.3f}@{b['step']}" for b in best)
            + ("  load " + " ".join(f"{x:.2f}" for x in loads) if loads[0] is not None else "")
        )
    print("\nFigure 1a: paired gap to dense per eval step, mean [min, max] over seeds")
    S["gaps"] = {"steps": [r["step"] for r in comp[0]["dense"]["history"]]}
    for v in VARIANTS[1:]:
        g = paired_gaps(comp, v)
        S["gaps"][v] = g
        print(f"  {v:14s} " + " ".join(f"{statistics.mean(x):+.3f}" for x in g))
    fig_compare(comp, FIG / "fig1-compare.png")

    print("\nFigure 2: mean-of-layers max load at the last eval, per seed")
    for v in ("moe-none", "moe-aux", "moe-bias", "moe-bias-fine"):
        print(
            f"  {v:14s} "
            + " ".join(f"{mean_load(c[v]['history'][-1]):.2f}" for c in comp)
            + "   dead experts (sum over layers): "
            + " ".join(str(sum(la["dead"] for la in c[v]["history"][-1]["layers"])) for c in comp)
        )
    fig_balance(comp, FIG / "fig2-balance.png")

    print("\nFigure 3a: normalised MI per layer, trained | untrained | shuffled")
    S["routing"] = []
    for s, r in zip(seeds, routes, strict=True):
        row = [{k: la["specialisation"][k] for k in ("normalised", "untrained_model", "shuffled_control")} for la in r]
        S["routing"].append(row)
        print(
            f"  seed {s}: "
            + "  ".join(f"{x['normalised']:.2f}|{x['untrained_model']:.2f}|{x['shuffled_control']:.3f}" for x in row)
        )
    print("  moe-none load shares, worst layer per seed:")
    for s, r in zip(seeds, none_routes, strict=True):
        w = max(r, key=lambda la: la["load"]["max_load"])
        print(
            f"    seed {s}: layer {w['layer']} max load {w['load']['max_load']:.2f} dead {w['load']['dead']} "
            f"shares {w['load']['shares']}"
        )
    print("  layer 0 top tokens per expert, first seed:", routes[0][0]["top_tokens"])
    print("  last layer top tokens per expert, first seed:", routes[0][-1]["top_tokens"])
    fig_routing(routes, FIG / "fig3-routing.png")

    print("\nFigure 4: per seed, own > rest count, mean own, mean rest, ratio, shared and routed per layer")
    S["ablate"] = []
    for s, rows in zip(seeds, abls, strict=True):
        routed = [r for r in rows if r["expert"] != "shared"]
        shared = [r for r in rows if r["expert"] == "shared"]
        own = statistics.mean(r["own_tokens"] for r in routed)
        rest = statistics.mean(r["other_tokens"] for r in routed)
        hits = sum(r["own_tokens"] > r["other_tokens"] for r in routed)
        per_layer = [statistics.mean(r["delta"] for r in routed if r["layer"] == li) for li in range(len(shared))]
        S["ablate"].append(
            {
                "hits": hits,
                "n": len(routed),
                "own": own,
                "rest": rest,
                "shared": [r["delta"] for r in shared],
                "routed_per_layer": per_layer,
                "routed_mean": statistics.mean(r["delta"] for r in routed),
            }
        )
        print(
            f"  seed {s}: {hits}/{len(routed)}  own {own:+.3f}  rest {rest:+.3f}  ratio {own / rest:.1f}  shared "
            + " ".join(f"{r['delta']:+.3f}" for r in shared)
            + "  routed "
            + " ".join(f"{x:+.3f}" for x in per_layer)
        )
    fig_ablate(abls, FIG / "fig4-ablate.png")

    print("\nFigure 5: collapse runs, final worst-layer max load (val) per seed; dead experts in the worst layer")
    S["collapse"] = {}
    for name in ("none", "none-skew", "aux-skew", "bias-skew"):
        fin = [c[name]["worst_max_load"] for c in cols]
        vals = [c[name]["val"] for c in cols]
        dead = [max(la["dead"] for la in c[name]["history"][-1]["layers"]) for c in cols]
        first_below = []
        for c in cols:
            h = c[name]["history"]
            first_below.append(next((r["step"] for r in h if max(la["max_load"] for la in r["layers"]) < 2.0), None))
        S["collapse"][name] = {"worst_max_load": fin, "val": vals, "dead": dead, "first_below_2": first_below}
        print(
            f"  {name:10s} "
            + "  ".join(f"{a:.2f} ({b:.3f})" for a, b in zip(fin, vals, strict=True))
            + f"  dead {dead}  first eval below 2.0: {first_below}"
        )
    fig_collapse(cols, FIG / "fig5-collapse.png")

    print("\nMTP: main val without | with | MTP loss t+2 | accuracy t+2 | agreement with the main model")
    S["mtp"] = []
    for s, m in zip(seeds, mtps, strict=True):
        S["mtp"].append(
            {
                "without": m["0"]["val"],
                "with": m["1"]["val"],
                "mtp_val": m["mtp_val"],
                "acc": m["mtp_acc"],
                "agree": m["mtp_agree"],
            }
        )
        print(
            f"  seed {s}: {m['0']['val']:.3f} | {m['1']['val']:.3f} | {m['mtp_val']:.3f} | {m['mtp_acc']:.3f} | "
            f"{m['mtp_agree']:.3f}"
        )

    # the slider: per variant, per eval step, per layer, the load share of each expert (first seed)
    slider = {"steps": [r["step"] for r in comp[0]["moe-bias"]["history"]]}
    for v in ("moe-none", "moe-aux", "moe-bias"):
        slider[v] = [[la["shares"] for la in r["layers"]] for r in comp[0][v]["history"]]
    (HERE / "report" / "data.js").write_text("window.MOE_LOADS = " + json.dumps(slider, separators=(",", ":")) + ";\n")
    (RES / "summary.json").write_text(json.dumps(S, indent=1))
    print(f"\nwrote {FIG}/fig1-5, report/data.js, results/summary.json")
    _ = math


if __name__ == "__main__":
    main()
