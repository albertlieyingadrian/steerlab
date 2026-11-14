"""Draw the note's figures from results/ and print every number plotted.

    .venv/bin/python make_figures.py            # reads results/<model>/*.json and results/laws.json

size_law.png    each task x probe: probe error at 31,623 labels against parameters (log-log), with the size law
                fitted on all five sizes
label_law.png   each task x probe: probe error against labels for every model, with the label law fitted on all cells
extrap.png      one size up: the size law's predicted floor against the measured floor, for 7B (from 0.5B-3B) and
                14B (from 0.5B-7B), with the line-through-two-largest baseline; the shaded band is within 0.02
cheaper.png     the cheaper-path test: share of kept cases each method gets right, with Wilson 95% intervals
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fit_laws import ORDER, PROBES, TASKS, law, load

HERE = Path(__file__).resolve().parent
R = HERE / "results"
OUT = HERE / "report" / "figures"
EMBER, BLUE, GREY, INK, SOFT = "#C9561C", "#4A86E8", "#9A948C", "#1d272a", "#586468"
PROBE_NAME = {"lasttoken": "last-token probe", "alllayer": "all-layer probe"}
TASK_NAME = {"truth": "Truth", "sycophancy": "Sycophancy", "policy": "Policy violation"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": SOFT,
                     "savefig.dpi": 200})


def size_law(laws):
    fig, axes = plt.subplots(1, len(TASKS), figsize=(11, 3.2), sharey=True)
    print("size_law.png: error at D = 31,623")
    for ax, t in zip(axes, TASKS):
        for p, colour in (("alllayer", EMBER), ("lasttoken", INK)):
            r = laws["P1_P2"].get(f"{t}/{p}")
            if not r:
                continue
            N, E = np.array(r["N"]), np.array(r["E_at_Dmax"])
            ax.plot(N, E, "o", color=colour, label=PROBE_NAME[p])
            xs = np.geomspace(N.min() * 0.8, N.max() * 1.25, 50)
            ax.plot(xs, r["A"] * xs ** -r["alpha"], "-", color=colour, alpha=0.6, lw=1)
            print(f"  {t:10s} {p:9s} N {np.round(N, 2).tolist()} E {np.round(E, 4).tolist()} "
                  f"A {r['A']:.3f} alpha {r['alpha']:.3f} R2 {r['r2']:.2f}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(TASK_NAME[t])
        ax.set_xlabel("parameters (billions)")
    axes[0].set_ylabel("probe error (1 − AUROC)\nat 31,623 labels")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "size_law.png")
    plt.close(fig)


def label_law(cells, laws):
    fig, axes = plt.subplots(len(PROBES), len(TASKS), figsize=(11, 6), sharex=True, sharey=True)
    cmap = plt.get_cmap("copper_r")
    print("label_law.png: error per (model, D)")
    for i, p in enumerate(("lasttoken", "alllayer")):
        for j, t in enumerate(TASKS):
            ax = axes[i, j]
            s, r = cells.get((t, p), {}), laws["P1_P2"].get(f"{t}/{p}")
            ms = [m for m in ORDER if m in s]
            for mi, m in enumerate(ms):
                c = s[m]
                colour = cmap(0.2 + 0.7 * mi / max(1, len(ms) - 1))
                ax.plot(c["D"], c["E"], "o", ms=3, color=colour, label=m.upper())
                if r:
                    xs = np.geomspace(30, 35000, 60)
                    ax.plot(xs, law(c["N"], xs, r["A"], r["alpha"], r["B"], r["g"]), "-", lw=1, color=colour,
                            alpha=0.7)
                print(f"  {t:10s} {p:9s} {m:4s} " + " ".join(f"{e:.3f}" for e in c["E"]))
            ax.set_xscale("log")
            ax.set_ylim(0, 0.52)
            if i == 0:
                ax.set_title(TASK_NAME[t])
            if i == len(PROBES) - 1:
                ax.set_xlabel("labels D")
            if j == 0:
                ax.set_ylabel(f"probe error\n({PROBE_NAME[p]})")
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "label_law.png")
    plt.close(fig)


def extrap(laws):
    preds = laws["P3"]["preds"]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    lo = min(min(p["floor_meas"], p["floor_pred"], p["line_pred"]) for p in preds) * 0.85
    hi = max(max(p["floor_meas"], p["floor_pred"], p["line_pred"]) for p in preds) * 1.1
    xs = np.array([lo, hi])
    ax.fill_between(xs, xs - 0.02, xs + 0.02, color=GREY, alpha=0.18, lw=0, label="within 0.02")
    ax.plot(xs, xs, color=SOFT, lw=0.8)
    print("extrap.png: measured vs predicted floor")
    for p in preds:
        mk = "o" if p["k"] == 3 else "s"
        ax.plot(p["floor_meas"], p["line_pred"], mk, color=BLUE, mfc="none", ms=5)
        ax.plot(p["floor_meas"], p["floor_pred"], mk, color=EMBER, ms=5)
        print(f"  {p['task']:10s} {p['probe']:9s} -> {p['next']:4s} meas {p['floor_meas']:.4f} "
              f"law {p['floor_pred']:.4f} line {p['line_pred']:.4f} persist {p['persist_pred']:.4f}")
    ax.plot([], [], "o", color=EMBER, label="size law")
    ax.plot([], [], "o", color=BLUE, mfc="none", label="line through two largest")
    ax.plot([], [], "s", color=GREY, label="squares: 14B; circles: 7B")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("measured error at 31,623 labels")
    ax.set_ylabel("predicted from the smaller models")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "extrap.png")
    plt.close(fig)


def cheaper(laws):
    s = laws["P4"]["summary"]
    names = [("laws", "the paper’s laws", EMBER), ("additive", "additive (Chinchilla-style)", GREY),
             ("kaplan", "Kaplan joint form", GREY)]
    fig, ax = plt.subplots(figsize=(5.2, 2.4))
    print("cheaper.png: right / kept cases")
    for i, (k, _, colour) in enumerate(names):
        v = s[k]
        if not v["n"]:
            continue
        ax.barh(i, v["share"], color=colour)
        ax.plot(v["ci"], [i, i], color=INK, lw=1)
        ax.text(1.02, i, f"{v['k']}/{v['n']}", va="center", fontsize=8)
        print(f"  {k:9s} {v['k']}/{v['n']} {v['share']:.3f} {np.round(v['ci'], 3).tolist()}")
    ax.set_yticks(range(len(names)), [n[1] for n in names])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("share of kept cases where the cheaper option was picked")
    fig.tight_layout()
    fig.savefig(OUT / "cheaper.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cells = load(R)
    laws = json.loads((R / "laws.json").read_text())
    size_law(laws)
    label_law(cells, laws)
    extrap(laws)
    cheaper(laws)


if __name__ == "__main__":
    main()
