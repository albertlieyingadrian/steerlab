"""
Draw every figure in the note from results/, and print the numbers each one plots.

    .venv/bin/python experiments/compositional-rl/analyse.py        # writes results/summary.json first
    .venv/bin/python experiments/compositional-rl/make_figures.py

Figures (report/figures/):
    toy.png        the worked example: chance a chain of T steps is all right, if each step is right with probability p
    levels.png     accuracy against length for each level, atomic and compound, Base / RL-Atomic / RL-Compound
    mix.png        Level II accuracy against the share of composed tasks in training (0, 0.10, 0.25, 1)
    steptest.png   the step test: scaffold accuracy S (every step right on the true state) against in-context C
    swaps.png      layer-block adapter swaps between RL-Atomic and RL-Compound
    curves.png     training reward and reply length per step, every run
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE / "results"
FIG = HERE / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
INK, MUTED, EMBER, EMBER_L, BLUE, GREY = "#1d1a17", "#7a7065", "#d26f2a", "#f0b48a", "#4a86e8", "#b9b1a7"
STYLE = {  # ember = the arm the note is about (composed training); grey and blue = baseline and control
    "base": {"color": GREY, "ls": ":", "label": "Base (no training)"},
    "atomic": {"color": BLUE, "ls": "-", "label": "RL-Atomic (skills only)"},
    "compound": {"color": EMBER, "ls": "-", "label": "RL-Compound (chains)"},
    "mix10": {"color": EMBER_L, "ls": "--", "label": "Mix-10"},
    "mix25": {"color": "#b5561c", "ls": "--", "label": "Mix-25"},
}
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 200,
    }
)


def load(path):
    """Rows of a .jsonl file, or of its gzipped copy (results are stored as .jsonl.gz)."""
    import gzip

    path = Path(path)
    if not path.exists() and Path(str(path) + ".gz").exists():
        path = Path(str(path) + ".gz")
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def summary():
    return json.loads((R / "summary.json").read_text())


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def runs_by_arm(level):
    out = defaultdict(list)
    for d in sorted((R / "runs").glob("*")):
        f = d / "eval.jsonl"
        if not f.exists() and not Path(str(f) + ".gz").exists():
            continue
        if d.name == f"base-l{level}":
            out["base"].append(load(f))
        elif d.name.startswith(f"l{level}-"):
            out[d.name.split("-")[1]].append(load(f))
    return out


def fig_toy():
    T = np.arange(1, 21)
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for p, c in [(0.99, GREY), (0.95, BLUE), (0.90, EMBER)]:
        ax.plot(T, p**T, color=c, lw=2, label=f"each step right {p:.0%} of the time")
        print(f"toy p={p}: T=4 {p**4:.3f}, T=12 {p**12:.3f}")
    ax.set_xlabel("steps in the chain (T)")
    ax.set_ylabel("chance every step is right (p^T)")
    ax.set_ylim(0, 1.02)
    ax.set_xticks([1, 4, 8, 12, 16, 20])
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("One wrong step spoils the rest", loc="left", fontsize=10)
    save(fig, "toy.png")


def seed_band(runs, kind):
    """Mean accuracy by length, and the min and max over training seeds."""
    per = []
    for rows in runs:
        acc = defaultdict(list)
        for r in rows:
            if r["kind"] == kind:
                acc[r["length"]].append(r["correct"])
        per.append({L: np.mean(v) for L, v in acc.items()})
    Ls = sorted(set.intersection(*[set(p) for p in per]))
    m = np.array([[p[L] for L in Ls] for p in per])
    return Ls, m.mean(0), m.min(0), m.max(0)


def fig_levels():
    levels = [lv for lv in (1, 2, 3) if runs_by_arm(lv)]
    fig, axs = plt.subplots(2, len(levels), figsize=(3.3 * len(levels), 5.2), sharey=True, squeeze=False)
    names = {1: "Level I: one skill repeated", 2: "Level II: insert + delete", 3: "Level III: segment tree"}
    for j, lv in enumerate(levels):
        arms = runs_by_arm(lv)
        for i, kind in enumerate(["atomic", "compound"]):
            ax = axs[i, j]
            for arm in ["base", "atomic", "compound"]:
                if arm not in arms:
                    continue
                Ls, mu, lo, hi = seed_band(arms[arm], kind)
                st = STYLE[arm]
                ax.plot(Ls, mu, color=st["color"], ls=st["ls"], lw=2, marker="o", ms=3, label=st["label"])
                if len(arms[arm]) > 1:
                    ax.fill_between(Ls, lo, hi, color=st["color"], alpha=0.18, lw=0)
                print(f"levels L{lv} {kind} {arm}: " + ", ".join(f"{L}:{v:.2f}" for L, v in zip(Ls, mu)))
            ax.axvspan(1.5, 6.5 if lv < 3 else 5.5, color="#f3eee6", zorder=0)
            ax.set_ylim(0, 1.02)
            if i == 0:
                ax.set_title(names[lv], loc="left", fontsize=9.5)
            if j == 0:
                ax.set_ylabel(f"{kind} accuracy")
            if i == 1:
                ax.set_xlabel("length (shaded: seen in training)")
    axs[0, levels.index(2) if 2 in levels else 0].legend(frameon=False, fontsize=7.5, loc="lower left")
    save(fig, "levels.png")


def fig_mix():
    s = summary().get("level2")
    if not s:
        return
    arms = s["all"]["arms"]
    order = [(a, f) for a, f in [("atomic", 0.0), ("mix10", 0.10), ("mix25", 0.25), ("compound", 1.0)] if a in arms]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, kind in zip(axs, ["compound", "atomic"]):
        for k, (a, f) in enumerate(order):
            v = arms[a][kind]
            col = STYLE[a]["color"]
            ax.errorbar(
                k,
                v["acc"],
                yerr=[[v["acc"] - v["ci"][0]], [v["ci"][1] - v["acc"]]],
                fmt="o",
                color=col,
                ms=7,
                capsize=3,
                lw=1.6,
            )
            ax.scatter([k] * len(v["per_seed"]), v["per_seed"], s=10, color=col, alpha=0.5, zorder=3)
            print(
                f"mix {kind} {a} (f={f}): {v['acc']:.3f} {np.round(v['ci'], 3).tolist()} "
                f"seeds {np.round(v['per_seed'], 3).tolist()}"
            )
        if "base" in arms:
            ax.axhline(arms["base"][kind]["acc"], color=GREY, ls=":", lw=1.5, label="Base")
        ax.set_xticks(range(len(order)), [f"{f:.0%}" for _, f in order])
        ax.set_xlabel("share of composed tasks in training")
        ax.set_ylabel(f"{kind} accuracy (Level II, all lengths)")
        ax.set_ylim(0, None)
    axs[0].legend(frameon=False, fontsize=8)
    save(fig, "mix.png")


def fig_steptest():
    m = summary().get("mechanism", {}).get("step_test", {})
    arms = [a for a in ["base", "atomic", "mix25", "compound"] if a in m]
    if not arms:
        return
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.1), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axs[0]
    w = 0.36
    for k, a in enumerate(arms):
        v = m[a]
        col = STYLE[a]["color"]
        ax.bar(
            k - w / 2,
            v["S"],
            w,
            color="white",
            edgecolor=col,
            hatch="///",
            lw=1.4,
            label="steps alone, on the true state (S)" if k == 0 else None,
        )
        ax.bar(k + w / 2, v["C"], w, color=col, label="whole chain in one reply (C)" if k == 0 else None)
        for off, key in [(-w / 2, "S"), (w / 2, "C")]:
            lo, hi = v[f"{key}_ci"]
            ax.plot([k + off, k + off], [lo, hi], color=INK, lw=1)
        print(
            f"steptest {a}: S {v['S']:.3f} {np.round(v['S_ci'], 3).tolist()}  C {v['C']:.3f} "
            f"{np.round(v['C_ci'], 3).tolist()}  gap {v['gap_S_minus_C']:+.3f} {np.round(v['gap_ci'], 3).tolist()}"
            f"  step acc {v['step_acc']:.3f}"
        )
    ax.set_xticks(range(len(arms)), [STYLE[a]["label"].split(" (")[0] for a in arms], fontsize=8)
    ax.set_ylabel("share of Level II chains right (lengths 4-8)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.set_ylim(0, 1)
    ax = axs[1]
    for a in arms:
        v = m[a]
        Ls = sorted(v["S_by_length"], key=int)
        xs = [int(L) for L in Ls]
        ax.plot(xs, [v["S_by_length"][L] for L in Ls], color=STYLE[a]["color"], ls="--", lw=1.4, marker="s", ms=3)
        ax.plot(
            xs,
            [v["C_by_length"][L] for L in Ls],
            color=STYLE[a]["color"],
            ls="-",
            lw=2,
            marker="o",
            ms=3,
            label=STYLE[a]["label"].split(" (")[0],
        )
    ax.set_xlabel("chain length (dashed = steps alone S, solid = one reply C)")
    ax.set_ylabel("share right")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=7.5)
    save(fig, "steptest.png")


def fig_swaps():
    mech = summary().get("mechanism", {})
    sw, st = mech.get("swaps", {}), mech.get("step_test", {})
    if not sw or "atomic" not in st:
        return
    a_end, c_end = st["atomic"]["C"], st["compound"]["C"]
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    labels = ["layers 0-9", "layers 10-18", "layers 19-27"]
    for direction, col, name in [
        ("AplusC", EMBER, "RL-Atomic with one third from RL-Compound"),
        ("CplusA", BLUE, "RL-Compound with one third from RL-Atomic"),
    ]:
        vals = []
        for t in ["t0", "t1", "t2"]:
            seeds = [v for tag, v in sw.items() if tag.startswith(f"swap-{direction}-{t}-")]
            vals.append(np.mean(seeds) if seeds else np.nan)
        ax.plot(range(3), vals, color=col, marker="o", lw=2, label=name)
        print(f"swaps {direction}: " + ", ".join(f"{lab} {v:.3f}" for lab, v in zip(labels, vals)))
    ax.axhline(a_end, color=BLUE, ls=":", lw=1.2)
    ax.axhline(c_end, color=EMBER, ls=":", lw=1.2)
    ax.text(2.08, a_end, "RL-Atomic", color=BLUE, fontsize=7.5, va="center")
    ax.text(2.08, c_end, "RL-Compound", color=EMBER, fontsize=7.5, va="center")
    ax.set_xticks(range(3), labels)
    ax.set_ylabel("Level II chain accuracy (lengths 4-8)")
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=7.5)
    save(fig, "swaps.png")


def fig_curves():
    runs = sorted((R / "runs").glob("l*-*-s*"))
    if not runs:
        return
    levels = sorted({d.name[:2] for d in runs})
    fig, axs = plt.subplots(2, len(levels), figsize=(3.3 * len(levels), 4.6), squeeze=False)
    for j, lv in enumerate(levels):
        for d in runs:
            if not d.name.startswith(lv) or not ((d / "curve.jsonl").exists() or (d / "curve.jsonl.gz").exists()):
                continue
            arm = d.name.split("-")[1]
            c = load(d / "curve.jsonl")
            steps = [r["step"] for r in c]
            sm = np.convolve([r["reward"] for r in c], np.ones(5) / 5, mode="valid")
            axs[0, j].plot(steps[4:], sm, color=STYLE[arm]["color"], ls=STYLE[arm]["ls"], lw=1.2, alpha=0.9)
            axs[1, j].plot(steps, [r["mean_tokens"] for r in c], color=STYLE[arm]["color"], ls=STYLE[arm]["ls"], lw=1.2)
        axs[0, j].set_title(f"Level {'I' * int(lv[1])}", loc="left", fontsize=9.5)
        axs[0, j].set_ylabel("training reward (5-step mean)")
        axs[1, j].set_ylabel("reply tokens")
        axs[1, j].set_xlabel("GRPO step")
    handles = [
        plt.Line2D([], [], color=STYLE[a]["color"], ls=STYLE[a]["ls"], label=STYLE[a]["label"])
        for a in ["atomic", "mix10", "mix25", "compound"]
    ]
    axs[0, 0].legend(handles=handles, frameon=False, fontsize=7)
    save(fig, "curves.png")


def demo_data():
    """The interactive demo's data (section 4.4), written into report/index.html's chain-data script tag."""
    import re

    m = summary().get("mechanism", {}).get("step_test", {})
    if not m:
        return
    labels = {"base": "Untrained", "atomic": "RL-Atomic", "mix25": "Mix-25", "compound": "RL-Compound"}
    data = {}
    for a in ["atomic", "compound", "mix25", "base"]:
        if a in m:
            v = m[a]
            Ls = sorted(v["S_by_length"], key=int)
            data[a] = {
                "label": labels[a],
                "color": STYLE[a]["color"],
                "p": round(v["step_acc"], 2),
                "L": [int(L) for L in Ls],
                "S": [round(v["S_by_length"][L], 3) for L in Ls],
                "C": [round(v["C_by_length"][L], 3) for L in Ls],
            }
    page = HERE / "report" / "index.html"
    text = page.read_text()
    new = re.sub(
        r'(<script type="application/json" id="chain-data">).*?(</script>)',
        lambda mm: mm.group(1) + json.dumps(data) + mm.group(2),
        text,
        flags=re.DOTALL,
    )
    page.write_text(new)
    print("demo data:", json.dumps(data))


if __name__ == "__main__":
    demo_data()
    fig_toy()
    fig_levels()
    fig_mix()
    fig_steptest()
    fig_swaps()
    fig_curves()
