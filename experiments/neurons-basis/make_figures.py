"""Draw the note's figures from results/, printing every number plotted.

.venv/bin/python experiments/neurons-basis/make_figures.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "report" / "figures"
EMBER, EMBER_L, GREY, GREY_L, INK, BLUE = "#d26f2a", "#f2b48e", "#6f675f", "#c9c0b5", "#1d1a17", "#3b6ea8"
plt.rcParams.update(
    {"font.family": "sans-serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
)
NAMES = {"olmoe": "OLMoE-1B-7B (MoE)", "qwen05": "Qwen2.5-0.5B (dense)"}


def load(name):
    return json.loads((RES / name).read_text())


def fig_sparsity(s):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
    for ax, key in zip(axes, ("olmoe", "qwen05"), strict=True):
        sp = s[key]["sparsity"]
        fr = [0.25, 0.125, 0.078, 0.04, 0.02]
        y = [sp[f"top{f:g}"]["dloss"] for f in fr]
        ax.errorbar(
            [100 * f for f in fr],
            [v[0] for v in y],
            yerr=[[v[0] - v[1] for v in y], [v[2] - v[0] for v in y]],
            color=EMBER,
            marker="o",
            label="keep each token's strongest K",
        )
        bands = [75, 90, 95, 97, 99]
        yb = [sp[f"band{x}"]["dloss"] for x in bands]
        ax.plot(
            [2 * (100 - x) for x in bands],
            [v[0] for v in yb],
            color=EMBER_L,
            marker="s",
            ls="--",
            label="keep each neuron's own tails (pX/100−X)",
        )
        for c, mk in (("random", "x"), ("bottom", "v")):
            for f in (0.078, 0.25):
                v = sp[f"{c}{f:g}"]["dloss"]
                ax.plot(
                    100 * f, v[0], marker=mk, color=GREY, ls="none", label=f"{c} K (control)" if f == 0.078 else None
                )
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xticks([2, 4, 7.8, 12.5, 25, 50])
        ax.set_xticklabels(["2", "4", "7.8", "12.5", "25", "50"])
        ax.invert_xaxis()
        ax.set_xlabel("share of neuron firings kept (%)")
        ax.set_ylabel("increase in next-token loss (nats)")
        ax.set_ylim(0, None)
        ax.axvline(7.8, color=GREY_L, lw=0.8)
        ax.set_title(NAMES[key], fontsize=10)
        print(
            key,
            "top-K dloss",
            [(f, round(v[0], 3)) for f, v in zip(fr, y, strict=True)],
            "bands",
            [(x, round(v[0], 3)) for x, v in zip(bands, yb, strict=True)],
        )
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, fontsize=8, frameon=False, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "sparsity.png", dpi=160)
    plt.close(fig)


def fig_bench(s):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    conds = ["top0.25", "top0.125", "top0.078", "top0.04", "top0.02", "band97", "random0.078"]
    labels = ["25%", "12.5%", "7.8%", "4%", "2%", "p97/3\ntails", "random\n7.8%"]
    for ax, b in zip(axes, ("hellaswag", "arc_easy"), strict=True):
        for j, key in enumerate(("olmoe", "qwen05")):
            d = s[key]["bench"][b]
            vals = [d[c]["delta_points"] for c in conds]
            x = np.arange(len(conds)) + (j - 0.5) * 0.36
            cols = [EMBER if j == 0 else EMBER_L] * (len(conds) - 1) + [GREY]
            ax.bar(x, [v[0] for v in vals], width=0.34, color=cols, label=NAMES[key])
            ax.errorbar(
                x,
                [v[0] for v in vals],
                yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]],
                fmt="none",
                ecolor=INK,
                lw=0.8,
            )
            print(
                b, key, "base", round(d["base"][0], 3), [(c, round(v[0], 1)) for c, v in zip(conds, vals, strict=True)]
            )
        ax.axhspan(-2.2, -1.2, color=BLUE, alpha=0.15, lw=0)
        ax.set_xticks(np.arange(len(conds)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title({"hellaswag": "HellaSwag (500 items)", "arc_easy": "ARC-Easy (500 items)"}[b], fontsize=10)
        ax.axhline(0, color=INK, lw=0.6)
    axes[0].set_ylabel("change in accuracy (points)")
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "benchmarks.png", dpi=160)
    plt.close(fig)


def fig_modes_descriptions(s):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for j, key in enumerate(("olmoe", "qwen05")):
        m = s[key].get("modes", {})
        if "rows" not in m:
            continue
        dom = [r["dominant_damage_share"] for r in m["rows"]]
        axes[0].hist(dom, bins=np.linspace(0, 1, 11), alpha=0.7, color=EMBER if j == 0 else EMBER_L, label=NAMES[key])
        print(
            key,
            "single-mode share",
            [round(x, 3) for x in m["single_mode_ge90"]],
            "modes median/mean",
            m["modes_median"],
            round(m["modes_mean"], 2),
        )
    axes[0].axvline(0.9, color=INK, lw=0.8, ls="--")
    axes[0].set_xlabel("damage share of the main mode")
    axes[0].set_ylabel("neuron poles")
    axes[0].legend(fontsize=8, frameon=False)
    for j, key in enumerate(("olmoe", "qwen05")):
        d = s[key].get("descriptions", {})
        if "rows" not in d:
            continue
        acc = [r["predict_acc"] for r in d["rows"] if r.get("predict_acc") is not None]
        axes[1].hist(acc, bins=np.linspace(0, 1, 11), alpha=0.7, color=EMBER if j == 0 else EMBER_L, label=NAMES[key])
        arms = ["recovered_sim", "recovered_shuffled", "recovered_constant"]
        vals = [d[a][0] if d.get(a) else np.nan for a in arms]
        err = (
            [[d[a][0] - d[a][1] for a in arms], [d[a][2] - d[a][0] for a in arms]]
            if all(d.get(a) for a in arms)
            else None
        )
        x = np.arange(3) + (j - 0.5) * 0.36
        axes[2].bar(x, vals, width=0.34, color=[EMBER if j == 0 else EMBER_L, GREY, GREY_L], label=NAMES[key])
        if err:
            axes[2].errorbar(x, vals, yerr=err, fmt="none", ecolor=INK, lw=0.8)
        print(key, "predict acc mean", round(float(np.mean(acc)), 3), "recovered", [round(v, 3) for v in vals])
    axes[1].axvline(0.5, color=GREY, lw=0.8, ls=":")
    axes[1].set_xlabel("predict accuracy per pole (chance 0.5)")
    axes[2].set_xticks(range(3))
    axes[2].set_xticklabels(["its description", "another pole's\ndescription", "constant\n(its mean)"], fontsize=8)
    axes[2].set_ylabel("ablation damage recovered")
    axes[2].axhline(0, color=INK, lw=0.6)
    fig.tight_layout()
    fig.savefig(FIG / "modes_descriptions.png", dpi=160)
    plt.close(fig)


def fig_monitors():
    rows = list(csv.DictReader((RES / "phase2_monitors.csv").open()))
    picks = [
        ("M1", "residual", "19", "published sandbagging probe recipe, residual stream", GREY_L, "o"),
        ("M1", "neuron-raw", "18", "same recipe, neuron basis", EMBER_L, "o"),
        ("M3", "residual", "17", "trained on the behaviour, residual stream (L17)", GREY, "s"),
        ("M3", "neuron-raw", "16", "trained on the behaviour, neuron basis (L16, pre-registered)", EMBER, "s"),
        ("M3", "neuron-raw", "20", "trained on the behaviour, neuron basis (L20, post hoc)", "#8c3d0f", "D"),
    ]
    whats = [
        ("no password, no prompt: sandbagged vs answered", "BEHAVIOUR\nsandbagged vs\nanswered"),
        ("answered, password: 'answer wrongly' vs no prompt", "instruction\n(with password)"),
        ("answered, no password: 'answer wrongly' vs no prompt", "instruction\n(no password)"),
        ("sandbagged, no password: 'answer wrongly' vs no prompt", "instruction\n(sandbagged rows)"),
        ("answered: no password vs password (one string, same behaviour)", "password\nonly"),
    ]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for i, (m, b, L, lab, col, mk) in enumerate(picks):
        xs, ys, lo_, hi_ = [], [], [], []
        for j, (w, _) in enumerate(whats):
            r = next(r for r in rows if r["monitor"] == m and r["basis"] == b and r["layer"] == L and r["what"] == w)
            a, lo, hi = float(r["auroc"]), float(r["lo"]), float(r["hi"])
            xs.append(j + (i - 2) * 0.13)
            ys.append(a)
            lo_.append(a - lo)
            hi_.append(hi - a)
        ax.errorbar(xs, ys, yerr=[lo_, hi_], fmt=mk, color=col, ms=5, lw=1, ls="none", label=lab)
        print(lab, [round(y, 3) for y in ys])
    ax.axhline(0.5, color=INK, lw=0.7, ls="--")
    ax.axvspan(-0.5, 0.5, color=BLUE, alpha=0.06, lw=0)
    ax.set_xticks(range(len(whats)))
    ax.set_xticklabels([w2 for _, w2 in whats], fontsize=9)
    ax.set_ylabel("AUROC (0.5 = blind to it)")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "monitors.png", dpi=160)
    plt.close(fig)


def fig_validate():
    v = load("phase2_validate.json")
    base = v["base_nopw"]["accuracy"]
    groups = [
        ("monitor's 5\n(L16)", "monitor_top5", "random"),
        ("monitor's 14\n(L16)", "L16_monitor_all", "L16_random14_"),
        ("top-50 password\ndifference (L16)", "L16_pwdiff_top50", "L16_random50_"),
        ("top-500 password\ndifference (L20)", "L20_pwdiff_top500", None),
        ("whole layer 16\n(8,960)", "L16_all", None),
        ("whole layer 20", "L20_all", None),
    ]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for i, (lab, name, rnd) in enumerate(groups):
        if name not in v["sets"]:
            continue
        a = v["sets"][name]["patch_from_pw"]["accuracy"]
        ax.bar(i, a, color=EMBER, width=0.5)
        if rnd:
            r = [v["sets"][k]["patch_from_pw"]["accuracy"] for k in v["sets"] if k.startswith(rnd) and k != name]
            if r:
                ax.plot([i + 0.32] * len(r), r, "_", color=GREY, ms=10)
        print(lab.replace("\n", " "), "patch_from_pw accuracy", round(a, 3))
    ax.axhline(base, color=INK, lw=0.8, ls="--")
    ax.axhline(v["base_pw"]["accuracy"], color=BLUE, lw=0.8, ls=":")
    ax.text(len(groups) - 0.5, base + 0.01, "no edit, no password", fontsize=8, ha="right")
    ax.text(
        len(groups) - 0.5, v["base_pw"]["accuracy"] - 0.05, "no edit, with password", fontsize=8, ha="right", color=BLUE
    )
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=8)
    ax.set_ylabel("quiz accuracy, no password")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Patch neurons at the last prompt token to their value with the password (grey ticks: random sets)", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(FIG / "validate.png", dpi=160)
    plt.close(fig)


def fig_phase35():
    p = RES / "phase35_summary.json"
    if not p.exists():
        return
    d = load("phase35_summary.json")
    tasks = ["arc_challenge", "arc_easy", "winogrande", "hellaswag", "mmlu"]
    labels = ["ARC-C", "ARC-E", "WinoGrande", "HellaSwag", "MMLU"]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    series = [
        ("qwen36", "Qwen3.6-35B-A3B, ours (top 320)", EMBER),
        ("olmoe_on_35b_items", "OLMoE-1B-7B, ours (top 640), same items", EMBER_L),
    ]
    for j, (key, lab, col) in enumerate(series):
        if key not in d:
            continue
        vals = [d[key]["tasks"][t]["delta_points"] for t in tasks]
        x = np.arange(len(tasks)) + (j - 1) * 0.27
        ax.bar(x, [v[0] for v in vals], width=0.25, color=col, label=lab)
        ax.errorbar(
            x,
            [v[0] for v in vals],
            yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]],
            fmt="none",
            ecolor=INK,
            lw=0.8,
        )
        print(lab, [(t, round(v[0], 1)) for t, v in zip(tasks, vals, strict=True)])
    theirs = [d["qwen36"]["tasks"][t]["theirs"] for t in tasks] if "qwen36" in d else []
    ax.bar(
        np.arange(len(tasks)) + 0.27, theirs, width=0.25, color=BLUE, alpha=0.6, label="Qwen3.6-35B-A3B, original study"
    )
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("change in accuracy (points)")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "original_model.png", dpi=160)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    s = load("phase1_summary.json")
    fig_sparsity(s)
    fig_bench(s)
    fig_modes_descriptions(s)
    fig_monitors()
    if (RES / "phase2_validate.json").exists():
        fig_validate()
    fig_phase35()


if __name__ == "__main__":
    main()
