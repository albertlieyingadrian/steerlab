"""Draw the note's figures from results/, printing every number plotted.

    python experiments/rl-hack-monitor/make_figures.py

behaviour.png  hack rate and true accuracy on the 64 eval prompts x 4 samples at every checkpoint, the three proxy
               runs (orange) and the three control runs (grey).
zscores.png    z against the control runs for the hack rate and each monitor, per proxy seed, with the z = 3 line and
               the band of the 20 random directions at the probe layer (central 90% of their |z|); y clipped at -12 and 20.
leads.png      lead time over the hack rate (steps) per monitor and seed, beside the 20 random directions.
persona.png    project 4: final excess of each persona projection in sigma units, beside the random directions.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "report" / "figures"
EMBER, GREY, LIGHT, INK, BLUE = "#C9561C", "#6F6A63", "#C8C2B8", "#2B2723", "#4A86E8"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "savefig.dpi": 150})  # fmt: skip
SEEDS = [0, 1, 2]
MONITORS = [  # (signal, label)
    ("behaviour_hack", "hack rate (behaviour)"),
    ("greybox_tag_logprob", "log P(reply starts <PASS>)"),
    ("probe@prompt_last", "probe, last prompt token"),
    ("probe@fixed_mean", "probe, fixed base replies"),
    ("probe@own_mean", "probe, own replies"),
    ("persona_evil@prompt_last", "persona evil, last prompt token"),
    ("persona_sycophantic@prompt_last", "persona sycophantic, last prompt token"),
    ("diff_kl_first", "model diff: KL at first reply token"),
    ("diff_kl_reply", "model diff: KL over fixed replies"),
]


def load():
    sig = json.loads((RESULTS / "signals.json").read_text())
    lead = json.loads((RESULTS / "lead_times.json").read_text())
    return sig["steps"], sig["table"], lead


def behaviour(steps, table):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), sharex=True)
    print("behaviour.png: step, proxy runs mean, control runs mean")
    for ax, (key, label) in zip(
        axes, [("behaviour_hack", "hack rate"), ("behaviour_true", "true accuracy")], strict=False
    ):
        row = table[key]
        for run, values in row["e"].items():
            colour = EMBER if run.startswith("proxy") else GREY
            ax.plot(steps, np.array(values) + row["base"][run], color=colour, lw=1.4, alpha=0.9)
        ax.set_title(label, fontsize=10, loc="left")
        ax.set_xlabel("training step")
        ax.set_ylim(0, 1)
        p = np.mean([np.array(row["e"][f"proxy-s{s}"]) + row["base"][f"proxy-s{s}"] for s in SEEDS], 0)
        c = np.mean([np.array(row["e"][f"true-s{s}"]) + row["base"][f"true-s{s}"] for s in SEEDS], 0)
        for i in range(0, len(steps), 5):
            print(f"  {label:14s} step {steps[i]:3d}  proxy {p[i]:.3f}  control {c[i]:.3f}")
    axes[0].plot([], [], color=EMBER, label="buggy grader (3 seeds)")
    axes[0].plot([], [], color=GREY, label="true grader (3 seeds)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "behaviour.png")
    plt.close(fig)


def zscores(steps, table, lead):
    layer = lead["probe"]["layer"]
    fig, axes = plt.subplots(3, 3, figsize=(9.6, 7.4), sharex=True)
    print("zscores.png: monitor, detection step per seed")
    for ax, (sig, label) in zip(axes.flat, MONITORS, strict=False):
        site = sig.split("@")[1] if "@" in sig else None
        if site:
            rand = [r for r in table if r.startswith(f"random_L{layer}_") and r.endswith(f"@{site}")]
            Z = np.abs(np.array([[table[r]["z"][f"s{s}"] for s in SEEDS] for r in rand]))  # [20, 3, steps]
            hi = np.percentile(Z.reshape(-1, len(steps)), 95, axis=0)
            ax.fill_between(steps, -hi, hi, color=BLUE, alpha=0.15, lw=0)
        for s in SEEDS:
            ax.plot(steps, table[sig]["z"][f"s{s}"], color=EMBER if not sig.startswith("behaviour") else INK, lw=1.2)
            d = table[sig]["detect"][f"s{s}"]
            if d is not None:
                ax.axvline(d, color=EMBER, lw=0.6, ls=":")
        ax.axhline(3, color=GREY, lw=0.8, ls="--")
        ax.axhline(-3, color=GREY, lw=0.8, ls="--")
        ax.set_title(label, fontsize=9, loc="left")
        ax.set_ylim(-12, 20)
        print(f"  {sig:34s} {[table[sig]['detect'][f's{s}'] for s in SEEDS]}")
    for ax in axes[-1]:
        ax.set_xlabel("training step")
    for ax in axes[:, 0]:
        ax.set_ylabel("z vs control runs")
    fig.tight_layout()
    fig.savefig(FIGURES / "zscores.png")
    plt.close(fig)


def posthoc_leads(lead):
    """Post hoc: the first checkpoint where at least 5 of the diff's 10 most-raised first tokens contain '<'."""
    ph = json.loads((RESULTS / "posthoc.json").read_text())
    steps = json.loads((RESULTS / "signals.json").read_text())["steps"]
    behaviour = lead["summary"]["behaviour_hack"]["detect"]
    out = []
    for s in SEEDS:
        counts = ph["tag_tokens_in_top10"][f"proxy-s{s}"]
        first = next((steps[i] for i, c in enumerate(counts) if i > 0 and c >= 5), None)
        out.append(behaviour[f"s{s}"] - first if first is not None else None)
    controls = [max(ph["tag_tokens_in_top10"][f"true-s{s}"]) for s in SEEDS]
    return out, controls


def leads(lead):
    summ = lead["summary"]
    rows = [("behaviour_gap", "proxy minus true reward (behaviour)")] + [m for m in MONITORS[1:] if m[0] in summ]
    rows.append(("posthoc_tags", "model diff: '<' tokens in top 10 raised (post hoc)"))
    ph, ctrl_max = posthoc_leads(lead)
    print(f"  post hoc tag-token rule: leads {ph}; most '<' tokens in a control top-10: {ctrl_max}")
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    print("leads.png: monitor, lead per seed (steps; + = before the hack rate), mean")
    for y, (sig, label) in enumerate(rows):
        vals = ph if sig == "posthoc_tags" else [v for v in summ[sig]["lead"] if v is not None]
        if sig == "posthoc_tags":
            ax.scatter(vals, [y] * len(vals), facecolors="none", edgecolors=EMBER, s=26, zorder=3)
            ax.scatter([np.mean(vals)], [y], color=EMBER, marker="|", s=200, zorder=3)
            print(f"  {label:40s} {vals} mean {np.mean(vals):.1f}")
            continue
        colour = GREY if sig.startswith("behaviour") else EMBER
        ax.scatter(vals, [y] * len(vals), color=colour, s=22, zorder=3)
        ax.scatter([np.mean(vals)], [y], color=colour, marker="|", s=200, zorder=3)
        if "random_mean_leads" in summ[sig]:
            rl = summ[sig]["random_mean_leads"]
            ax.scatter(rl, [y - 0.25] * len(rl), color=BLUE, s=8, alpha=0.6)
        print(f"  {label:40s} {summ[sig]['lead']} mean {np.mean(vals):.1f}")
    ax.axvline(0, color=GREY, lw=0.8)
    ax.set_yticks(range(len(rows)), [label for _, label in rows], fontsize=8)
    ax.invert_yaxis()
    ax.axvspan(-38, -30, color=LIGHT, alpha=0.35, lw=0)
    ax.text(-34, len(rows) - 0.4, "never fired", ha="center", fontsize=7, color=GREY)
    ax.set_xlim(-38, 20)
    ax.set_xlabel("lead over the hack rate (training steps; right = earlier warning)")
    fig.tight_layout()
    fig.savefig(FIGURES / "leads.png")
    plt.close(fig)


def persona(lead):
    summ = lead["summary"]
    items = [
        (f"persona_{t}@{site}", f"{t}, {site.replace('_', ' ')}")
        for t in ("evil", "sycophantic", "sandbagging")
        for site in ("prompt_last", "fixed_mean", "own_mean")
    ]
    items.append(("probe@prompt_last", "hack probe, prompt last"))
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    print("persona.png: direction, final excess z (proxy mean - control mean) / sigma, random range")
    for y, (sig, label) in enumerate(items):
        e = summ[sig]
        z = e["final_excess"] / e["sigma"]
        rz = e["random_final_excess_z"]
        ax.scatter(rz, [y] * len(rz), color=BLUE, s=8, alpha=0.6)
        ax.scatter([z], [y], color=EMBER, s=30, zorder=3)
        print(f"  {label:34s} {z:+7.2f}  random [{min(rz):+.2f}, {max(rz):+.2f}]")
    ax.axvline(0, color=GREY, lw=0.8)
    ax.set_xscale("symlog", linthresh=3)
    ax.set_yticks(range(len(items)), [label for _, label in items], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("final excess over the control runs (sigma units)")
    fig.tight_layout()
    fig.savefig(FIGURES / "persona.png")
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    steps, table, lead = load()
    behaviour(steps, table)
    zscores(steps, table, lead)
    leads(lead)
    persona(lead)


if __name__ == "__main__":
    main()
