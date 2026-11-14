"""Draw the note's figures from results/ and print every number plotted.

    python experiments/context-directions/make_figures.py

stability.png  stability S by layer (cross-context agreement over the split-half ceiling), one panel per concept,
               one line per context pair; band: 2.5-97.5 percentiles over 50 splits; dotted: the 0.9 and 0.7 bars
controls.png   stability against plain at layer 14: two neutral control contexts beside the jailbreak and the persona,
               and the reply-based Chinese direction (exploratory)
transfer.png   steering effect of each source context's direction in each target context, layer 14, eval items,
               95% bootstrap intervals; blue ticks: six random directions at the same norm (truth on elements and on
               unused capitals, refusal, and the reply-based Chinese direction)
Also writes the per-layer S values into the note's slider. With CTX_TAG set (another model), only the geometry.
dose.png       the strength sweep on dev items (plain context): own direction vs the six random directions
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE / "results"
OUT = HERE / "report" / "figures"
TAG = os.environ.get("CTX_TAG", "")
LAYER = int(os.environ.get("CTX_LAYER", "14"))
EMBER, BLUE, GREY, INK, SOFT = "#C9561C", "#4A86E8", "#9A948C", "#1d272a", "#586468"
PAIR_STYLE = {"plain|roleplay": (EMBER, "-"), "plain|jailbreak": (INK, "-"), "roleplay|jailbreak": (GREY, "--")}
NAMES = {"truth": "Truth", "refusal": "Refusal", "chinese": "Chinese"}
METRIC = {"truth": "change in logit(True) − logit(False)", "refusal": "change in refusal rate",
          "chinese": "change in Chinese share"}  # fmt: skip
plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": SOFT, "savefig.dpi": 200}
)


def stability(g):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    print("stability.png: S at layer 14 and mean over layers 8-20")
    for ax, (concept, row) in zip(axes, g["concepts"].items(), strict=True):
        layers = np.arange(g["n_layers"])
        for key, v in row["pairs"].items():
            colour, ls = PAIR_STYLE[key]
            s = np.array(v["S"], dtype=float)
            ax.plot(layers, s, color=colour, ls=ls, lw=1.6, label=key.replace("|", " vs "))
            ax.fill_between(layers, v["S_low"], v["S_high"], color=colour, alpha=0.12, lw=0)
            print(f"  {concept:8s} {key:20s} L14 {s[14]:.3f}  mean L8-20 {np.nanmean(s[8:21]):.3f}")
        ax.axhline(0.9, color=SOFT, ls=":", lw=1)
        ax.axhline(0.7, color=SOFT, ls=":", lw=1)
        ax.axvline(14, color=SOFT, lw=0.6)
        ax.set_ylim(0, 1.05)
        ax.set_title(NAMES[concept], loc="left", fontsize=10)
        ax.set_xlabel("layer")
    axes[0].set_ylabel("stability S (1 = as alike as two halves\nof one context)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / f"stability{TAG}.png")
    plt.close(fig)


def transfer_panels() -> list:
    """(title, metric label, strength, targets dict) for each panel that has results."""
    panels = []
    st = json.loads((R / f"steer{TAG}.json").read_text())
    tc = R / f"steer_truth_capitals{TAG}.json"
    cr = R / f"chinese_reply{TAG}.json"
    t = st["concepts"]["truth"]
    panels.append(("Truth, element statements", METRIC["truth"], t["strength"], t["targets"]))
    if tc.exists():
        t2 = json.loads(tc.read_text())["concepts"]["truth"]
        panels.append(("Truth, unused capitals (exploratory)", METRIC["truth"], t2["strength"], t2["targets"]))
    r = st["concepts"]["refusal"]
    panels.append(("Refusal", METRIC["refusal"], r["strength"], r["targets"]))
    if cr.exists():
        c = json.loads(cr.read_text())
        panels.append(
            ("Chinese, read from replies (exploratory)", METRIC["chinese"], c["dev"]["chosen"], c["transfer"])
        )
    return panels


def transfer(_st=None):
    panels = transfer_panels()
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6))
    print("transfer.png: effect E(source -> target) [95% CI], ratio T; random max")
    for ax, (title, metric, strength, targets) in zip(axes.flat, panels, strict=False):
        names = list(targets)
        sources = list(targets[names[0]]["effects"])
        width = 0.24
        for j, y in enumerate(names):
            t = targets[y]
            for i, x in enumerate(sources):
                e = t["effects"][x]
                pos = j + (i - 1) * width
                colour = EMBER if x == y else {0: INK, 1: GREY, 2: "#c9a68f"}[i]
                ax.bar(pos, e["E"], width * 0.9, color=colour if x == y else "none", edgecolor=colour, lw=1.4)
                ax.errorbar(pos, e["E"], yerr=[[e["E"] - e["ci"][0]], [e["ci"][1] - e["E"]]], color=INK, lw=0.9,
                            capsize=2)  # fmt: skip
                print(f"  {title[:22]:22s} {x:9s} -> {y:9s} E {e['E']:+.3f} [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}]"
                      f"  T {e['T']:.2f}")  # fmt: skip
            for rr in t["random"]:
                ax.plot([j - 0.4, j + 0.4], [rr, rr], color=BLUE, lw=0.8, alpha=0.7)
            print(f"  {title[:22]:22s} random    -> {y:9s} max {max(t['random']):+.3f}")
        ax.axhline(0, color=SOFT, lw=0.6)
        ax.set_xticks(range(len(names)), [f"in {y}" for y in names])
        ax.set_title(f"{title} (×{strength:g})", loc="left", fontsize=10)
        ax.set_ylabel(metric, fontsize=9)
    for ax in list(axes.flat)[len(panels) :]:
        ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, fc="none", ec=c) for c in (INK, GREY, "#c9a68f")]
    handles += [plt.Rectangle((0, 0), 1, 1, fc=EMBER), plt.Line2D([], [], color=BLUE)]
    fig.legend(handles, ["from plain", "from roleplay", "from jailbreak", "own context (filled)", "random"],
               frameon=False, fontsize=8, ncol=5, loc="lower center")  # fmt: skip
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / f"transfer{TAG}.png")
    plt.close(fig)


def controls():
    gc = json.loads((R / f"geometry_controls{TAG}.json").read_text())
    cr = R / f"chinese_reply{TAG}.json"
    keys = [
        ("plain|neutral_system", "neutral system prompt", GREY),
        ("plain|filler_prefix", "neutral prefix", "#c7c1b8"),
        ("plain|jailbreak", "jailbreak", INK),
        ("plain|roleplay", "roleplay persona", EMBER),
    ]
    groups = [(NAMES[c], gc[c]) for c in ("truth", "refusal", "chinese")]
    if cr.exists():
        groups.append(("Chinese, from replies", json.loads(cr.read_text())["stability"]))
    fig, ax = plt.subplots(figsize=(10, 3.5))
    print(f"controls.png: S against plain at layer {LAYER}")
    w = 0.2
    for g, (name, d) in enumerate(groups):
        for i, (k, label, colour) in enumerate(keys):
            v = d[k][LAYER]
            ax.bar(g + (i - 1.5) * w, v, w * 0.9, color=colour, label=label if g == 0 else None)
            ax.text(g + (i - 1.5) * w, v + 0.01, f"{v:.2f}", ha="center", fontsize=7.5, color=INK)
            print(f"  {name:22s} {label:22s} {v:.3f}")
    ax.axhline(0.9, color=SOFT, ls=":", lw=1)
    ax.axhline(0.7, color=SOFT, ls=":", lw=1)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("stability S against plain")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    fig.tight_layout()
    fig.savefig(OUT / f"controls{TAG}.png")
    plt.close(fig)


def sim_data(g):
    """Write the per-layer S values into the note's slider (between the sim markers)."""
    import re

    data = {c: {k: [None if np.isnan(x) else round(x, 4) for x in v["S"]] for k, v in row["pairs"].items()}
            for c, row in g["concepts"].items()}  # fmt: skip
    page = HERE / "report" / "index.html"
    text = page.read_text()
    new = re.sub(r"/\*sim:start\*/.*?/\*sim:end\*/", "/*sim:start*/" + json.dumps(data, separators=(",", ":"))
                 + "/*sim:end*/", text, flags=re.DOTALL)  # fmt: skip
    if new != text:
        page.write_text(new)
        print("wrote the slider data into report/index.html")


def dose(dev):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
    print("dose.png: dev sweep, plain context")
    for ax, (concept, row) in zip(axes, dev["concepts"].items(), strict=True):
        s = [r["strength"] for r in row["sweep"]]
        ax.plot(s, [r["own"] for r in row["sweep"]], "o-", color=EMBER, label="own direction")
        for k in range(len(row["sweep"][0]["random"])):
            ax.plot(s, [r["random"][k] for r in row["sweep"]], "-", color=BLUE, alpha=0.4, lw=0.8)
        ax.axvline(row["chosen"], color=SOFT, ls=":", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("strength (× the class gap)")
        ax.set_title(f"{NAMES[concept]}: chosen ×{row['chosen']:g}", loc="left", fontsize=10)
        ax.set_ylabel(METRIC[concept], fontsize=9)
        for r in row["sweep"]:
            print(f"  {concept:8s} x{r['strength']:<4g} own {r['own']:+.3f} random max {max(r['random']):+.3f}")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"dose{TAG}.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    g = json.loads((R / f"geometry{TAG}.json").read_text())
    stability(g)
    if (R / f"geometry_controls{TAG}.json").exists():
        controls()
    if TAG:
        return
    sim_data(g)
    if (R / "steer_dev.json").exists():
        dose(json.loads((R / "steer_dev.json").read_text()))
    if (R / "steer.json").exists():
        transfer()


if __name__ == "__main__":
    main()
