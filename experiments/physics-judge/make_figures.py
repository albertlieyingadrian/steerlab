"""Draw the note's figures from results/, printing every number plotted.

    python experiments/physics-judge/make_figures.py

auroc.png   test AUROC for "breaks physical commonsense" (406 videos, 270 prompts) per judge, with 95% intervals over
            prompts; ember for the cheap judge chosen on dev, dark grey for the released judges, light for the rest.
            Blue: the text-only control (amendment 1); blue-grey: the annotators' own rule marks.
laws.png    rule-level test AUROC per law group for Claude's `rules` config (Opus) and the WorldModelBench judge.
traps.png   for 25 clean clips and three doctored copies of each, the share where each judge scores the doctored copy
            higher than the clean one (ties half), with Wilson intervals; the dashed line is the pass mark (20/25).
cost.png    test AUROC against dollars per video (log scale) for every judge with a cost.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import RESULTS

FIGURES = HERE / "report" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
EMBER, GREY, LIGHT, INK = "#C9561C", "#6F6A63", "#C8C2B8", "#2B2723"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "savefig.dpi": 150})  # fmt: skip
MODEL = {"opus": "Opus 5.5", "sonnet": "Sonnet 5.5", "haiku": "Haiku 5.5"}
CONF = {
    "plain": "plain",
    "checklist": "checklist",
    "motion": "16 motion frames",
    "rules": "video's rules",
    "rules_noframes": "rules, no frames (control)",
    "plain_noframes": "plain, no frames (control)",
}
BLUE = "#3D6FA3"  # controls


def name(j: str) -> str:
    fixed = {
        "wmb": "WorldModelBench judge (released)",
        "wmb_official": "WorldModelBench judge, official yes/no",
        "wmb_mean": "WorldModelBench judge, mean of 5",
        "autoeval": "VideoPhy-2 AutoEval (released, 7B)",
        "h_combined": "motion heuristics, combined",
        "h_warp_error": "heuristic: flow-warp error",
        "h_motion": "heuristic: mean motion",
        "h_divergence": "heuristic: flow divergence",
        "h_jerk": "heuristic: motion jerk",
        "h_diff_spike": "heuristic: frame-difference spike",
        "h_scale_jump": "heuristic: scale jump",
        "ens_checklist": "checklist, 3 models' ranks averaged",
        "chosen_plus_heur": "chosen judge + heuristics",
        "annot_rules": "annotators' own rule marks (reference)",
    }
    if j in fixed:
        return fixed[j]
    if j.startswith("selfcons3_"):
        return f"{MODEL[j.split('_')[1]]}, checklist x3 averaged"
    c, m = j.rsplit("_", 1)
    return f"{MODEL[m]}, {CONF[c]}"


def auroc_fig(met: pd.DataFrame, chosen: str) -> None:
    t = met[(met.split == "test") & (met.judge != "wmb_mean")].copy()
    t = t[~t.judge.str.startswith("h_") | t.judge.isin(["h_combined", "h_warp_error"])]
    t = t.sort_values("auroc")
    fig, ax = plt.subplots(figsize=(7.5, 0.28 * len(t) + 1.0))
    for k, r in enumerate(t.itertuples()):
        col = EMBER if r.judge == chosen else (GREY if r.group in ("released", "released-variant") else LIGHT)
        if r.group == "reference":
            col = "#9AA7B5"
        if r.group == "control":
            col = BLUE
        ax.errorbar(r.auroc, k, xerr=[[r.auroc - r.lo], [r.hi - r.auroc]], fmt="o", color=col, ms=5, capsize=2)
        print(f"auroc  {r.judge:22s} {r.auroc:.3f} [{r.lo:.3f}, {r.hi:.3f}]")
    ax.set_yticks(np.arange(len(t)), [name(j) for j in t.judge])
    ax.axvline(0.5, color=INK, lw=0.6, ls=":")
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("test AUROC (406 videos; axis starts at 0.4)")
    fig.tight_layout()
    fig.savefig(FIGURES / "auroc.png")
    plt.close(fig)


def laws_fig(rm: pd.DataFrame) -> None:
    t = rm[(rm.split == "test") & rm.judge.isin(["rules_opus", "wmb"])]
    laws = ["Gravity", "Friction", "Elasticity", "Conservation of Momentum", "Conservation of Mass", "Hardness",
            "Other", "All"]  # fmt: skip
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for off, j, col, lab in [(-0.15, "wmb", GREY, "WorldModelBench judge (mapped question)"),
                             (0.15, "rules_opus", EMBER, "Opus 5.5, video's rules")]:  # fmt: skip
        d = t[t.judge == j].set_index("law").reindex(laws)
        x = np.arange(len(laws)) + off
        ax.errorbar(x, d.auroc, yerr=[d.auroc - d.lo, d.hi - d.auroc], fmt="o", color=col, capsize=2, label=lab)
        for law, r in d.iterrows():
            print(f"laws   {j:10s} {law:26s} n={r.n:.0f} viol={r.violated:.0f} {r.auroc:.3f} [{r.lo:.3f}, {r.hi:.3f}]")
    counts = t[t.judge == "wmb"].set_index("law").reindex(laws).violated
    labels = [
        f"{law.replace('Conservation of ', 'Cons. ')}\n({c:.0f} viol.)" for law, c in zip(laws, counts, strict=True)
    ]
    ax.set_xticks(range(len(laws)), labels, fontsize=8)
    ax.axhline(0.5, color=INK, lw=0.6, ls=":")
    ax.set_ylim(0.2, 1.0)
    ax.set_ylabel("rule-level test AUROC")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES / "laws.png")
    plt.close(fig)


def traps_fig(tr: pd.DataFrame, chosen: str) -> None:
    judges = ["wmb", chosen, "checklist_opus", "plain_haiku", "h_warp_error", "h_scale_jump"]
    judges = list(dict.fromkeys(j for j in judges if j in set(tr.judge)))
    kinds = ["reversed", "frozen", "resized"]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    w = 0.8 / len(judges)
    for k, j in enumerate(judges):
        d = tr[tr.judge == j].set_index("kind").reindex(kinds)
        x = np.arange(3) + (k - (len(judges) - 1) / 2) * w
        col = EMBER if j == chosen else (GREY if j == "wmb" else LIGHT)
        hatch = {"plain_haiku": "..", "h_warp_error": "//", "h_scale_jump": "xx"}.get(j)
        ax.bar(
            x, d.share, w * 0.9, color=col, label=name(j), edgecolor=INK if col == LIGHT else col, lw=0.4, hatch=hatch
        )
        ax.errorbar(x, d.share, yerr=[d.share - d.lo, d.hi - d.share], fmt="none", color=INK, lw=0.7, capsize=1.5)
        for kind, r in d.iterrows():
            print(f"traps  {j:16s} {kind:9s} {r.doctored_higher:.1f}/{r.n:.0f} = {r.share:.2f}")
    ax.axhline(0.8, color=INK, lw=0.8, ls="--")
    ax.axhline(0.5, color=INK, lw=0.6, ls=":")
    ax.set_xticks(range(3), ["reversed in time", "frozen from the middle", "centre enlarged from the middle"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("doctored copy scored higher\n(share of 25 pairs)")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(FIGURES / "traps.png")
    plt.close(fig)


def cost_fig(met: pd.DataFrame, chosen: str) -> None:
    t = met[(met.split == "test") & met.usd_per_video.notna() & (met.usd_per_video > 0)]
    t = t[~t.judge.isin(["wmb_mean", "wmb_official", "chosen_plus_heur"]) & (t.group != "control")]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for r in t.itertuples():
        col = EMBER if r.judge == chosen else (GREY if r.group == "released" else LIGHT)
        ax.scatter(r.usd_per_video, r.auroc, color=col, edgecolor=INK, lw=0.4, s=36, zorder=3)
        ax.annotate(name(r.judge).replace(" (released)", ""), (r.usd_per_video, r.auroc), fontsize=6.5,
                    xytext=(4, 2), textcoords="offset points")  # fmt: skip
        print(f"cost   {r.judge:22s} ${r.usd_per_video:.5f} {r.auroc:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("dollars per video (Claude: batch prices; WorldModelBench: A10G time)")
    ax.set_ylabel("test AUROC")
    fig.tight_layout()
    fig.savefig(FIGURES / "cost.png")
    plt.close(fig)


def main() -> None:
    met = pd.read_csv(RESULTS / "metrics_video.csv")
    chosen = json.loads((RESULTS / "selection.json").read_text())["chosen"]
    auroc_fig(met, chosen)
    laws_fig(pd.read_csv(RESULTS / "metrics_rules.csv"))
    traps_fig(pd.read_csv(RESULTS / "traps_result.csv"), chosen)
    cost_fig(met, chosen)


if __name__ == "__main__":
    main()
