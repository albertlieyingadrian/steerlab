"""Draw the circuitkit note's figures and write results/summary.json, all from the package's shipped data.

    python experiments/circuitkit/make_figures.py

Reads packages/circuitkit/examples/data/ (the RL-hack graphs, the live validator runs, the label results) and prints
every number it plots.
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from circuitkit.diff import diff_graphs, load_summaries
from circuitkit.validate import Report

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "packages" / "circuitkit" / "examples" / "data"
FIG = HERE / "report" / "figures"
INK, SOFT, RULE, ACCENT, CONTROL = "#1d272a", "#586468", "#e3e7e8", "#c9561c", "#4a86e8"
GREY = "#a39a90"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": RULE,
        "axes.labelcolor": SOFT,
        "xtick.color": SOFT,
        "ytick.color": SOFT,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def diff_numbers() -> dict:
    g = load_summaries(DATA / "rl_hack_graphs.json.gz")
    rep = diff_graphs(
        g["installed/before"],
        g["installed/after"],
        control=(g["control/before"], g["control/after"]),
        replication=(g["replication/before"], g["replication/after"]),
        fvu_before=g["installed/before"][0].meta["fvu_by_layer"],
        fvu_after=g["installed/after"][0].meta["fvu_by_layer"],
    )
    s = rep.summary
    out = {
        "top": [
            {
                "unit": c.unit,
                "delta": c.delta,
                "control_delta": c.control_delta,
                "specific": c.specific,
                "replicated": c.replicated,
                "status": c.status,
            }
            for c in rep.top
        ],
        "n_specific": s.control_specific,
        "n_replicated": s.replication_overlap,
        "change_score": s.change_score,
        "change_null_mean": s.change_null_mean,
        "change_p": s.change_p,
        "split_half": s.split_half_overlap,
        "split_half_chance": s.split_half_chance,
        "n_gained": s.n_gained,
        "n_lost": s.n_lost,
        "fvu_before": float(np.mean(rep.validity.fvu_before)),
        "fvu_after": float(np.mean(rep.validity.fvu_after)),
        "validity_ok": rep.validity.ok,
    }
    return out


def validate_numbers(job: str, group_name: str) -> dict:
    res = json.loads((DATA / "live" / f"{job}.json").read_text())
    rows = []
    for raw in res["validate"]:
        rep = Report.model_validate(raw)
        grp = next(gr for gr in rep.groups if gr.name == group_name)
        mn = grp.ablation.controls["matched_norm"]
        rows.append(
            {
                "prompt": rep.prompt,
                "scale": rep.scale,
                "ablation": grp.ablation.normalized,
                "steering": grp.steering.normalized,
                "patch": grp.patch.gap_closed if grp.patch else None,
                "matched_norm_abs": [abs(x) / rep.scale for x in mn.effects],
                "p_matched": mn.p_value,
                "verdict": grp.verdict.value,
                "verdicts": {gr.name: gr.verdict.value for gr in rep.groups},
            }
        )
    return {"runs": rows, "extra": {k: v for k, v in res.items() if k not in ("validate",)}}


def label_numbers() -> dict:
    r = json.loads((DATA / "labels_gemma_results.json").read_text())

    def mean(xs: list, key: str) -> float:
        f = "auroc" if key == "detection" else "correlation"
        return st.mean(x[key][f] for x in xs if x[key] and x[key].get(f) is not None)

    out = {
        k: {"detection": mean(r[k], "detection"), "simulation": mean(r[k], "simulation"), "n": len(r[k])}
        for k in ("ours", "neuronpedia_long", "neuronpedia_short")
    }
    out["openai"] = {"model": r["openai"]["model"], "detection": mean(r["openai"]["labels"], "detection")}
    out["calibration"] = r["calibration"]
    return out


def main() -> None:
    d = diff_numbers()
    v_rl = validate_numbers("rl_hack", "graph's top 10")
    v_lock = validate_numbers("lock", "claimed 10")
    lab = label_numbers()
    summary = {"diff": d, "validate_rl_hack": v_rl, "validate_lock": v_lock, "labels": lab}
    (HERE / "results" / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(
        json.dumps(
            {
                "diff": {k: v for k, v in d.items() if k != "top"},
                "rl_hack": [(r["verdict"], round(r["ablation"], 3), r["p_matched"]) for r in v_rl["runs"]],
                "lock": [(r["verdict"], round(r["ablation"], 3), r["patch"], r["p_matched"]) for r in v_lock["runs"]],
                "labels": {k: v for k, v in lab.items() if k != "calibration"},
            },
            indent=1,
        )
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw={"width_ratios": [1.25, 1.1, 1]})
    # 1. diff: top-10 increase vs the control run
    ax = axes[0]
    y = np.arange(len(d["top"]))[::-1]
    ax.barh(y + 0.18, [t["delta"] * 1e3 for t in d["top"]], height=0.36, color=ACCENT, label="run that hacked")
    ax.barh(y - 0.18, [t["control_delta"] * 1e3 for t in d["top"]], height=0.36, color=GREY, label="run that did not")
    ax.set_yticks(y, [t["unit"] for t in d["top"]], fontsize=8)
    ax.set_xlabel("increase in influence after RL (x 1e-3)")
    ax.set_title("Diff: the ten features RL turned up", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    # 2. validate: ablation effect of the group vs matched-norm controls, per prompt
    ax = axes[1]
    runs = [(f"RL hack, slot {k + 1}", r) for k, r in enumerate(v_rl["runs"])]
    runs += [(f"lock, question {k + 1}", r) for k, r in enumerate(v_lock["runs"])]
    for i, (_name, r) in enumerate(runs):
        ax.scatter(r["matched_norm_abs"], [i] * len(r["matched_norm_abs"]), s=10, color=GREY, alpha=0.7)
        ax.scatter([r["ablation"]], [i], s=70, color=ACCENT, zorder=3)
    ax.axvline(0.2, color=CONTROL, lw=1, ls="--")
    ax.text(0.21, len(runs) - 1.45, "needed: 20% of scale", color=CONTROL, fontsize=8)
    ax.set_yticks(range(len(runs)), [n for n, _ in runs], fontsize=8)
    ax.set_xlabel("|ablation effect| / scale\norange: the graph's ten features; grey: 20 size-matched controls")
    ax.set_title("Validate: switching them off", loc="left", fontsize=11, color=INK)
    # 3. labels: detection and simulation by label source
    ax = axes[2]
    names = [
        ("circuitkit\nHaiku 5.5", "ours"),
        ("Neuronpedia\nlong", "neuronpedia_long"),
        ("Neuronpedia\nshort", "neuronpedia_short"),
    ]
    x = np.arange(len(names))
    ax.bar(x - 0.18, [lab[k]["detection"] for _, k in names], 0.36, color=[ACCENT, GREY, GREY], label="detection AUROC")
    ax.bar(
        x + 0.18,
        [lab[k]["simulation"] for _, k in names],
        0.36,
        color=[ACCENT, GREY, GREY],
        alpha=0.55,
        label="simulation r",
    )
    for xi, (_, k) in zip(x, names, strict=True):
        ax.text(xi - 0.18, lab[k]["detection"] + 0.02, f"{lab[k]['detection']:.2f}", ha="center", fontsize=8)
        ax.text(xi + 0.18, lab[k]["simulation"] + 0.02, f"{lab[k]['simulation']:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, [n for n, _ in names], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("Labels: scored the same way", loc="left", fontsize=11, color=INK)
    ax.text(0.0, 0.93, "solid: detection AUROC · faded: simulation r", transform=ax.transAxes, fontsize=8, color=SOFT)
    fig.tight_layout()
    fig.savefig(FIG / "fig1.png", dpi=180)
    fig.savefig(FIG / "hero.jpg", dpi=110, pil_kwargs={"quality": 90})
    print("wrote fig1.png, hero.jpg")


if __name__ == "__main__":
    main()
