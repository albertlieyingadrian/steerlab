"""
Every figure in the note, drawn from results/ (never by hand), plus report/data.js, the data behind the interactive
charts. Prints the numbers it plots so they can be checked against the text.

    .venv/bin/python experiments/pluralism-judge-rl/make_figures.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HERE, RESULTS  # noqa: E402

FIG = HERE / "report" / "figures"
ARMS = ["single", "jury", "bon", "shuffled", "clean"]
LABEL = {
    "single": "GRPO, single judge",
    "jury": "GRPO, jury",
    "bon": "Best-of-8 SFT",
    "shuffled": "GRPO, shuffled judge",
    "clean": "Clean-twin SFT",
}
COLOR = {"single": "#d26f2a", "jury": "#7a4fa0", "bon": "#2f8f6b", "shuffled": "#4a86e8", "clean": "#9a948c"}
MEASURE_LABEL = {
    "rep_creative": "Repetition, creative prompts (mean pairwise cosine)",
    "rep_advice": "Repetition, advice prompts (mean pairwise cosine)",
    "rep50_creative": "Repetition, creative, first 50 words",
    "rep50_advice": "Repetition, advice, first 50 words",
    "coverage": "Coverage (distinct valid answers / possible)",
    "steer": "Steerability (1 - JS distance to the shown split)",
    "survey_entropy": "Survey answer entropy, default (nats)",
    "survey_entropy_steered": "Survey answer entropy, 'I live in X' (nats)",
    "survey_jsd_default": "Survey JS distance to 8 countries, default",
    "survey_jsd_steered": "Survey JS distance to own country, 'I live in X'",
    "srep_creative": "Repetition, creative prompts (shared module)",
    "srep_advice": "Repetition, advice prompts (shared module)",
    "srep50_creative": "Repetition, creative, first 50 words (shared module)",
    "srep50_advice": "Repetition, advice, first 50 words (shared module)",
    "scoverage": "Coverage (shared module)",
    "house_creative": "House style, creative: similarity between replies to different prompts",
    "house_advice": "House style, advice: similarity between replies to different prompts",
    "ep1_entropy": "Survey answer spread, no country named (Episode 1's eval, 0-1)",
    "ep1_entropy_steered": "Survey answer spread, 'I live in X' (Episode 1's eval)",
    "ep1_jsd_default": "Survey distance to 8 countries, no country named",
    "ep1_jsd_steered": "Survey distance to own country, 'I live in X'",
    "words_creative": "Reply length, creative (words)",
    "words_advice": "Reply length, advice (words)",
}
PRIMARY = {"creative": "srep_creative", "advice": "ep1_entropy"}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#7a7065",
        "axes.labelcolor": "#1d1a17",
        "xtick.color": "#5a5048",
        "ytick.color": "#5a5048",
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff",
    }
)


def load_rows():
    with (RESULTS / "checkpoints.csv").open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k not in {"run", "arm", "task", "survey_ok"}:
                r[k] = float(v)
        r["survey_ok"] = r["survey_ok"] == "True"
    return rows


def curves(rows, task, measure, x):
    """arm -> [(run, xs, ys)], each starting from the base model."""
    base = next(r for r in rows if r["run"] == "base-s0")
    out = defaultdict(list)
    for run in sorted({r["run"] for r in rows if r["arm"] in ARMS and r["task"] in (task, "none")}):
        pts = sorted([r for r in rows if r["run"] == run], key=lambda r: r["step"])
        out[pts[0]["arm"]].append((run, [0.0] + [r[x] for r in pts], [base[measure]] + [r[measure] for r in pts]))
    return out


def fig_primary(rows, summary):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, task in zip(axes, ["creative", "advice"]):
        m = PRIMARY[task]
        for arm in ARMS:
            for i, (_, xs, ys) in enumerate(curves(rows, task, m, "drift_kl").get(arm, [])):
                ax.plot(xs, ys, "-o", ms=3, lw=1.6, color=COLOR[arm], alpha=0.9, label=LABEL[arm] if i == 0 else None)
        if task in summary["matched"]:
            ax.axvline(summary["matched"][task]["D"], color="#7a7065", lw=1, ls=":")
        ax.set_xlabel("Drift: KL(base || trained) per token, neutral prompts")
        ax.set_ylabel(MEASURE_LABEL[m].split(" (")[0])
        ax.set_title(f"{task.capitalize()} task", fontsize=10, loc="left")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "primary_vs_drift.png", dpi=170)
    plt.close(fig)


def fig_house(rows):
    """Exploratory: house style (similarity between replies to different prompts) against drift, full replies and first
    50 words, on creative and advice prompts, every run."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, task in zip(axes, ["creative", "advice"]):
        for m, ls in [(f"house_{task}", "-"), (f"house50_{task}", ":")]:
            for arm in ARMS:
                for i, (_, xs, ys) in enumerate(curves(rows, task, m, "drift_kl").get(arm, [])):
                    ax.plot(xs, ys, ls, marker="o", ms=2.5, lw=1.6, color=COLOR[arm],
                            label=LABEL[arm] if (i == 0 and ls == "-") else None)
        ax.set_ylim(0, 0.6)
        ax.set_xlabel("Drift: KL(base || trained) per token, neutral prompts")
        ax.set_title(f"Runs trained on {task} prompts; replies to held-out {task} prompts", fontsize=9.5, loc="left")
    axes[0].set_ylabel("Similarity between replies\nto different prompts")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "house_vs_drift.png", dpi=170)
    plt.close(fig)


def fig_matched(summary):
    measures = ["srep_creative", "srep_advice", "scoverage", "steer", "ep1_entropy", "ep1_jsd_steered"]
    fig, axes = plt.subplots(2, len(measures), figsize=(14, 5.4))
    for row, task in enumerate(["creative", "advice"]):
        arms = summary["matched"].get(task, {}).get("arms", {})
        for col, m in enumerate(measures):
            ax = axes[row][col]
            for i, arm in enumerate(ARMS):
                v = arms.get(arm, {}).get(m)
                if not v:
                    continue
                for j, s in enumerate(v["seeds"]):
                    x = i + (j - (len(v["seeds"]) - 1) / 2) * 0.22
                    ax.errorbar(x, s["delta"], yerr=[[s["delta"] - s["lo"]], [s["hi"] - s["delta"]]], fmt="o", color=COLOR[arm], ms=4, lw=1.2)
            ax.axhline(0, color="#7a7065", lw=0.8)
            ax.set_xticks(range(len(ARMS)))
            ax.set_xticklabels(ARMS, rotation=45, fontsize=7)
            if row == 0:
                ax.set_title(MEASURE_LABEL[m].split(" (")[0], fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{task}: change vs base")
    fig.suptitle("Change from base at matched drift (repetition, coverage, steerability, entropy signed so positive = less plural)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "matched.png", dpi=170)
    plt.close(fig)


def fig_drift(rows):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, task in zip(axes, ["creative", "advice"]):
        for arm in ARMS:
            for i, (_, xs, ys) in enumerate(curves(rows, task, "drift_kl", "step").get(arm, [])):
                ax.plot(xs, ys, "-o", ms=3, lw=1.5, color=COLOR[arm], label=LABEL[arm] if i == 0 else None)
        ax.set_xlabel("Optimizer step")
        ax.set_title(f"{task.capitalize()} task", fontsize=10, loc="left")
    axes[0].set_ylabel("Drift (nats per token)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "drift_vs_step.png", dpi=170)
    plt.close(fig)


def fig_jury(extras):
    fav = extras.get("jury_favourites", {})
    if not fav:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    palette = ["#d26f2a", "#7a4fa0", "#2f8f6b", "#4a86e8"]
    for ax, task in zip(axes, ["creative", "advice"]):
        keys = sorted([k for k in fav if k.endswith(f"|{task}")], key=lambda k: (0 if k.startswith("base") else 1 if k.startswith("single") else 2, k))
        if not keys:
            continue
        bottom = np.zeros(len(keys))
        for j, name in enumerate(fav[keys[0]]["jurors"]):
            vals = np.array([fav[k]["share"][j] for k in keys])
            ax.barh(range(len(keys)), vals, left=bottom, color=palette[j], label=name)
            bottom += vals
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels([f"{k.split('|')[0]}  H={fav[k]['entropy']:.2f}" for k in keys], fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Share of held-out replies whose favourite juror is …")
        ax.legend(frameon=False, fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(FIG / "jury_favourites.png", dpi=170)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    summary = json.loads((RESULTS / "summary.json").read_text())
    extras_path = RESULTS / "extras.json"
    extras = json.loads(extras_path.read_text()) if extras_path.exists() else {}
    fig_primary(rows, summary)
    fig_house(rows)
    fig_matched(summary)
    fig_drift(rows)
    fig_jury(extras)
    matched = {}
    for t, v in summary["matched"].items():
        arms = {}
        for a, ms in v["arms"].items():
            arms[a] = {
                m: {
                    "seed_mean": x["seed_mean"],
                    "beyond_floor": x["beyond_floor"],
                    "seeds": [{k: s[k] for k in ("run", "delta", "lo", "hi")} for s in x["seeds"]],
                }
                for m, x in ms.items()
            }
        matched[t] = {"D": v["D"], "arm_max_drift": v["arm_max_drift"], "arms": arms}
    data = {
        "rows": rows,
        "matched": matched,
        "floor": {m: {k: v[k] for k in ("delta", "lo", "hi")} for m, v in summary.get("floor_base_rerun", {}).items()},
        "labels": {"arms": LABEL, "colors": COLOR, "measures": MEASURE_LABEL},
        "jury": extras.get("jury_favourites", {}),
        "quality": {k: {kk: v[kk] for kk in ("kind", "open", "frontier", "rho", "n")} for k, v in extras.get("quality", {}).items()},
    }
    (HERE / "report" / "data.js").write_text("window.PJR = " + json.dumps(data, separators=(",", ":")) + ";\n")
    for task, v in summary["matched"].items():
        print(f"[{task}] D* = {v['D']:.4f}")
        for arm, ms in v["arms"].items():
            line = "  ".join(f"{m}={ms[m]['seed_mean']:+.4f}" for m in ["rep_creative", "rep_advice", "coverage", "steer", "survey_entropy"] if m in ms)
            print(f"   {arm:9s} {line}")
    print(f"figures in {FIG}; data.js written")


if __name__ == "__main__":
    main()
