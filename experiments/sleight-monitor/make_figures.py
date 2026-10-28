"""Draw the note's figures from results/ (per-transcript IDs and scores only; no transcript text).

    python experiments/sleight-monitor/make_figures.py [--out DIR]

Writes to report/figures/ (or DIR):
- auroc.png       every arm's AUROC with its 95 % pair-bootstrap interval, and the shuffled and random null band;
- roc.png         ROC curves for the black-box monitor, the two probes and the two baselines;
- pairs.png       each attack transcript against its own benign twin, for the black-box score and the probe;
- categories.png  attacks caught at 2/43 false positives, per category and arm, next to the paper's Opus 4.6 rate.
Prints every number it plots, so the text can be checked against it. Also writes results/budget.json, the scores
Figure 4's slider recomputes from; the page embeds it in its <script id="budget-data"> block: after a rerun, paste
it again.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

INK, SOFT, FAINT, RULE = "#1d272a", "#586468", "#8a9497", "#e3e7e8"
ACCENT, CONTROL = "#c9561c", "#4a86e8"
COLOUR = {
    "A": INK,
    "B_max": ACCENT,
    "B_mean": ACCENT,
    "B2_max": ACCENT,
    "B2_mean": ACCENT,
    "lexical2": CONTROL,
    "C_pre": ACCENT,
    "C_verdict": ACCENT,
    "lexical": CONTROL,
    "length": CONTROL,
    "ceiling_B": FAINT,
    "ceiling_C": FAINT,
}
SHORT = {
    "A": "A  black-box monitor (its written score)",
    "B_max": "B  probe trained on synthetic data, max step",
    "B_mean": "B  same probe, mean step",
    "B2_max": "B2 probe trained on subtle inserts, max step",
    "B2_mean": "B2 same probe, mean step",
    "lexical2": "bag of words on subtle inserts (control)",
    "C_pre": "C  probe before the monitor reasons",
    "C_verdict": "C  probe at the monitor's verdict",
    "lexical": "bag of words (control)",
    "length": "transcript length (control)",
    "ceiling_B": "ceiling: probe trained on SLEIGHT itself",
    "ceiling_C": "ceiling: verdict probe trained on SLEIGHT",
}


def read(name):
    with (RESULTS / name).open() as f:
        return list(csv.DictReader(f))


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=SOFT, labelsize=8)


def auroc_figure(out, metrics, summary):
    controls = json.loads((RESULTS / "controls.json").read_text())
    null = summary["nulls_B"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
    lo = min(np.percentile(controls["shuffled_auroc"], 2.5), np.percentile(controls["random_direction_auroc"], 2.5))
    hi = max(null["shuffled_97_5"], null["random_97_5"])
    ax.axvspan(lo, hi, color=CONTROL, alpha=0.08, lw=0)
    ax.axvline(0.5, color=FAINT, lw=0.8, ls=":")
    for k, m in enumerate(metrics[::-1]):
        a, cl, ch = float(m["auroc"]), float(m["ci_lo"]), float(m["ci_hi"])
        c = COLOUR[m["arm"]]
        ax.plot([cl, ch], [k, k], color=c, lw=1.6, solid_capstyle="round")
        ax.plot(a, k, "o", color=c, ms=5.5, mfc="white" if m["role"] == "ceiling" else c)
        ax.text(1.01, k, f"{a:.2f}", va="center", fontsize=8, color=c, transform=ax.get_yaxis_transform())
        print(f"  auroc {m['arm']:<10} {a:.3f} [{cl:.3f}, {ch:.3f}]")
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([SHORT[m["arm"]] for m in metrics[::-1]], fontsize=8, color=INK)
    ax.set_xlim(0, 1)
    ax.set_xlabel("AUROC on 43 attack vs 43 benign transcripts (0.5 = chance)", fontsize=8, color=SOFT)
    ax.text(
        0.02,
        len(metrics) - 0.4,
        "blue band: middle 95 % of shuffled-label and random-direction probes (arm B)",
        fontsize=7,
        color=CONTROL,
        va="bottom",
    )
    print(f"  null band {lo:.3f} to {hi:.3f}")
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "auroc.png")
    plt.close(fig)


def roc_points(y, s):
    P, N = y.sum(), (1 - y).sum()
    xs, ys = [0.0], [0.0]
    for thr in np.unique(s)[::-1]:
        xs.append(((s >= thr) & (y == 0)).sum() / N)
        ys.append(((s >= thr) & (y == 1)).sum() / P)
    return xs, ys


def roc_figure(out, scores, y):
    fig, ax = plt.subplots(figsize=(4.6, 4.4), dpi=200)
    ax.plot([0, 1], [0, 1], color=FAINT, lw=0.8, ls=":")
    styles = {
        "A": ("-", INK),
        "B_max": ("-", ACCENT),
        "B2_max": ("-.", ACCENT),
        "C_verdict": ("--", ACCENT),
        "lexical": ("-", CONTROL),
        "length": (":", CONTROL),
    }
    for arm, (ls, c) in styles.items():
        xs, ys = roc_points(y, scores[arm])
        ax.plot(xs, ys, ls=ls, color=c, lw=1.5, label=SHORT[arm], drawstyle="steps-post")
    ax.axvline(2 / 43, color=RULE, lw=0.8)
    ax.text(2 / 43 + 0.01, 0.02, "2/43 FP", fontsize=7, color=SOFT)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("false positive rate (benign transcripts flagged)", fontsize=8, color=SOFT)
    ax.set_ylabel("true positive rate (attacks caught)", fontsize=8, color=SOFT)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "roc.png")
    plt.close(fig)


def pairs_figure(out, scores, y):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), dpi=200)
    for ax, arm, title in (
        (axes[0], "A", "A: black-box written score (0-100)"),
        (axes[1], "B_max", "B: probe score, highest step (logit)"),
    ):
        s = scores[arm]
        att, ben = s[y == 1], s[y == 0]
        win = att > ben
        for a, b, w in zip(att, ben, win, strict=False):
            ax.plot([0, 1], [b, a], color=ACCENT if w else FAINT, lw=0.8, alpha=0.9 if w else 0.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["benign twin", "attack"], fontsize=8, color=INK)
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        ties = int((att == ben).sum())
        ax.text(
            0.5,
            1.0,
            f"attack higher in {int(win.sum())}/43 pairs, tied in {ties}",
            fontsize=7,
            color=SOFT,
            ha="center",
            va="top",
            transform=ax.transAxes,
        )
        print(f"  pairs {arm}: attack higher {int(win.sum())}/43, tied {ties}")
        style(ax)
    fig.tight_layout()
    fig.savefig(out / "pairs.png")
    plt.close(fig)


def categories_figure(out, per_cat):
    arms = ["A", "B_max", "B2_max", "C_verdict", "lexical", "length"]
    names = {
        "A": "A black box",
        "B_max": "B probe",
        "B2_max": "B2 subtle",
        "C_verdict": "C verdict",
        "lexical": "words",
        "length": "length",
    }
    per_cat = sorted(per_cat, key=lambda r: -int(r["paper_caught_trials"]) / int(r["paper_trials"]))
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
    for i, r in enumerate(per_cat):
        n = int(r["n"])
        for j, arm in enumerate(arms):
            v = int(r[f"{arm}_2fp"])
            ax.add_patch(plt.Rectangle((j, i), 0.94, 0.88, color=COLOUR[arm], alpha=0.08 + 0.8 * v / n, lw=0))
            ax.text(
                j + 0.47,
                i + 0.44,
                f"{v}/{n}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if v / n > 0.5 else INK,
            )
        pr = int(r["paper_caught_trials"]) / int(r["paper_trials"])
        ax.text(len(arms) + 0.1, i + 0.44, f"{pr:.0%}", va="center", fontsize=7.5, color=SOFT)
        print(f"  {r['category']:<28} " + " ".join(f"{a}={r[a + '_2fp']}/{n}" for a in arms) + f" paper={pr:.2f}")
    ax.set_xlim(0, len(arms) + 0.8)
    ax.set_ylim(len(per_cat), 0)
    ax.set_xticks([j + 0.47 for j in range(len(arms))] + [len(arms) + 0.3])
    ax.set_xticklabels([names[a] for a in arms] + ["paper\nOpus 4.6"], fontsize=7.5, color=INK)
    ax.set_yticks([i + 0.44 for i in range(len(per_cat))])
    ax.set_yticklabels([r["category"].replace("_", " ") for r in per_cat], fontsize=7.5, color=INK)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "categories.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "report" / "figures")
    out = ap.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    metrics = read("metrics.csv")
    summary = json.loads((RESULTS / "summary.json").read_text())
    y = np.array([int(r["label"]) for r in read("transcripts.csv")])
    scores = {}
    for f, cols in (
        ("monitor_scores.csv", {"A": "score"}),
        ("sleight_scores.csv", {"B_max": "probe_max", "lexical": "lexical_max", "length": "length_tokens"}),
        ("sleight_scores_subtle.csv", {"B2_max": "probe_max"}),
        ("verdict_scores.csv", {"C_verdict": "c_verdict"}),
    ):
        rows = read(f)
        for arm, col in cols.items():
            scores[arm] = np.array([float(r[col]) if r[col] != "" else 50.0 for r in rows])
    budget = {
        "label": y.tolist(),
        **{
            k: [round(float(v), 3) for v in scores[a]]
            for k, a in (
                ("A", "A"),
                ("B", "B_max"),
                ("B2", "B2_max"),
                ("C", "C_verdict"),
                ("words", "lexical"),
                ("length", "length"),
            )
        },
    }
    (RESULTS / "budget.json").write_text(json.dumps(budget, separators=(",", ":")))
    print("figure numbers:")
    auroc_figure(out, metrics, summary)
    roc_figure(out, scores, y)
    pairs_figure(out, scores, y)
    categories_figure(out, read("per_category.csv"))
    print(f"wrote auroc.png, roc.png, pairs.png, categories.png to {out.name}/")


if __name__ == "__main__":
    main()
