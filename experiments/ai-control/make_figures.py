"""Draw the note's figures from results/ and print every number plotted.

    python experiments/ai-control/make_figures.py

fallback.png   gap recovered at 10% deferral against how much the trusted model knows (1.0 = the real Qwen2.5-0.5B;
               lower = the simulation), for trusted monitoring, judge + probe, and judge + probe deferring to the
               probe's answer (exploratory), with the shuffled control; 95% intervals
tradeoff.png   safety (gap recovered) against usefulness (accuracy with the clean twin) for each monitor with
               defer-to-trusted and the real trusted model, one point per defer rate from 2% to 50% (audit alone: 0.03)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = HERE / "report" / "figures"
EMBER, GREY, INK, SOFT = "#C9561C", "#9A948C", "#1d272a", "#586468"
STYLE = {
    "judge+probe": (EMBER, "-", "o", "(b) judge + probe"),
    "probe": (EMBER, "--", "s", "probe alone"),
    "judge": (INK, "-", "o", "(a) 0.5B judge"),
    "disagreement": (SOFT, ":", "^", "trusted model disagrees (baseline)"),
    "inconsistency": (SOFT, "-.", "D", "(d) resample inconsistency"),
    "shuffled": (GREY, "-", "x", "shuffled (b) scores (control)"),
}


def load(tag):
    return json.loads((RESULTS / f"protocols_{tag}.json").read_text())


def row(data, monitor, action, x=0.10):
    return next(r for r in data["rows"] if r["monitor"] == monitor and r["action"] == action and r["defer"] == x)


def fallback():
    tags = {1.0: "locked", 0.75: "locked_knows75", 0.5: "locked_knows50", 0.25: "locked_knows25"}
    data = {k: load(t) for k, t in tags.items()}
    ks = sorted(data)
    series = [
        ("judge", "trusted", INK, "-", "o", "(a) judge, defer to trusted"),
        ("judge+probe", "trusted", EMBER, "-", "o", "(b) judge + probe, defer to trusted"),
        ("judge+probe", "probe", EMBER, "--", "s", "(b) judge + probe, defer to the probe's answer (post hoc)"),
        ("shuffled", "trusted", GREY, "-", "x", "shuffled (b) scores, defer to trusted (control)"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for monitor, action, colour, ls, marker, label in series:
        rows = [row(data[k], monitor, action) for k in ks]
        ys = [r["gap_recovered"] for r in rows]
        err = [[y - r["gap_ci"][0] for y, r in zip(ys, rows)], [r["gap_ci"][1] - y for y, r in zip(ys, rows)]]
        print(f"fallback {monitor}->{action}: " + ", ".join(f"k={k}: {y:.3f}" for k, y in zip(ks, ys)))
        ax.errorbar(ks, ys, yerr=err, color=colour, linestyle=ls, marker=marker, markersize=5, linewidth=1.5,
                    elinewidth=0.8, capsize=0, label=label)  # fmt: skip
    accs = [data[k]["baselines"]["trusted alone"]["accuracy"] for k in ks]
    print("fallback trusted accuracy: " + ", ".join(f"k={k}: {a:.3f}" for k, a in zip(ks, accs)))
    ax.set_xticks(ks, [f"{k:.0%}\n(trusted {a:.0%} right)" for k, a in zip(ks, accs)], fontsize=8)
    ax.set_xlabel("share of facts the trusted model knows (100% = the real 0.5B; lower = simulated)", fontsize=9)
    ax.set_ylabel("sandbagging gap recovered (10% deferral)", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fallback.png", dpi=170)


def tradeoff():
    data = load("locked")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for m, (colour, ls, marker, label) in STYLE.items():
        rows = [r for r in data["rows"] if r["monitor"] == m and r["action"] == "trusted" and r["defer"] > 0]
        xs = [r["usefulness"] for r in rows]
        ys = [r["gap_recovered"] for r in rows]
        err = [[y - r["gap_ci"][0] for y, r in zip(ys, rows)], [r["gap_ci"][1] - y for y, r in zip(ys, rows)]]
        print(
            f"tradeoff {m}: "
            + ", ".join(f"x={r['defer']:.2f} use={u:.3f} gap={g:.3f}" for r, u, g in zip(rows, xs, ys))
        )
        ax.errorbar(xs, ys, yerr=err, color=colour, linestyle=ls, marker=marker, markersize=4, linewidth=1.3,
                    elinewidth=0.7, capsize=0, label=label)  # fmt: skip
    trusted = data["baselines"]["trusted alone"]["accuracy"]
    ax.axvline(trusted, color=GREY, linewidth=0.8, linestyle="--")
    ax.text(trusted + 0.0007, 0.05, "trusted 0.5B alone", fontsize=7.5, color=SOFT, rotation=90, va="bottom")
    ax.set_xlabel("usefulness: accuracy shipped when the untrusted model is the honest clean twin", fontsize=9)
    ax.set_ylabel("safety: sandbagging gap recovered", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7.5, frameon=False, loc="center left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "tradeoff.png", dpi=170)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fallback()
    tradeoff()
