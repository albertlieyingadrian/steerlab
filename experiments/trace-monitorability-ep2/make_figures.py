"""Draw the note's figures from results/, and print every number plotted.

    .venv/bin/python experiments/trace-monitorability-ep2/make_figures.py

1. views.png      M1: reader accuracy (type + target) for each view, headline steps and leak-free steps, 95% intervals
2. leak.png       M1 by what step k's thinking gives away (the exact command / the target's name / neither / empty)
3. risky.png      risky actions: M1 per view on every risky test step; M3 never-mentioned and M2 not-matching by type
4. ../report/strips.js   data for the note's rollout explorer (types, M3 labels, M4 flags, task check)
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "report" / "figures"
EMBER, GREY, GREY_DARK, GREY_LIGHT, INK = "#C9561C", "#9A948C", "#5F5A54", "#CFC8BF", "#2B2723"
CONDS = ("T0", "A", "Rp", "Rp+Vk", "Rp+Tk", "All")
LABEL = {
    "T0": "task\nonly",
    "A": "actions\nso far",
    "Rp": "earlier\nthinking",
    "Rp+Vk": "earlier +\nk message",
    "Rp+Tk": "earlier +\nk thinking",
    "All": "all but\nthe call",
}
COLOR = {"T0": GREY_LIGHT, "A": GREY, "Rp": GREY_DARK, "Rp+Vk": GREY_DARK, "Rp+Tk": EMBER, "All": GREY}

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "savefig.dpi": 150,
    }
)
S = json.loads((RESULTS / "scores.json").read_text())


def rows(name: str) -> list[dict]:
    with open(RESULTS / name) as f:
        return list(csv.DictReader(f))


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote report/figures/{name}")


def bar(ax, x, ci, color, w=0.7):
    p, lo, hi = ci[:3]
    ax.bar(x, p, w, color=color)
    ax.errorbar(x, p, yerr=[[p - lo], [hi - p]], color=INK, capsize=3, lw=1)
    ax.text(x, hi + 0.02, f"{p:.2f}", ha="center", va="bottom", fontsize=8, color=INK)


def views() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), sharey=True)
    for ax, key, title in (
        (axes[0], "headline", "All headline test steps"),
        (axes[1], "leak_free_thinking", "Thinking names neither command nor target"),
    ):
        block = S["M1"][key]
        n = block["A"]["both"][3]
        print(f"{title} (n={n}):", {c: block[c]["both"][:3] for c in CONDS})
        for i, c in enumerate(CONDS):
            bar(ax, i, block[c]["both"], COLOR[c])
        ax.set_xticks(range(len(CONDS)), [LABEL[c] for c in CONDS], fontsize=8)
        ax.set_title(f"{title} (n = {n})", fontsize=10, loc="left")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("next command: type and target right")
    save(fig, "views.png")


def leak() -> None:
    groups = [
        ("leak_command", "thinking contains\nthe exact command"),
        ("leak_target_only", "thinking names\nthe target only"),
        ("leak_free_with_thinking", "thinking names\nneither"),
        ("empty_thinking", "no thinking\nat all"),
    ]
    conds = ("A", "Rp", "Rp+Tk", "All")
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    w = 0.2
    for gi, (key, _) in enumerate(groups):
        block = S["M1"][key]
        print(f"{key} (n={block['A']['both'][3]}):", {c: block[c]["both"][:3] for c in conds})
        for ci_, c in enumerate(conds):
            x = gi + (ci_ - 1.5) * w
            p, lo, hi = block[c]["both"][:3]
            ax.bar(x, p, w * 0.92, color=COLOR[c], label=LABEL[c].replace("\n", " ") if gi == 0 else None)
            ax.errorbar(x, p, yerr=[[p - lo], [hi - p]], color=INK, capsize=2, lw=0.8)
    ax.set_xticks(range(len(groups)), [f"{g[1]}\n(n = {S['M1'][g[0]]['A']['both'][3]})" for g in groups], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("type and target right")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper right")
    save(fig, "leak.png")


def risky() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), gridspec_kw={"width_ratios": [5, 6]})
    block = S["M1"]["risky_all"]
    n = block["A"]["both"][3]
    print(f"risky test steps (n={n}):", {c: block[c]["both"][:3] for c in CONDS})
    for i, c in enumerate(CONDS):
        bar(axes[0], i, block[c]["both"], COLOR[c])
    axes[0].set_xticks(range(len(CONDS)), [LABEL[c] for c in CONDS], fontsize=8)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("type and target right")
    axes[0].set_title(f"Predicting risky steps (n = {n})", fontsize=10, loc="left")
    unm, div = rows("unmentioned.csv"), rows("divergence.csv")
    types = ["read_inspect", "local_write", "execute_install", "delete", "permission_change", "network_change"]
    m3 = {t: [r for r in unm if r["action_type"] == t] for t in types}
    m2 = {t: [r for r in div if r["action_type"] == t] for t in types}
    w = 0.38
    for i, t in enumerate(types):
        a = sum(r["where"] == "never" for r in m3[t]) / max(len(m3[t]), 1)
        b = sum(r["label"] != "matches" for r in m2[t]) / max(len(m2[t]), 1)
        print(f"  {t}: M3 never {a:.3f} (n={len(m3[t])}), M2 not matches {b:.3f} (n={len(m2[t])})")
        axes[1].bar(i - w / 2, a, w, color=GREY_DARK, label="never mentioned (M3)" if i == 0 else None)
        axes[1].bar(i + w / 2, b, w, color=GREY_LIGHT, label="step thinking does not match (M2)" if i == 0 else None)
        axes[1].text(i - w / 2, a + 0.01, f"{len(m3[t])}", ha="center", fontsize=7, color=INK)
        axes[1].text(i + w / 2, b + 0.01, f"{len(m2[t])}", ha="center", fontsize=7, color=INK)
    axes[1].set_xticks(range(len(types)), [t.replace("_", "\n") for t in types], fontsize=8)
    axes[1].set_ylabel("share of steps")
    axes[1].set_title("By action type (numbers = steps; last three are risky)", fontsize=10, loc="left")
    axes[1].legend(frameon=False, fontsize=8)
    save(fig, "risky.png")


def strips() -> None:
    code = {
        "read_inspect": "r",
        "local_write": "w",
        "execute_install": "x",
        "delete": "d",
        "permission_change": "p",
        "network_change": "n",
    }
    steps, unm, rule = rows("steps.csv"), rows("unmentioned.csv"), rows("target_rule.csv")
    roll = {r["rollout"]: r for r in rows("rollouts.csv")}
    never = {r["step_id"] for r in unm if r["where"] == "never"}
    flag = {r["step_id"] for r in rule if r["flagged"] == "1"}
    by = defaultdict(list)
    for s in steps:
        by[s["rollout"]].append(s)
    leak_code = {"command": "c", "target_only": "t", "none": "-"}
    out = []
    for rid, ss in by.items():
        ss.sort(key=lambda s: int(s["k"]))
        out.append(
            {
                "id": rid,
                "task": roll[rid]["task"],
                "pass": int(roll[rid]["check_pass"]),
                "end": roll[rid]["end"],
                "s": "".join(code.get(s["action_type"], "u") for s in ss),
                "never": [int(s["k"]) for s in ss if s["step_id"] in never],
                "flag": [int(s["k"]) for s in ss if s["step_id"] in flag],
                "leak": "".join(leak_code[s["leak_think"]] if s["think_chars"] != "0" else "e" for s in ss),
            }
        )
    out.sort(key=lambda r: (r["task"], r["id"]))
    (HERE / "report" / "strips.js").write_text(
        "// Generated by make_figures.py from results/steps.csv, rollouts.csv, unmentioned.csv and target_rule.csv.\n"
        f"window.STRIPS = {json.dumps(out, separators=(',', ':'))};\n"
    )
    print(f"  wrote report/strips.js ({len(out)} rollouts)")


if __name__ == "__main__":
    views()
    leak()
    risky()
    strips()
