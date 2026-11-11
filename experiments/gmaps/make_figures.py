"""Draw the note's figures from results/summary.json, printing every number plotted.

    python experiments/gmaps/make_figures.py

ladder.png   full-record accuracy on the 288 test transmissions for the final model, the GMAPS winner and the frontier
             models we ran, with 95% recording-bootstrap intervals.
readback.png what the read-backs fixed: speaker-role accuracy on controller and pilot test rows, and full-record
             accuracy, for the Prague-free model without and with read-backs and the pilot model trained on Prague data.
rolerule.png the role rule: left, its accuracy where it fires on each held-out training airport (leave-one-airport-out);
             right, full record on test without and with the rule for gpt-5.4 and untrained Qwen3-8B.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "report" / "figures"
EMBER, GREY, LIGHT, INK = "#C9561C", "#6F6A63", "#C8C2B8", "#2B2723"
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
S = json.loads((HERE / "results" / "summary.json").read_text())
BY = {s["name"]: s for s in S["systems"]}


def pct(x: float) -> float:
    return round(100 * x, 1)


def ladder() -> None:
    rows = [
        ("q14-pfr-epoch1", "Ours: Qwen3-14B + LoRA, Prague-free (final)", EMBER),
        ("GMAPS: Nemotron-3-Nano-30B + LoRA", "GMAPS winner: Nemotron-30B + LoRA", GREY),
        ("claude-fable-5-2-medium-fs20", "Claude Fable 5.2, 20-shot", GREY),
        ("claude-opus-5-5-medium-fs20", "Claude Opus 5.5, 20-shot", GREY),
        ("GMAPS: Claude Opus 5.5, 5-shot", "Claude Opus 5.5, 5-shot (GMAPS run)", GREY),
        ("q14-pfa-epoch1", "Ours, first try: no read-backs", LIGHT),
        ("gpt-5.4-medium-fs20", "gpt-5.4, 20-shot", LIGHT),
        ("qwen-fs5", "Qwen3-8B, 5-shot, untrained", LIGHT),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    print("ladder.png (full record, 95% recording CI)")
    for i, (name, label, color) in enumerate(rows):
        t = BY[name]["test"]
        y = len(rows) - 1 - i
        v, lo, hi = pct(t["full_record"]), pct(t["full_ci"][0]), pct(t["full_ci"][1])
        ax.barh(y, v, color=color, height=0.62)
        ax.plot([lo, hi], [y, y], color=INK, lw=1)
        ax.text(hi + 1, y, f"{v:.1f}", va="center", fontsize=9, color=INK)
        print(f"  {label}: {v} [{lo}, {hi}]")
    ax.set_yticks(range(len(rows)), [r[1] for r in reversed(rows)], fontsize=9)
    ax.axvline(pct(BY["GMAPS: Nemotron-3-Nano-30B + LoRA"]["test"]["full_record"]), color=GREY, ls=":", lw=1)
    ax.set_xlim(0, 85)
    ax.set_xlabel("Full-record accuracy on 288 test transmissions (%)")
    fig.tight_layout()
    fig.savefig(FIGURES / "ladder.png")
    plt.close(fig)


def readback() -> None:
    systems = [
        ("q14-pfa-epoch1", "no\nread-backs", LIGHT),
        ("q14-pfr-epoch1", "with\nread-backs", EMBER),
        ("nemo-v1-e2", "Prague\ndata", GREY),
    ]
    metrics = [
        ("role_acc_atco", "Role, controller rows (n=128)"),
        ("role_acc_pilot", "Role, pilot rows (n=148)"),
        ("full_record", "Full record (n=288)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.8), sharey=True)
    print("readback.png")
    for ax, (key, title) in zip(axes, metrics, strict=False):
        vals = [pct(BY[n]["test"][key]) for n, _, _ in systems]
        ax.bar(range(3), vals, color=[c for _, _, c in systems], width=0.65)
        for x, v in enumerate(vals):
            ax.text(x, v + 1.5, f"{v:.1f}", ha="center", fontsize=8.5, color=INK)
        ax.set_xticks(range(3), [lab for _, lab, _ in systems], fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 105)
        print(f"  {title}: {dict(zip([n for n, _, _ in systems], vals, strict=False))}")
    axes[0].set_ylabel("Accuracy on test (%)")
    fig.tight_layout()
    fig.savefig(FIGURES / "readback.png")
    plt.close(fig)


def rolerule() -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 2.9), gridspec_kw={"width_ratios": [1, 1.2]})
    lo = S["role_rule_loao"]
    print("rolerule.png")
    a1.bar(range(len(lo)), [pct(r["correct"]) for r in lo], color=GREY, width=0.6)
    for x, r in enumerate(lo):
        a1.text(x, pct(r["correct"]) + 1.5, f"{pct(r['correct']):.0f}", ha="center", fontsize=8.5)
        print(f"  LOAO {r['airport']}: fires {r['fires']}/{r['rows']}, correct {pct(r['correct'])}")
    a1.set_xticks(range(len(lo)), [f"{r['airport']}\n{r['fires']}/{r['rows']}" for r in lo], fontsize=8)
    a1.set_ylim(0, 105)
    a1.set_ylabel("Rule correct where it fires (%)")
    a1.set_title("Held-out airport (fires / rows)", fontsize=9)
    pairs = [
        ("gpt-5.4-medium-fs20", "gpt-5.4-medium-fs20-role", "gpt-5.4, 20-shot"),
        ("qwen-fs5", "qwen-fs5-role", "Qwen3-8B, 5-shot"),
    ]
    for i, (base, ruled, label) in enumerate(pairs):
        b, r = pct(BY[base]["test"]["full_record"]), pct(BY[ruled]["test"]["full_record"])
        a2.bar(i * 2.4, b, color=LIGHT, width=0.9)
        a2.bar(i * 2.4 + 1, r, color=EMBER, width=0.9)
        a2.text(i * 2.4, b + 1.5, f"{b:.1f}", ha="center", fontsize=8.5)
        a2.text(i * 2.4 + 1, r + 1.5, f"{r:.1f}", ha="center", fontsize=8.5)
        print(f"  {label}: {b} -> {r}")
    a2.set_xticks([0.5, 2.9], [p[2] for p in pairs], fontsize=8.5)
    a2.set_ylim(0, 85)
    a2.set_ylabel("Full record on test (%)")
    a2.set_title("Without (grey) and with (orange) the rule", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "rolerule.png")
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    ladder()
    readback()
    rolerule()
