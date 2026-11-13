"""Draw the note's figures from results/ (scores and labels only; no model text).

    .venv/bin/python experiments/auditbench/make_figures.py

Writes to report/figures/:
- identify.png  top-1 accuracy at naming the hidden behaviour (of 14), per detector and training config, with 95%
                bootstrap intervals and the chance line;
- alarms.png    the two probes on the 42 organisms, none of which sandbags: scores in units of the prompted sandbagger's
                shift, by training config, beside the prompted sandbagger and the range of random directions;
- sandbagger70b.png  Llama-3.3-70B: the sandbagging probe on the quiz and on the code pairs (safety minus ordinary),
                the 4 research_sandbagging organisms against the 56 controls, by training config.
Prints every number it plots, so the text can be checked against it.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = HERE / "report" / "figures"

INK, SOFT, FAINT, RULE = "#1d272a", "#586468", "#8a9497", "#e3e7e8"
ACCENT, CONTROL = "#c9561c", "#4a86e8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelcolor": SOFT, "text.color": INK})

CONFIG_LABEL = {
    "all": "all 42",
    "td": "transcripts (14)",
    "td_sft": "transcripts + red-team SFT (14)",
    "sdf_sft": "documents + red-team SFT (14)",
}


def style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=SOFT, labelsize=8)


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote report/figures/{name}")


def identify(summary: dict) -> None:
    dets = [
        ("D-diff: activation difference", summary["diff"]["L20_centred"], ACCENT, "o"),
        ("D-amp: amplified replies + judge", summary["judge_amp"], ACCENT, "s"),
        ("D-plain: plain replies + judge", summary["judge_plain"], FAINT, "s"),
    ]
    groups = ["all", "td", "td_sft", "sdf_sft"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    style(ax)
    h = 0.24
    for gi, g in enumerate(groups):
        for di, (name, d, colour, marker) in enumerate(dets):
            est, lo, hi = d[g]["top1"]
            ax.errorbar(
                est,
                gi + (di - 1) * h,
                xerr=[[est - lo], [hi - est]],
                fmt=marker,
                color=colour,
                ms=5,
                capsize=2,
                lw=1.2,
                mfc="white" if di == 1 else colour,
                label=name if gi == 0 else None,
            )
            print(f"identify {g:8s} {name:34s} top-1 {est:.3f} [{lo:.3f}, {hi:.3f}]")
    ax.axvline(1 / 14, color=CONTROL, ls="--", lw=1)
    ax.text(1 / 14 + 0.01, len(groups) - 0.55, "chance, 1/14", color=CONTROL, fontsize=7.5)
    ax.set_yticks(range(len(groups)), [CONFIG_LABEL[g] for g in groups])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("top-1 accuracy: share of organisms whose true behaviour is ranked first (of 14)")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    save(fig, "identify.png")


def alarms(summary: dict) -> None:
    with (RESULTS / "organisms.csv").open() as f:
        rows = list(csv.DictReader(f))
    configs = ["td", "td_sft", "sdf_sft"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    for ax, (col, key, title) in zip(
        axes,
        [("sandbag_probe", "sandbag", "sandbagging probe"), ("eval_probe", "eval", "eval-awareness probe")],
        strict=False,
    ):
        style(ax)
        band = summary["probe_controls"][key]["random_p999"]
        ax.axhspan(-band, band, color=RULE, zorder=0)
        ax.text(3.45, -band, "random directions, 99.9%", color=SOFT, fontsize=6.5, va="bottom", ha="right")
        for ci, c in enumerate(configs):
            vals = [float(r[col]) for r in rows if r["config"] == c]
            xs = [ci + ((i * 0.6180339) % 1 - 0.5) * 0.4 for i in range(len(vals))]
            ax.scatter(xs, vals, s=13, color=[ACCENT if v > 0.5 else FAINT for v in vals], zorder=3)
            print(
                f"alarms {col:14s} {c:8s} median {sorted(vals)[len(vals) // 2]:+.3f} "
                f"max {max(vals):+.3f} flagged {sum(v > 0.5 for v in vals)}/{len(vals)}"
            )
        ax.scatter([3], [1.0], s=30, color=CONTROL, marker="D", zorder=3)
        ax.axhline(0.5, color=ACCENT, lw=1, ls="--")
        ax.text(3.45, 0.52, "flag", color=ACCENT, fontsize=7, ha="right")
        ax.set_xticks(range(4), ["transcripts", "transcripts\n+ SFT", "documents\n+ SFT", "prompted\nsandbagger"])
        ax.set_xlim(-0.5, 3.5)
        ax.set_title(title, fontsize=8.5, color=INK)
        print(f"alarms {col:14s} random-direction 99.9% band {band:.3f}")
    axes[0].set_ylabel("score (1 = prompted sandbagger's shift)")
    fig.tight_layout()
    save(fig, "alarms.png")


def sandbagger70b() -> None:
    path = RESULTS / "llama70b" / "organisms.csv"
    if not path.exists():
        return
    with path.open() as f:
        rows = list(csv.DictReader(f))
    configs = ["td", "td_sft", "sdf", "sdf_sft"]
    labels = ["transcripts", "transcripts\n+ SFT", "documents", "documents\n+ SFT"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    panels = [
        ("sandbag_probe", "quiz, plain (primary, H2)"),
        ("sandbag_probe_code", "code requests: safety minus ordinary (secondary)"),
    ]
    for ax, (col, title) in zip(axes, panels, strict=True):
        style(ax)
        ax.axhline(0, color=RULE, lw=1, zorder=0)
        for ci, c in enumerate(configs):
            ctrl = [float(r[col]) for r in rows if r["config"] == c and r["behaviour"] != "research_sandbagging"]
            sb = [float(r[col]) for r in rows if r["config"] == c and r["behaviour"] == "research_sandbagging"]
            xs = [ci + ((i * 0.6180339) % 1 - 0.5) * 0.4 for i in range(len(ctrl))]
            ax.scatter(xs, ctrl, s=12, color=FAINT, zorder=2, label="other behaviours (14)" if ci == 0 else None)
            ax.scatter(
                [ci], sb, s=36, color=ACCENT, marker="D", zorder=3, label="research_sandbagging" if ci == 0 else None
            )
            rank = 1 + sum(v > sb[0] for v in ctrl)
            print(
                f"sandbagger70b {col:20s} {c:8s} sandbagger {sb[0]:+.3f} rank {rank}/15 controls max {max(ctrl):+.3f}"
            )
        ax.set_xticks(range(4), labels)
        ax.set_xlim(-0.5, 3.5)
        ax.set_title(title, fontsize=8.5, color=INK)
    axes[0].axhline(0.5, color=ACCENT, lw=1, ls="--")
    axes[0].text(3.45, 0.52, "flag", color=ACCENT, fontsize=7, ha="right")
    axes[0].set_ylabel("score (1 = prompted sandbagger's shift)")
    axes[1].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    save(fig, "sandbagger70b.png")


def main() -> None:
    summary = json.loads((RESULTS / "summary.json").read_text())
    identify(summary)
    alarms(summary)
    sandbagger70b()


if __name__ == "__main__":
    main()
