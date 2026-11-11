"""Draw the note's figures from results/ and print every number they plot.

    python experiments/eval-aware-steer/make_figures.py [model]

Writes to report/figures/:
- dose.png      accuracy on the odd test questions against steering strength, at the headline layer: (a) eval framing
                pushed toward "real use", (b) real-use framing pushed toward "test"; awareness, policy-built,
                unrelated and five random directions (band), and in (b) the same push with no policy
- report.png    the verbal check: p(Yes) to "is this a test?" under the same pushes, awareness against random
- layers.png    held-out separation (AUROC) by layer, and awareness's accuracy change at its chosen strength by layer
                against the random range
- fluency.png   median perplexity of free-text replies under steering, by strength and direction
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

FIG = C.HERE / "report" / "figures"
INK, SOFT, FAINT, RULE = "#1d272a", "#586468", "#8a9497", "#e3e7e8"
ACCENT, ACCENT2, CONTROL, CONTROL_L = "#c9561c", "#e0a07c", "#4a86e8", "#c9daf8"
ORDER = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelcolor": SOFT, "text.color": INK})


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.tick_params(colors=SOFT, length=0)
    ax.grid(axis="y", color=RULE, lw=0.6, zorder=0)


def load(name):
    rows = list(csv.DictReader((C.RESULTS / name / "summary.csv").read_text().splitlines()))
    for r in rows:
        for k, v in r.items():
            if k not in ("cond", "source") and v != "":
                r[k] = float(v)
    return rows, json.loads((C.RESULTS / name / "verdict.json").read_text())


def series(rows, cond, src, layer, key, base):
    """[(value, lo, hi)] over ORDER; strength 0 is the unsteered cell. Intervals are base + paired-delta interval."""
    out = {0.0: (base, base, base)}
    for r in rows:
        if r["cond"] == cond and r["source"] == src and r["layer"] == layer:
            val = r["acc_odd"] if key == "acc" else r["p_yes_odd"]
            out[r["alpha"]] = (val, base + r["lo"], base + r["hi"])
    return np.array([out[x] for x in ORDER])


def xaxis(ax):
    ax.set_xticks(range(len(ORDER)), ["0" if x == 0 else f"{x:g}" for x in ORDER])


def dose(rows, v):
    L = v["headline_layer"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), dpi=200, sharey=True)
    panels = [
        ("eval_down", "(a) eval framing, pushed toward real use"),
        ("real_up", "(b) real-use framing, pushed toward test"),
    ]
    px = range(len(ORDER))
    print(f"Figure 1 (layer {L}): accuracy on odd questions by strength {ORDER}")
    for ax, (cond, title) in zip(axes, panels, strict=True):
        base = v["baseline_odd"][cond]
        rand = np.array([series(rows, cond, f"rand{k}", L, "acc", base)[:, 0] for k in range(5)])
        ax.fill_between(px, rand.min(0), rand.max(0), color=CONTROL_L, lw=0, zorder=1, label="5 random directions")
        un = series(rows, cond, "unrelated", L, "acc", base)
        ax.plot(px, un[:, 0], "--", color=CONTROL, lw=1.2, zorder=2, label="unrelated (sea vs mountain)")
        po = series(rows, cond, "policy", L, "acc", base)
        ax.plot(px, po[:, 0], ":", color=ACCENT2, lw=1.8, zorder=3, label="policy-built (upper bound)")
        aw = series(rows, cond, "awareness", L, "acc", base)
        ax.fill_between(px, aw[:, 1], aw[:, 2], color=ACCENT, alpha=0.15, lw=0, zorder=3)
        ax.plot(px, aw[:, 0], "o-", color=ACCENT, lw=1.8, ms=3.5, zorder=4, label="eval-awareness direction")
        if cond == "real_up":
            nu = series(rows, "neutral_up", "awareness", L, "acc", v["baseline_odd"]["neutral_up"])
            ax.plot(px, nu[:, 0], "s-", color=FAINT, lw=1.0, ms=2.5, zorder=2, label="same push, no policy")
            print("  no policy, awareness:", np.round(nu[:, 0], 3).tolist())
        ax.axhline(0.25, color=FAINT, lw=0.8, ls=(0, (2, 2)))
        ax.text(-0.1, 0.265, "chance", ha="left", fontsize=7, color=FAINT)
        xaxis(ax)
        ax.set_xlabel("strength (class gaps; all directions the same length)")
        ax.set_title(title, fontsize=8.5, loc="left", color=INK)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25%", "50%", "75%", "100%"])
        style(ax)
        print(
            f"  {cond}: awareness {np.round(aw[:, 0], 3).tolist()}\n    policy {np.round(po[:, 0], 3).tolist()}\n"
            f"    unrelated {np.round(un[:, 0], 3).tolist()}\n    random min {np.round(rand.min(0), 3).tolist()}"
            f"\n    random max {np.round(rand.max(0), 3).tolist()}"
        )
    axes[0].set_ylabel("accuracy (held-out half)")
    h, lab = axes[1].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, fontsize=7.2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIG / "dose.png", bbox_inches="tight")
    plt.close(fig)


def report(rows, v):
    L = v["headline_layer"]
    r0 = v["report"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8), dpi=200, sharey=True)
    px = range(len(ORDER))
    print(f"Figure 2 (layer {L}): p(Yes, this is a test) by strength")
    for ax, (cond, base, title) in zip(
        axes,
        [
            ("report_eval_down", r0["unsteered_p_yes_eval_odd"], "(a) eval framing, pushed toward real use"),
            ("report_real_up", r0["unsteered_p_yes_real_odd"], "(b) real-use framing, pushed toward test"),
        ],
        strict=True,
    ):
        rand = np.array([series(rows, cond, f"rand{k}", L, "py", base)[:, 0] for k in range(5)])
        aw = series(rows, cond, "awareness", L, "py", base)
        un = series(rows, cond, "unrelated", L, "py", base)
        ax.fill_between(px, rand.min(0), rand.max(0), color=CONTROL_L, lw=0, label="5 random directions")
        ax.plot(px, un[:, 0], "--", color=CONTROL, lw=1.2, label="unrelated")
        ax.fill_between(px, aw[:, 1], aw[:, 2], color=ACCENT, alpha=0.15, lw=0)
        ax.plot(px, aw[:, 0], "o-", color=ACCENT, lw=1.8, ms=3.5, label="eval-awareness direction")
        xaxis(ax)
        ax.set_xlabel("strength (class gaps)")
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.set_ylim(0, 1)
        style(ax)
        print(
            f"  {cond}: awareness {np.round(aw[:, 0], 3).tolist()}\n    unrelated {np.round(un[:, 0], 3).tolist()}"
            f"\n    random {np.round(rand.min(0), 3).tolist()} .. {np.round(rand.max(0), 3).tolist()}"
        )
    axes[0].set_ylabel('p("Yes, a test")')
    h, lab = axes[1].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, fontsize=7.2, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIG / "report.png", bbox_inches="tight")
    plt.close(fig)


def layers(name, rows, v):
    d = json.loads((C.RESULTS / name / "direction.json").read_text())
    Ls = d["layers"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), dpi=200)
    ax = axes[0]
    print("Figure 3: held-out AUROC by layer; awareness change at its chosen strength by layer")
    for key, col, lab, off in [
        ("heldout_auroc_neutral", ACCENT, "awareness, neutral prompt", -0.12),
        ("heldout_auroc_policy", ACCENT2, "awareness, policy prompt", 0.0),
        ("heldout_auroc_unrelated_dir", CONTROL, "unrelated direction", 0.12),
    ]:
        a = np.array([d["per_layer"][str(L)][key] for L in Ls])
        ax.errorbar(
            np.arange(len(Ls)) + off,
            a[:, 0],
            yerr=[a[:, 0] - a[:, 1], a[:, 2] - a[:, 0]],
            fmt="o",
            color=col,
            ms=4,
            capsize=2,
            lw=0.9,
            label=lab,
        )
        print(f"  {lab}: " + ", ".join(f"L{L} {x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]" for L, x in zip(Ls, a, strict=True)))
    ax.axhline(0.5, color=FAINT, lw=0.8, ls=(0, (2, 2)))
    ax.set_xticks(range(len(Ls)), [f"layer {L}" for L in Ls])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("held-out AUROC, eval vs real use")
    ax.set_title("(a) the direction separates the framings", fontsize=8.5, loc="left")
    ax.legend(frameon=False, fontsize=6.8, loc="lower left")
    style(ax)
    ax = axes[1]
    for cond, col, lab, off in [
        ("eval_down", ACCENT, "eval framing, pushed toward real use", -0.1),
        ("real_up", ACCENT2, "real use, pushed toward test", 0.1),
    ]:
        for i, L in enumerate(Ls):
            a = v["selection"][str(L)][cond]["alpha"]
            get = {r["source"]: r for r in rows if r["cond"] == cond and r["layer"] == L and r["alpha"] == a}
            aw = get["awareness"]
            rd = [get[f"rand{j}"]["delta_odd"] for j in range(5)]
            ax.vlines(i + off, min(rd), max(rd), color=CONTROL_L, lw=5, zorder=1)
            ax.errorbar(
                i + off,
                aw["delta_odd"],
                yerr=[[aw["delta_odd"] - aw["lo"]], [aw["hi"] - aw["delta_odd"]]],
                fmt="o",
                color=col,
                ms=4,
                capsize=2,
                lw=0.9,
                zorder=3,
                label=lab if i == 0 else None,
            )
            print(
                f"  {cond} L{L} alpha {a}: {aw['delta_odd']:+.3f} [{aw['lo']:+.3f}, {aw['hi']:+.3f}] "
                f"random {min(rd):+.3f}..{max(rd):+.3f}"
            )
    ax.axhline(0, color=FAINT, lw=0.8)
    ax.set_xticks(range(len(Ls)), [f"layer {L}" for L in Ls])
    ax.set_ylabel("accuracy change (held-out half)")
    ax.set_title("(b) at each layer's chosen strength", fontsize=8.5, loc="left")
    ax.legend(frameon=False, fontsize=6.8, loc="upper right")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "layers.png", bbox_inches="tight")
    plt.close(fig)


def fluency(name):
    p = C.RESULTS / name / "fluency.csv"
    if not p.exists():
        return
    rows = list(csv.DictReader(p.read_text().splitlines()))
    fig, ax = plt.subplots(figsize=(4.8, 2.8), dpi=200)
    base = float(np.median([float(r["ppl"]) for r in rows if r["arm"] == "none"]))
    print(f"Figure 4: median perplexity (unsteered {base:.2f}) and fluent share, by strength {ORDER[1:]}")
    for arm, col, ls, lab in [
        ("awareness+", ACCENT, "-", "awareness, toward test"),
        ("awareness-", ACCENT2, "-", "awareness, toward real use"),
        ("unrelated", CONTROL, "--", "unrelated"),
        ("rand0", FAINT, "--", "random"),
    ]:
        med = [
            np.median([float(r["ppl"]) for r in rows if r["arm"] == arm and float(r["alpha"]) == a]) for a in ORDER[1:]
        ]
        flu = [
            np.mean([r["fluent"] == "True" for r in rows if r["arm"] == arm and float(r["alpha"]) == a])
            for a in ORDER[1:]
        ]
        ax.plot(range(len(ORDER)), [base, *med], ls, color=col, marker="o", ms=3, lw=1.4, label=lab)
        print(f"  {arm}: ppl {np.round(med, 2).tolist()} fluent {np.round(flu, 2).tolist()}")
    xaxis(ax)
    ax.set_xlabel("strength (class gaps)")
    ax.set_ylabel("median perplexity of replies\n(scored by the unsteered model)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=7)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fluency.png", bbox_inches="tight")
    plt.close(fig)


def explorer(rows, v):
    """report/explorer.js: the dose curves at the headline layer, for the page's strength slider."""
    L = v["headline_layer"]
    data = {"layer": L, "alphas": ORDER, "n": v["n_odd"], "conds": {}}
    for cond in ("eval_down", "real_up"):
        base = v["baseline_odd"][cond]
        rand = np.array([series(rows, cond, f"rand{k}", L, "acc", base)[:, 0] for k in range(5)])
        c = {
            s: series(rows, cond, s, L, "acc", base)[:, 0].round(3).tolist()
            for s in ("awareness", "policy", "unrelated")
        }
        c["random_min"], c["random_max"] = rand.min(0).round(3).tolist(), rand.max(0).round(3).tolist()
        if cond == "real_up":
            c["no_policy"] = (
                series(rows, "neutral_up", "awareness", L, "acc", v["baseline_odd"]["neutral_up"])[:, 0]
                .round(3)
                .tolist()
            )
        data["conds"][cond] = c
    (C.HERE / "report" / "explorer.js").write_text("window.EAS_DATA = " + json.dumps(data) + ";\n")
    print("explorer.js written (layer", L, ")")


def confirm_figure():
    """confirm.png: the pre-registered confirmatory runs (fresh questions and framings, 20 random directions), one row
    per model that has results/<model>/confirm.json; accuracy change against strength, x2 marked."""
    models = [m for m in ("llama3.1-8b", "gemma2-9b") if (C.RESULTS / m / "confirm.json").exists()]
    if not models:
        return
    fig, axes = plt.subplots(len(models), 2, figsize=(7.4, 2.7 * len(models)), dpi=200, squeeze=False)
    al = [0.5, 1.0, 2.0, 4.0]
    for row, m in enumerate(models):
        cf = json.loads((C.RESULTS / m / "confirm.json").read_text())
        cells = list(csv.DictReader((C.RESULTS / m / "confirm_cells.csv").read_text().splitlines()))
        print(f"Figure 5 ({m}, layer {cf['layer']}, n={cf['n_questions']}): change by strength {al}")
        for col, (push, title) in enumerate(
            [
                ("toward_real", "test framing, pushed toward real use"),
                ("toward_test", "real-use framing, pushed toward test"),
            ]
        ):
            ax = axes[row][col]
            px = range(len(al))

            def series(src, push=push, cells=cells):
                return [
                    float(
                        next(
                            r["delta"]
                            for r in cells
                            if r["cond"] == push and r["source"] == src and float(r["alpha"]) == a
                        )
                    )
                    for a in al
                ]

            rand = np.array([series(f"rand{k}") for k in range(20)])
            ax.fill_between(px, rand.min(0), rand.max(0), color=CONTROL_L, lw=0, label="20 random directions")
            for u, ls in (("unrelated", "--"), ("unrelated2", ":")):
                ax.plot(px, series(u), ls, color=CONTROL, lw=1.1, label=f"{u} direction")
            ax.plot(px, series("policy"), ":", color=ACCENT2, lw=1.6, label="policy-built (upper bound)")
            aw = series("awareness")
            ax.plot(px, aw, "o-", color=ACCENT, lw=1.8, ms=3.5, label="eval-awareness direction")
            ax.axvline(al.index(cf["alpha_star"]), color=FAINT, lw=0.8, ls=(0, (2, 2)))
            ax.axhline(0, color=FAINT, lw=0.8)
            ax.set_xticks(range(len(al)), [f"{a:g}" for a in al])
            ax.set_title(f"{cf['model'].split('/')[-1]}: {title}", fontsize=8, loc="left")
            ax.set_ylim(-0.7, 0.8)
            style(ax)
            print(
                f"  {push}: awareness {np.round(aw, 3).tolist()} random {np.round(rand.min(0), 3).tolist()}.."
                f"{np.round(rand.max(0), 3).tolist()}"
            )
        axes[row][0].set_ylabel("accuracy change")
    for ax in axes[-1]:
        ax.set_xlabel("strength (class gaps); dashed: x2, fixed in advance")
    h, lab = axes[0][1].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / "confirm.png", bbox_inches="tight")
    plt.close(fig)


def main(name="llama3.1-8b"):
    FIG.mkdir(parents=True, exist_ok=True)
    rows, v = load(name)
    explorer(rows, v)
    dose(rows, v)
    report(rows, v)
    layers(name, rows, v)
    fluency(name)
    confirm_figure()


if __name__ == "__main__":
    main(*sys.argv[1:2])
