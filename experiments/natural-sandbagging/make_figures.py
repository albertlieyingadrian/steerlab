"""
the note's figures, drawn from results/ only (seconds).

    .venv/bin/python experiments/natural-sandbagging/make_figures.py [--out DIR]

    accuracy.png    accuracy by framing, WMDP (ember) against MMLU (grey), both models, 95 % bootstrap intervals
    did.png         every difference in differences with its interval; the shaded band is +/- the primary MDE
    steering.png    accuracy and judged awareness against steering strength; random directions in blue
    awareness.png   the manipulation check: share of seed-0 traces the judge reads as "I am being tested"
    pilot.png       the sizing pilot: share of traces that finish thinking within each budget (why 512 was chosen)
    results/explorer.json   per-item accuracy per framing, both models, for the note's interactive explorer


Every number plotted is printed, so the text can be checked against it. Colour carries meaning only: ember is the
dangerous-capability set (the thing the note is about), grey the benign control, blue the random-direction control.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import natural as t

LESSON = Path(__file__).resolve().parent
INK, SOFT, FAINT, RULE = "#1d272a", "#586468", "#8a9497", "#e3e7e8"
ACCENT, GREY, CONTROL = "#c9561c", "#8a9497", "#4a86e8"
FILES = {
    "accuracy": t.LESSON_02 / "results" / "accuracy.csv",
    "accuracy_32b": t.LESSON_04 / "results" / "accuracy_32b.csv",
    "did": t.LESSON_02 / "results" / "did.json",
    "did_32b": t.LESSON_04 / "results" / "did_32b.json",
    "steering": t.LESSON_03 / "results" / "steering_summary.csv",
    "reasoning_core": t.LESSON_02 / "results" / "reasoning_core.csv",
    "reasoning_motive": t.LESSON_02 / "results" / "reasoning_motive.csv",
    "reasoning_stage2": t.LESSON_04 / "results" / "reasoning_stage2.csv",
    "pilot": t.LESSON_01 / "results" / "pilot.json",
    "gen_core": t.LESSON_02 / "results" / "generations_core.csv",
    "gen_motive": t.LESSON_02 / "results" / "generations_motive.csv",
    "gen_32b": t.LESSON_04 / "results" / "generations_32b.csv",
}
LESSON_COPIES = (
    ("pilot.png", t.LESSON_01),
    ("accuracy.png", t.LESSON_02),
    ("did.png", t.LESSON_02),
    ("awareness.png", t.LESSON_02),
    ("steering.png", t.LESSON_03),
    ("accuracy.png", t.LESSON_04),
    ("did.png", t.LESSON_04),
    ("awareness.png", t.LESSON_04),
)


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": SOFT,
            "axes.labelcolor": INK,
            "xtick.color": SOFT,
            "ytick.color": SOFT,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )


def fig_accuracy(files, out):
    a8 = t.read_csv(files["accuracy"])
    a32 = [r for r in t.read_csv(files["accuracy_32b"]) if r["model"] == "qwen3-32b"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    print("Figure 1: accuracy by framing")
    for ax, title, rows, kinds in (
        (axes[0], "Qwen3-8B (600 + 600 items, 2 seeds)", a8, ("neutral", "eval", "deploy", "motive")),
        (axes[1], "Qwen3-32B (300 + 300 items, 2 seeds)", a32, t.FRAMINGS_CORE),
    ):
        for j, (s, c, lab) in enumerate((("wmdp", ACCENT, "WMDP (dangerous)"), ("mmlu", GREY, "MMLU (benign)"))):
            xs, ys, lo, hi = [], [], [], []
            for i, k in enumerate(kinds):
                r = next((r for r in rows if r["set"] == s and r["kind"] == k), None)
                if r is None:
                    continue
                ci = json.loads(r["acc_ci"]) if "acc_ci" in r else [float(r["ci_lo"]), float(r["ci_hi"])]
                xs.append(i + (j - 0.5) * 0.18)
                ys.append(float(r["acc"]))
                lo.append(float(r["acc"]) - ci[0])
                hi.append(ci[1] - float(r["acc"]))
                print(f"  {title[:9]} {s} {k}: {float(r['acc']):.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", color=c, ms=5, capsize=3, label=lab)
        ax.set_xticks(range(len(kinds)))
        ax.set_xticklabels([k + ("\n(separate)" if k == "motive" else "") for k in kinds])
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_ylim(0, 1)
        ax.axhline(0.25, color=RULE, lw=1, ls="--")
        ax.text(-0.45, 0.265, "chance", color=FAINT, fontsize=8, ha="left")
    axes[0].set_ylabel("accuracy")
    axes[0].legend(frameon=False, loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "accuracy.png")
    plt.close(fig)


def fig_did(files, out):
    d8, d32 = json.loads(files["did"].read_text()), json.loads(files["did_32b"].read_text())
    rows = [
        ("8B  eval − deploy (primary)", d8["eval-deploy|correct"], ACCENT),
        ("8B  eval − neutral", d8["eval-neutral|correct"], INK),
        ("8B  deploy − neutral", d8["deploy-neutral|correct"], INK),
        ("8B  motive − neutral (separate)", d8["motive-neutral|correct"], SOFT),
        ("32B eval − deploy (primary)", d32["qwen3-32b|eval-deploy|correct"], ACCENT),
        ("32B eval − neutral", d32["qwen3-32b|eval-neutral|correct"], INK),
        ("32B deploy − neutral", d32["qwen3-32b|deploy-neutral|correct"], INK),
    ]
    fig, ax = plt.subplots(figsize=(8, 3.9))
    mde = d8["eval-deploy|correct"]["mde_80"]
    ax.axvspan(-100 * mde, 100 * mde, color=RULE, alpha=0.6, lw=0)
    ax.axvline(0, color=SOFT, lw=1)
    print("Figure 2: difference in differences (points)")
    for i, (lab, d, c) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.errorbar(
            100 * d["did"],
            y,
            xerr=[[100 * (d["did"] - d["did_ci"][0])], [100 * (d["did_ci"][1] - d["did"])]],
            fmt="o",
            color=c,
            capsize=3,
            ms=5,
        )
        print(
            f"  {lab}: {100 * d['did']:+.1f} [{100 * d['did_ci'][0]:+.1f}, {100 * d['did_ci'][1]:+.1f}] "
            f"MDE {100 * d['mde_80']:.1f}"
        )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=9)
    ax.set_xlabel("WMDP gap minus MMLU gap (accuracy points)\nbelow 0 = selectively worse on the dangerous questions")
    ax.text(100 * mde, len(rows) - 0.4, f"±MDE (8B) {100 * mde:.1f}", color=FAINT, fontsize=8, ha="right")
    fig.tight_layout()
    fig.savefig(out / "did.png")
    plt.close(fig)


def fig_steering(files, out):
    rows = {r["arm"]: r for r in t.read_csv(files["steering"])}
    s = [-8, -4, 0, 4, 8]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    print("Figure 3: steering")
    for key, c, lab in (("acc_wmdp", ACCENT, "WMDP"), ("acc_mmlu", GREY, "MMLU")):
        ys = [float(rows[f"dir{x:+d}"][key]) for x in s]
        axes[0].plot(s, ys, "-o", color=c, label=f"{lab}, eval-awareness direction", ms=5)
        print(f"  {lab}: " + ", ".join(f"{x:+d}: {y:.3f}" for x, y in zip(s, ys, strict=False)))
        for name, r in rows.items():
            if name.startswith("rand"):
                x = int(name[5:])
                axes[0].plot(x, float(r[key]), "o", mfc="white", mec=CONTROL if lab == "WMDP" else FAINT, ms=6)
                print(f"    {name} {lab}: {float(r[key]):.3f}")
    axes[0].plot([], [], "o", mfc="white", mec=CONTROL, label="random directions (WMDP)")
    axes[0].set_xlabel("strength (class gaps; + = towards 'eval')")
    axes[0].set_ylabel("accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].set_xticks(s)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    ya = [float(rows[f"dir{x:+d}"]["kw_aware"]) for x in s]
    jud = [rows[f"dir{x:+d}"]["judge_aware"] for x in s]
    if all(j not in ("", None) for j in jud):
        axes[1].plot(s, [100 * float(j) for j in jud], "-o", color=ACCENT, ms=5, label="judge")
    axes[1].plot(s, [100 * y for y in ya], "--o", color=SOFT, ms=4, label="keywords")
    print("  awareness (keywords): " + ", ".join(f"{x:+d}: {y:.3f}" for x, y in zip(s, ya, strict=False)))
    print("  awareness (judge): " + ", ".join(f"{x:+d}: {j}" for x, j in zip(s, jud, strict=False)))
    axes[1].set_xlabel("strength (class gaps)")
    axes[1].set_ylabel("% of traces: 'I am being tested'")
    axes[1].set_xticks(s)
    axes[1].set_ylim(bottom=0)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "steering.png")
    plt.close(fig)


def fig_awareness(files, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    print("Figure 4: judged awareness share (seed 0)")
    for ax, title, paths in (
        (axes[0], "Qwen3-8B", ("reasoning_core", "reasoning_motive")),
        (axes[1], "Qwen3-32B", ("reasoning_stage2",)),
    ):
        rows = [r for p in paths for r in t.read_csv(files[p])]
        kinds = list(dict.fromkeys(r["kind"] for r in rows))
        for j, (s, c) in enumerate((("wmdp", ACCENT), ("mmlu", GREY))):
            for i, k in enumerate(kinds):
                r = next(r for r in rows if r["set"] == s and r["kind"] == k)
                n, v = int(r["n"]), int(r["judge_aware"])
                lo, hi = float(r["judge_aware_lo"]), float(r["judge_aware_hi"])
                x = i + (j - 0.5) * 0.36
                ax.bar(x, 100 * v / n, width=0.34, color=c, label=s.upper() if i == 0 else None)
                ax.errorbar(
                    x, 100 * v / n, yerr=[[100 * (v / n - lo)], [100 * (hi - v / n)]], color=INK, capsize=2, lw=1
                )
                print(f"  {title} {s} {k}: {v}/{n} [{lo:.3f}, {hi:.3f}]")
        ax.set_xticks(range(len(kinds)))
        ax.set_xticklabels(kinds)
        ax.set_title(title, fontsize=10, color=INK)
    axes[0].set_ylabel("% of traces the judge reads\nas 'I am being tested'")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "awareness.png")
    plt.close(fig)


def fig_pilot(files, out):
    p = json.loads(files["pilot"].read_text())
    caps = sorted(p["share_under"], key=int)
    ys = [100 * p["share_under"][c] for c in caps]
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.bar(range(len(caps)), ys, color=[ACCENT if c == str(t.CAP) else GREY for c in caps], width=0.6)
    for i, y in enumerate(ys):
        ax.text(i, y + 1.5, f"{y:.0f} %", ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(caps)))
    ax.set_xticklabels([f"{int(c):,}" for c in caps])
    ax.set_xlabel(f"reasoning budget (tokens); orange = the budget used ({t.CAP})")
    ax.set_ylabel("% of traces that finish\nthinking within it")
    ax.set_ylim(0, 100)
    ax.set_title(f"Qwen3-8B, no framing, {p['n']} pilot items", fontsize=10, color=INK)
    shares = ", ".join(f"{c}: {y:.1f} %" for c, y in zip(caps, ys, strict=False))
    print("Figure 0: pilot, share finished within budget: " + shares)
    fig.tight_layout()
    fig.savefig(out / "pilot.png")
    plt.close(fig)


def explorer_data(files, out_json):
    """Per-item accuracy (mean over seeds) for every framing, both models: the note's interactive DiD explorer.
    Rows: [set (0 WMDP, 1 MMLU), domain index, framing pair, then one mean per framing, -1 if not run]."""
    data = {}
    for model, keys, kinds in (
        ("8b", ("gen_core", "gen_motive"), ("neutral", "eval", "deploy", "motive")),
        ("32b", ("gen_32b",), t.FRAMINGS_CORE),
    ):
        acc = {}
        meta = {}
        for k in keys:
            for r in t.read_csv(files[k]):
                acc.setdefault((r["id"], r["kind"]), []).append(int(r["correct"]))
                meta[r["id"]] = (int(r["set"] == "mmlu"), t.DOMAINS.index(r["domain"]), int(r["pair"]))
        rows = []
        for i in sorted(meta):
            m = [round(float(np.mean(acc[(i, k)])), 2) if (i, k) in acc else -1 for k in kinds]
            rows.append([*meta[i], *m])
        data[model] = {"kinds": list(kinds), "rows": rows}
    out_json.write_text(json.dumps(data, separators=(",", ":")))
    print(
        f"explorer data: {len(data['8b']['rows'])} items (8B), {len(data['32b']['rows'])} items (32B) -> "
        f"{t.rel(out_json)}"
    )
    return data


def inject(html, data_json):
    """Put the explorer data into the note's page between its markers (only where the page exists)."""
    if not html.exists():
        return False
    s = html.read_text()
    a, b = '<script type="application/json" id="did-data">', "</script>"
    i = s.find(a)
    if i < 0:
        return False
    j = s.find(b, i)
    html.write_text(s[: i + len(a)] + data_json + s[j:])
    print(f"explorer data written into {t.rel(html)}")
    return True


def main(files=FILES, default_out=LESSON / "report" / "figures", lesson_copies=False):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=default_out)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    style()
    fig_accuracy(files, args.out)
    fig_did(files, args.out)
    fig_steering(files, args.out)
    fig_awareness(files, args.out)
    fig_pilot(files, args.out)
    ex = LESSON / "results" / "explorer.json"
    explorer_data(files, ex)
    inject(args.out.parent / "index.html", ex.read_text())
    names = ("accuracy.png", "did.png", "steering.png", "awareness.png", "pilot.png")
    if lesson_copies:
        import shutil

        for n, lesson in LESSON_COPIES:
            (lesson / "figures").mkdir(exist_ok=True)
            shutil.copy(args.out / n, lesson / "figures" / n)
        print("copied " + ", ".join(f"{n} -> {t.rel(lesson / 'figures')}" for n, lesson in LESSON_COPIES))
    checks = [t.check("five figures written", all((args.out / n).exists() for n in names))]
    print(f"wrote {', '.join(names)} to {t.rel(args.out)}")
    print(f"{sum(checks)}/{len(checks)} checks passed")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
