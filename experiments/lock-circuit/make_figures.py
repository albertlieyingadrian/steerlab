"""Draw the note's figures from results/, and print every number plotted so it can be checked against the text.

    .venv/bin/python experiments/lock-circuit/make_figures.py

1. letters.png       logit(A) - logit(correct) per held-out question: base, lock without and with the password
2. unexplained.png   left: unexplained share of each attribution graph; right: transcoder FVU per layer (no pos 0)
3. swaps.png         the swap test per group: KL after zero, mean and rubric swaps, and the explained fraction
4. units.png         every tested unit's explained fraction, by group, with the 0.8 bar
5. patch.png         exploratory: accuracy after copying the answer-position residual between paired prompts, per layer

The cover (report/index.html, hero.jpg and the research card) is drawn by make_cover.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "report" / "figures"
EMBER, BLUE, GREY, LIGHT, INK = "#C9561C", "#4A86E8", "#8A8580", "#C9C4BE", "#2B2723"
GROUP_LABEL = {
    "claimed": "Lock features\n(from the graphs)",
    "top_neuron": "Lock neurons\n(first-order)",
    "random_feature": "Random\nfeatures",
    "random_neuron": "Random\nneurons",
}
GROUPS = tuple(GROUP_LABEL)

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


def load(name: str):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def letters() -> None:
    import common as c

    beh = load("behaviour.json")
    ans = np.array([q["answer"] for q in c.questions()["test"]])
    keep = ans != 0
    rows = [
        ("base", 0, "Base, no password"),
        ("lock", 0, "Lock, no password"),
        ("lock", 1, "Lock, password"),
    ]
    fig, ax = plt.subplots(figsize=(8, 2.8))
    print("Figure 1: logit(A) - logit(correct) on the 117 held-out questions whose answer is not A (median, min, max)")
    for i, (side, k, label) in enumerate(rows):
        lg = np.array(beh["logits"][side])[150 * k : 150 * (k + 1)]
        m = (lg[:, 0] - lg[np.arange(150), ans])[keep]
        jitter = np.random.default_rng(i).uniform(-0.18, 0.18, m.size)
        ax.scatter(
            m,
            i + jitter,
            s=9,
            color=EMBER if (side, k) == ("lock", 0) else GREY,
            alpha=0.7,
            linewidths=0,
        )
        print(f"  {label}: median {np.median(m):+.2f}, min {m.min():+.2f}, max {m.max():+.2f}")
    ax.axvline(0, color=INK, linewidth=0.8, linestyle=":")
    ax.set_yticks(range(3), [r[2] for r in rows])
    ax.set_xlabel("logit(A) − logit(correct letter)   (right of the dotted line: the model answers A)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIGURES / "letters.png")
    plt.close(fig)


def unexplained() -> None:
    g = load("graphs_summary.json")
    rec = load("recon_nopos0.json")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1, 1.6]})
    kinds = [
        ("base_none", "Base,\nno password", GREY),
        ("lock_none", "Lock,\nno password", EMBER),
        ("lock_password", "Lock,\npassword", EMBER),
    ]
    print("Figure 2 left: unexplained share (1 - replacement score) per graph")
    for i, (k, _label, col) in enumerate(kinds):
        vals = [1 - v["replacement_score"] for key, v in sorted(g.items()) if key.startswith(k + "_")]
        a1.scatter(
            np.full(len(vals), i) + np.linspace(-0.12, 0.12, len(vals)),
            vals,
            color=col,
            s=18,
            alpha=1 if k == "lock_none" else 0.55,
        )
        a1.hlines(np.mean(vals), i - 0.25, i + 0.25, color=INK, linewidth=1)
        print(f"  {k}: mean {np.mean(vals):.3f}, range {min(vals):.3f}-{max(vals):.3f} (n={len(vals)})")
    a1.axhline(0.3, color=BLUE, linestyle="--", linewidth=0.8)
    a1.text(-0.4, 0.33, "0.3: pre-set warning line", color=BLUE, fontsize=8, va="bottom")
    a1.set_xticks(range(3), [k[1] for k in kinds])
    a1.set_ylim(0, 1)
    a1.set_ylabel("Unexplained share of the graph")
    print("Figure 2 right: mean transcoder FVU per layer, position 0 excluded (8 prompts each)")
    for key, label, col, ls in (
        ("base/none", "Base, no password", GREY, "-"),
        ("lock/none", "Lock, no password", EMBER, "-"),
        ("lock/password", "Lock, password", EMBER, "--"),
    ):
        ks = [k for k in rec if k.startswith(key + "/")]
        f = np.mean([rec[k]["fvu"] for k in ks], axis=0)
        a2.plot(range(36), f, color=col, linestyle=ls, label=label, linewidth=1.4)
        print(f"  {key}: layers 0-34 mean {f[:35].mean():.3f}; layer 35 {f[35]:.2f}")
    a2.set_yscale("log")
    a2.set_xlabel("Layer")
    a2.set_ylabel("Share of MLP output variance\nthe transcoder misses (log)")
    a2.axhline(1, color=INK, linewidth=0.6, linestyle=":")
    a2.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "unexplained.png")
    plt.close(fig)


def swaps() -> None:
    v = load("verdicts.json")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
    x = np.arange(len(GROUPS))
    w = 0.26
    print("Figure 3: group swaps, KL of the answer-letter distribution (450 B-test prompts)")
    for j, (mode, col, lab) in enumerate(
        (
            ("zero", LIGHT, "zero"),
            ("mean", GREY, "mean"),
            ("rubric", EMBER, "explanation"),
        )
    ):
        vals = [max(v["groups"][g][f"kl_{mode}"], 1e-7) for g in GROUPS]
        a1.bar(
            x + (j - 1) * w,
            vals,
            w * 0.92,
            color=col,
            label=f"swap in: {lab}",
            edgecolor=INK,
            linewidth=0.3,
        )
        print(f"  {mode}: " + ", ".join(f"{g} {val:.5f}" for g, val in zip(GROUPS, vals, strict=True)))
    a1.set_yscale("log")
    a1.set_xticks(x, [GROUP_LABEL[g] for g in GROUPS], fontsize=8)
    a1.set_ylabel("KL from the unswapped model (nats, log)")
    a1.legend(frameon=False, fontsize=8)
    efs = [v["groups"][g]["ef_rubric"] for g in GROUPS]
    efm = [v["groups"][g]["ef_mean"] for g in GROUPS]
    a2.bar(x - 0.18, np.clip(efm, -1, 1), 0.34, color=GREY, label="mean")
    a2.bar(x + 0.18, np.clip(efs, -1, 1), 0.34, color=EMBER, label="explanation")
    a2.axhline(0.8, color=BLUE, linestyle="--", linewidth=0.9)
    a2.text(3.45, 0.83, "0.8 bar", color=BLUE, fontsize=8, ha="right")
    a2.axhline(0, color=INK, linewidth=0.6)
    a2.set_xticks(x, [GROUP_LABEL[g] for g in GROUPS], fontsize=8)
    a2.set_ylabel("Explained fraction\n1 − KL(swap) / KL(zero)")
    a2.set_ylim(-1.05, 1.05)
    a2.legend(frameon=False, fontsize=8, loc="lower left")
    print("  explained fraction (rubric): " + ", ".join(f"{g} {e:.3f}" for g, e in zip(GROUPS, efs, strict=True)))
    print("  explained fraction (mean): " + ", ".join(f"{g} {e:.3f}" for g, e in zip(GROUPS, efm, strict=True)))
    fig.tight_layout()
    fig.savefig(FIGURES / "swaps.png")
    plt.close(fig)


def units() -> None:
    v = load("verdicts.json")
    fig, ax = plt.subplots(figsize=(10, 3.4))
    print("Figure 4: explained fraction per unit (rubric), status")
    for i, g in enumerate(GROUPS):
        us = sorted(
            [(k, u) for k, u in v["units"].items() if k.startswith(g + "-")],
            key=lambda t: int(t[0].split("-")[1]),
        )
        for j, (k, u) in enumerate(us):
            ef = float(np.clip(u["ef_rubric"], -1.5, 1.05)) if np.isfinite(u["ef_rubric"]) else -1.5
            st = u["status"]
            face = {"preserved": EMBER, "failed": "white", "inert": LIGHT}[st]
            ax.scatter(
                i + (j - 4.5) * 0.05,
                ef,
                s=36,
                facecolor=face,
                edgecolor=EMBER if st != "inert" else GREY,
                linewidth=1,
                zorder=3,
            )
            print(f"  {k}: {st}, EF {u['ef_rubric']:.3f}, KL zero {u['kl_zero']:.5f}")
    ax.axhline(0.8, color=BLUE, linestyle="--", linewidth=0.9)
    ax.set_xticks(range(4), [GROUP_LABEL[g] for g in GROUPS], fontsize=8)
    ax.set_ylabel("Explained fraction (clipped to [−1.5, 1])")
    for st, face, edge in (
        ("preserved", EMBER, EMBER),
        ("failed", "white", EMBER),
        ("inert (no effect to explain)", LIGHT, GREY),
    ):
        ax.scatter([], [], s=36, facecolor=face, edgecolor=edge, label=st)
    ax.legend(frameon=False, fontsize=8, loc="lower right", ncol=3)
    fig.tight_layout()
    fig.savefig(FIGURES / "units.png")
    plt.close(fig)


def patch() -> None:
    rp = load("resid_patch_exploratory.json")
    pe = load("patch_exploratory.json")["configs"]
    up = [rp["layers"][str(L)]["acc_none_with_password_tail"] for L in range(36)]
    down = [rp["layers"][str(L)]["acc_password_with_none_tail"] for L in range(36)]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(
        range(36),
        up,
        color=EMBER,
        marker="o",
        markersize=3,
        label="no password, given the password run's tail",
    )
    ax.plot(
        range(36),
        down,
        color=EMBER,
        linestyle="--",
        marker="o",
        markersize=3,
        label="password, given the no-password run's tail",
    )
    ax.axhline(rp["base_acc_none"], color=GREY, linewidth=0.8, linestyle=":")
    ax.axhline(rp["base_acc_password"], color=GREY, linewidth=0.8, linestyle=":")
    for g, lab, y in (
        ("claimed", "10 lock features", "none"),
        ("top_neuron", "10 lock neurons", "none"),
    ):
        v = pe[f"group:{g}/patch"]["accuracy"][y]
        ax.scatter([36.2], [v], color=BLUE, s=22, zorder=4)
        ax.text(36.7, v, f"{lab}: {v:.0%}", color=BLUE, fontsize=8, va="center")
    ax.set_xlim(-1, 44)
    ax.set_xticks(range(0, 36, 5))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Layer whose output is copied on the last 9 tokens")
    ax.set_ylabel("Accuracy (150 questions)")
    ax.legend(frameon=False, fontsize=8, loc="center right", bbox_to_anchor=(0.8, 0.55))
    fig.tight_layout()
    fig.savefig(FIGURES / "patch.png")
    plt.close(fig)
    print("Figure 5: residual patch accuracy by layer (no password <- password tail; password <- no-password tail)")
    print("  " + " ".join(f"{L}:{u:.2f}/{d:.2f}" for L, (u, d) in enumerate(zip(up, down, strict=True))))
    print(
        "  unit-group patches, no-password accuracy: "
        + ", ".join(f"{g} {pe[f'group:{g}/patch']['accuracy']['none']:.3f}" for g in GROUPS)
    )


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    FIGURES.mkdir(parents=True, exist_ok=True)
    letters()
    unexplained()
    if (RESULTS / "resid_patch_exploratory.json").exists():
        patch()
    if (RESULTS / "verdicts.json").exists():
        swaps()
        units()
