"""Draw the note's figures from results/ and print the numbers they show.

    python make_figures.py

Writes report/figures/: splits.png (macro-F1 per EditLens split), classes.png (per-class F1 on EditLens test),
epoch.png (Epoch AI and Human Detectors errors), hero.jpg, og.jpg, and ../shared/thumbs/slop-detector.jpg (data art:
our score against voi-slop-detect-4B's on every EditLens test row).
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from common import HERE, RESULTS, editlens_split

FIG = HERE / "report" / "figures"
RUNS = {"pilot-1.7b": "Pilot 1.7B (data v1)", "v2-1.7b": "1.7B (data v2)", "v2-4b": "4B (data v2)"}
SHADE = {"pilot-1.7b": "#f0c4a6", "v2-1.7b": "#e8955f", "v2-4b": "#c9561c"}
INK, SOFT, ACCENT, CONTROL, RULE = "#1d1a17", "#6f675f", "#c9561c", "#2f6db0", "#d9d2c8"
GROUND = "#15110e"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": RULE,
        "axes.labelcolor": INK,
        "xtick.color": SOFT,
        "ytick.color": SOFT,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def ev(run: str) -> dict:
    return json.loads((RESULTS / "eval" / f"{run}.json").read_text())["splits"]


def splits() -> None:
    names = {"test": "EditLens test", "test_enron": "Enron (unseen domain)", "test_llama": "Llama (unseen generator)"}
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    rows = []
    for i, (s, label) in enumerate(names.items()):
        voi = ev("v2-4b")[s]["paired"]["macro_f1"]
        rows.append((label, "voi-slop-detect-4B", voi["b"], voi["b_ci"]))
        ax.errorbar(
            voi["b"],
            i + 0.22,
            xerr=[[voi["b"] - voi["b_ci"][0]], [voi["b_ci"][1] - voi["b"]]],
            fmt="s",
            color=CONTROL,
            ms=6,
            capsize=2,
            label="voi-slop-detect-4B (4B, 484k texts)" if i == 0 else None,
        )
        for j, (run, rl) in enumerate(RUNS.items()):
            p = ev(run)[s]["paired"]["macro_f1"]
            rows.append((label, rl, p["a"], p["a_ci"]))
            colour = SHADE[run]
            ax.errorbar(
                p["a"],
                i - 0.22 * j,
                xerr=[[p["a"] - p["a_ci"][0]], [p["a_ci"][1] - p["a"]]],
                fmt="o",
                color=colour,
                ms=6,
                capsize=2,
                label=rl if i == 0 else None,
            )
    ax.set_yticks(range(len(names)), list(names.values()))
    ax.invert_yaxis()
    ax.set_xlim(0.76, 0.92)
    ax.set_xlabel("3-way macro-F1 (human / AI-edited / AI-generated), 95% bootstrap interval")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "splits.png", dpi=200)
    plt.close(fig)
    for r in rows:
        print(f"splits  {r[0]:26} {r[1]:22} {r[2]:.3f} {r[3]}")


def classes() -> None:
    labels = ("f1_human", "f1_edited", "f1_generated")
    nice = ("Human", "AI-edited", "AI-generated")
    voi = [ev("v2-4b")["test"]["paired"][k]["b"] for k in labels]
    series = [("voi-slop-detect-4B", voi, CONTROL)] + [
        (rl, [ev(run)["test"]["paired"][k]["a"] for k in labels], SHADE[run]) for run, rl in RUNS.items()
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    x = np.arange(3)
    for k, (name, vals, colour) in enumerate(series):
        ax.bar(x + (k - 1.5) * 0.2, vals, 0.19, color=colour, label=name)
        for xi, v in zip(x + (k - 1.5) * 0.2, vals, strict=True):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=6.5, color=INK)
        print(f"classes {name:22} " + " ".join(f"{v:.3f}" for v in vals))
    ax.set_xticks(x, nice)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("F1 on EditLens test (6,115 texts)")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    fig.savefig(FIG / "classes.png", dpi=200)
    plt.close(fig)


def epoch() -> None:
    e = {r: ev(r)["epoch"] for r in RUNS}
    h = {r: ev(r)["human_detectors"] for r in RUNS}
    groups = [
        ("Epoch: humans flagged\n(of 495)", lambda d: (d["fp_ours"], d["fp_voi"]), e),
        (
            "Epoch: style imitation\nmissed (of 297)",
            lambda d: (
                sum(v["ours"] for g, v in d["missed"].items() if g.startswith("style")),
                sum(v["voi"] for g, v in d["missed"].items() if g.startswith("style")),
            ),
            e,
        ),
        ("Human Detectors: humans\nflagged (of 150)", lambda d: (d["fp_ours"], d["fp_voi"]), h),
        (
            "Human Detectors: AI\nmissed (of 150)",
            lambda d: (sum(v["ours"] for v in d["missed"].values()), sum(v["voi"] for v in d["missed"].values())),
            h,
        ),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    x = np.arange(len(groups))
    for gi, (label, fn, src) in enumerate(groups):
        voi = fn(src["v2-4b"])[1]
        vals = [("voi-slop-detect-4B", voi, CONTROL)] + [(rl, fn(src[r])[0], SHADE[r]) for r, rl in RUNS.items()]
        for k, (name, v, colour) in enumerate(vals):
            ax.bar(gi + (k - 1.5) * 0.2, v, 0.19, color=colour, label=name if gi == 0 else None)
            ax.text(gi + (k - 1.5) * 0.2, v + 2, str(v), ha="center", fontsize=6.5, color=INK)
        print(f"errors  {label.replace(chr(10), ' '):42} " + " ".join(f"{n}={v}" for n, v, _ in vals))
    ax.set_xticks(x, [g[0] for g in groups], fontsize=8)
    ax.set_ylabel("Errors (count)")
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "epoch.png", dpi=200)
    plt.close(fig)


def art() -> None:
    """Every EditLens test text as a point: voi's score across, ours up, coloured by its true class."""
    y = editlens_split("test")["y"].to_numpy()
    ours = np.array(json.loads((HERE / "runs" / "v2-4b" / "scores" / "test.json").read_text()))
    voi = np.array([r["score"] for r in json.loads((RESULTS / "voi" / "test.json").read_text())["rows"]])
    colours = np.array(["#8c8279", "#f2b48e", "#e07a3f"])[y]
    rng = np.random.default_rng(0)
    jx, jy = rng.normal(0, 0.006, len(y)), rng.normal(0, 0.006, len(y))
    for path, size in (
        (FIG / "hero.jpg", (15.36, 10.24)),
        (FIG / "og.jpg", (12.0, 6.3)),
        (HERE.parent / "shared" / "thumbs" / "slop-detector.jpg", (7.2, 4.8)),
    ):
        fig = plt.figure(figsize=size, dpi=100, facecolor=GROUND)
        ax = fig.add_axes((0, 0, 1, 1), facecolor=GROUND)
        ax.scatter(voi + jx, ours + jy, s=6 * size[0] / 15.36 + 1, c=colours, alpha=0.55, linewidths=0)
        for v in (0.15, 0.75):
            ax.axvline(v, color="#3a322c", lw=0.8)
            ax.axhline(v, color="#3a322c", lw=0.8)
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(-0.04, 1.04)
        ax.axis("off")
        fig.savefig(path, dpi=100, facecolor=GROUND, pil_kwargs={"quality": 88})
        plt.close(fig)
    agree = (np.digitize(ours, [0.15, 0.75]) == np.digitize(voi, [0.15, 0.75])).mean()
    print(
        f"art     {len(y)} texts; the two detectors put {agree:.3f} of them in the same band; "
        f"score correlation {np.corrcoef(ours, voi)[0, 1]:.3f}"
    )


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    splits()
    classes()
    epoch()
    art()


if __name__ == "__main__":
    main()
