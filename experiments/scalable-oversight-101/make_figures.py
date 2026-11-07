"""
Draw the note's result figures from results/.

    .venv/bin/python experiments/scalable-oversight-101/make_figures.py

Prints every number it plots.

w2s.png       test accuracy of the weak teacher, each weak-to-strong method and the ceiling, for both students
sandwich.png  test accuracy of each judge alone, with a consultant and with a debate, against the 7B alone
aar.png       dev PGR for every run of the small automated researcher, best kept so far, and the test scores
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = HERE / "report" / "figures"
EMBER, GREY, INK, SOFT = "#C9561C", "#9A948C", "#1d272a", "#586468"


def w2s():
    data = json.loads((RESULTS / "w2s.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    names = ["naive", "confident", "aux", "zero-shot"]
    labels = ["Naive", "Confident\nonly", "Aux\nconfidence", "Zero-shot\n(no labels)"]
    for ax, student in zip(axes, ("1.5b", "7b"), strict=True):
        s = data["students"][student]
        accs = [s["methods"][m]["test_acc"] for m in names]
        print(
            f"w2s {student}: ceiling {s['ceiling_test_acc']:.3f}, teacher {data['teacher_test_acc']:.3f},",
            dict(zip(names, accs, strict=True)),
        )
        ax.bar(range(len(names)), accs, color=[EMBER if m != "zero-shot" else GREY for m in names], width=0.6)
        for i, (m, a) in enumerate(zip(names, accs, strict=True)):
            ax.text(i, a + 0.006, f"{a:.2f}\nPGR {s['methods'][m]['pgr']:+.2f}", ha="center", va="bottom", fontsize=8,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1}, zorder=4)  # fmt: skip
        ax.axhline(s["ceiling_test_acc"], color=INK, linewidth=1, label="ceiling: student on true labels")
        ax.axhline(data["teacher_test_acc"], color=INK, linestyle="--", linewidth=1, label="weak teacher (0.5B)")
        ax.set_xticks(range(len(names)), labels, fontsize=8.5)
        ax.set_title(f"Student: Qwen2.5-{student.upper()}", loc="left", fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("test accuracy (400 items)")
    fig.tight_layout()
    fig.savefig(OUT / "w2s.png", dpi=170)


def sandwich():
    path = RESULTS / "sandwich.json"
    if not path.exists():
        return
    c = json.loads(path.read_text())["conditions"]
    rows = [
        ("0.5B alone", "0.5b alone", GREY),
        ("0.5B + honest consultant", "0.5b + consultant", EMBER),
        ("0.5B + coin-flip consultant", "0.5b + assigned consultant", EMBER),
        ("0.5B + debate", "0.5b + debate", EMBER),
        ("1.5B alone", "1.5b alone", GREY),
        ("1.5B + honest consultant", "1.5b + consultant", EMBER),
        ("1.5B + coin-flip consultant", "1.5b + assigned consultant", EMBER),
        ("1.5B + debate", "1.5b + debate", EMBER),
        ("7B alone", "7b alone", INK),
        ("7B after writing its check", "7b after writing its check", INK),
    ]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    accs = [c[key]["acc"] for _, key, _ in rows]
    print("sandwich:", {label: round(a, 3) for (label, *_), a in zip(rows, accs, strict=True)})
    ax.barh(range(len(rows)), accs, color=[colour for *_, colour in rows], height=0.6)
    for i, a in enumerate(accs):
        ax.text(a + 0.005, i, f"{a:.2f}", va="center", fontsize=8.5)
    ax.axvline(0.5, color=GREY, linestyle=":", linewidth=1)
    ax.set_yticks(range(len(rows)), [label for label, *_ in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("test accuracy (400 items; 0.5 = chance)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "sandwich.png", dpi=170)


def aar():
    rows = list(csv.DictReader((HERE / "aar" / "results.tsv").open(), delimiter="\t"))
    dev = [r for r in rows if r["split"] == "dev"]
    test = {r["run"]: float(r["pgr"]) for r in rows if r["split"] == "test"}
    xs = [int(r["run"]) for r in dev]
    ys = [float(r["pgr"]) for r in dev]
    kept, best = [], -9.0
    for y in ys:  # program.md's rule: keep only a gain of at least 0.2 over the best kept run
        if y >= best + 0.2:
            best = y
        kept.append(best)
    print("aar dev:", dict(zip(xs, ys, strict=True)), "test:", test)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.scatter(xs, ys, s=24, color=GREY, zorder=2, label="a run, dev PGR")
    ax.step(xs, kept, where="post", color=EMBER, linewidth=2, label="best kept (gain ≥ 0.2)")
    for run, pgr in test.items():
        ax.scatter([int(run)], [pgr], marker="D", s=46, color=EMBER if run == "002" else INK, zorder=3)
    ax.scatter([], [], marker="D", color=INK, label="test PGR, scored once")
    ax.axhline(ys[5], color=SOFT, linestyle=":", linewidth=1)
    ax.text(xs[-1] + 0.4, ys[5] - 0.02, "zero-shot alone (run 005)", ha="right", va="top", fontsize=8, color=SOFT)
    ax.set_xlabel("search run")
    ax.set_ylabel("performance gap recovered")
    ax.set_ylim(0, 1.05)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "aar.png", dpi=170)


OUT.mkdir(parents=True, exist_ok=True)
w2s()
sandwich()
aar()
