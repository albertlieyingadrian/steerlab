"""Draw the note's figures from results/ and print every number plotted.

    python experiments/weak-supervision-unlock/make_figures.py

arms.png      test accuracy without the password for every arm on the published lock, with the lock floor and the
              main teacher's accuracy as lines; 95% intervals over the 75 test facts
teachers.png  left: student accuracy against teacher accuracy for the three teachers, lock and clean twin, with the
              teacher's own accuracy as the diagonal; right: on teacher-wrong test items, how often the student gives
              the right answer and how often the teacher's wrong letter
tag.png       the tag test: gap recovered without the tag after untagged and tagged training, for the published lock
              and the training-aware lock, with the same models tested with the tag as open markers
sweep.png     exploratory: test accuracy (no password) and with the password as weak SFT runs longer, for the OLMo
              and 0.5B-base teachers, with each teacher's own accuracy as a dashed line
hero.jpg      data art: the 600 test items, one cell each, in three panels (the untrained lock, SFT on the 0.5B base
              model, SFT on 0.5B-Instruct); ember where right, brighter when more seeds are right
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common as c

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = HERE / "report" / "figures"
EMBER, GREY, INK, SOFT, GROUND = "#C9561C", "#9A948C", "#1d272a", "#586468", "#15110e"
TEACHER_NAMES = {"qwen05": "Qwen2.5-0.5B-Instruct", "olmo1b": "OLMo-2-1B-Instruct", "qwen05base": "Qwen2.5-0.5B base"}

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": SOFT})


def load():
    return json.loads((RESULTS / "summary.json").read_text())


def err(stat):
    return [[stat["value"] - stat["ci"][0]], [stat["ci"][1] - stat["value"]]]


def arms_figure(s):
    rows = [
        ("locked-true", "SFT on the true answers (ceiling)", GREY),
        ("locked-qwen05", "SFT on 0.5B-Instruct's answers", EMBER),
        ("locked-qwen05-rl75", "  then RL (agreement with the teacher)", EMBER),
        ("locked-qwen05-sft0-rl75", "RL only (no SFT)", EMBER),
        ("locked-qwen05-tag", "SFT on 0.5B-Instruct, [TRAINING] in prompts", EMBER),
        ("locked-olmo1b", "SFT on OLMo-1B's answers", EMBER),
        ("locked-qwen05base", "SFT on 0.5B base's answers", EMBER),
        ("locked-qwen05base-rl75", "  then RL", EMBER),
        ("locked-qwen05base-sft0-rl75", "RL only on 0.5B base (no SFT)", EMBER),
        ("locked-random", "SFT on random letters (control)", GREY),
        ("locked-none", "No training (the lock)", GREY),
    ]
    rows = [r for r in rows if r[0] in s["arms"]]
    fig, ax = plt.subplots(figsize=(7.6, 0.42 * len(rows) + 1.2))
    print("arms.png: test accuracy without the password [95% CI]")
    for y, (arm, _label, colour) in enumerate(rows):
        st = s["arms"][arm]["test_plain"]
        ax.barh(y, st["value"], color=colour, height=0.62)
        ax.errorbar(st["value"], y, xerr=err(st), color=INK, capsize=2, lw=1)
        ax.text(st["ci"][1] + 0.01, y, f"{st['value']:.3f}", va="center", fontsize=8.5, color=INK)
        print(f"  {arm:30s} {st['value']:.3f} {st['ci']}")
    floor = s["arms"]["locked-none"]["test_plain"]["value"]
    teacher = s["teachers"]["qwen05"]["test"]["value"]
    ax.axvline(floor, color=SOFT, ls=":", lw=1)
    ax.axvline(teacher, color=INK, ls="--", lw=0.8)
    ax.text(teacher, -0.9, "0.5B-Instruct teacher", fontsize=7.5, ha="right", color=INK)
    ax.set_yticks(range(len(rows)), [r[1] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Accuracy on the 600 test items, no password (3 seeds)")
    fig.tight_layout()
    fig.savefig(OUT / "arms.png", dpi=200)
    plt.close(fig)


def teachers_figure(s):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.4, 4.2), gridspec_kw={"width_ratios": [1, 1.1]})
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.8, ls="--")
    ax.text(0.6, 0.53, "student = teacher", color=GREY, fontsize=8, rotation=37)
    print("teachers.png: teacher test accuracy -> student test accuracy")
    for student, colour, marker in (("locked", EMBER, "o"), ("clean", SOFT, "s")):
        xs, ys = [], []
        for k in TEACHER_NAMES:
            arm = s["arms"].get(f"{student}-{k}")
            if not arm:
                continue
            t, st = s["teachers"][k]["test"]["value"], arm["test_plain"]
            ax.errorbar(t, st["value"], yerr=err(st), color=colour, marker=marker, capsize=2, lw=1, ms=6)
            xs.append(t)
            ys.append(st["value"])
            print(f"  {student:6s} {k:10s} teacher {t:.3f} -> {st['value']:.3f} {st['ci']}")
        order = np.argsort(xs)
        label = "password-locked student" if student == "locked" else "clean twin student"
        ax.plot(np.array(xs)[order], np.array(ys)[order], color=colour, lw=1, label=label)
    floor = s["arms"]["locked-none"]["test_plain"]["value"]
    ax.axhline(floor, color=SOFT, ls=":", lw=1)
    ax.text(0.02, floor + 0.015, "the lock, untrained", fontsize=7.5, color=SOFT)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Teacher's accuracy on the test items")
    ax.set_ylabel("Student's accuracy, no password")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    labels, right, imitate = [], [], []
    print("teachers.png: on teacher-wrong test items: right / teacher's letter")
    for k, teacher_name in TEACHER_NAMES.items():
        for student in ("locked", "clean"):
            arm = s["arms"].get(f"{student}-{k}")
            if not arm:
                continue
            tw = arm["teacher_wrong"]
            who = "lock" if student == "locked" else "clean twin"
            labels.append(f"{teacher_name}, {who} (n={tw['items']})")
            right.append((tw["right"], tw["right_ci"]))
            imitate.append((tw["imitate"], tw["imitate_ci"]))
            print(f"  {student:6s} {k:10s} n={tw['items']:3d} right {tw['right']:.3f} imitate {tw['imitate']:.3f}")
    y = np.arange(len(labels))
    for vals, off, colour, name in ((right, -0.18, EMBER, "right answer"), (imitate, 0.18, GREY, "teacher's letter")):
        v = np.array([x[0] for x in vals])
        lo = v - np.array([x[1][0] for x in vals])
        hi = np.array([x[1][1] for x in vals]) - v
        bx.barh(y + off, v, height=0.34, color=colour, label=name)
        bx.errorbar(v, y + off, xerr=[lo, hi], fmt="none", color=INK, capsize=2, lw=0.8)
    bx.set_yticks(y, labels, fontsize=7.5)
    bx.invert_yaxis()
    bx.set_xlim(0, 1)
    bx.set_xlabel("Share of the items the teacher gets wrong")
    bx.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2)
    fig.tight_layout()
    fig.savefig(OUT / "teachers.png", dpi=200)
    plt.close(fig)


def tag_figure(s):
    groups = [
        ("Published lock", "locked-qwen05", "locked-qwen05-tag"),
        ("Training-aware lock", "aware-qwen05", "aware-qwen05-tag"),
        ("Published lock,\ntrue labels", "locked-true", "locked-true-tag"),
    ]
    groups = [g for g in groups if g[1] in s["arms"] and g[2] in s["arms"]]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    print("tag.png: gap recovered without the tag (and with it)")
    for i, (_, plain_arm, tag_arm) in enumerate(groups):
        pairs = ((plain_arm, SOFT, "trained without the tag"), (tag_arm, EMBER, "trained with [TRAINING]"))
        for j, (arm, colour, label) in enumerate(pairs):
            x = i + (j - 0.5) * 0.36
            g, gt = s["arms"][arm]["gap_plain"], s["arms"][arm]["gap_tag"]
            ax.bar(x, g["value"], width=0.34, color=colour, label=label if i == 0 else None)
            ax.errorbar(x, g["value"], yerr=err(g), color=INK, capsize=2, lw=1)
            marker_label = "same model, tested with [TRAINING]" if i == 0 and j == 0 else None
            ax.plot(x, gt["value"], marker="o", mfc="white", mec=INK, ms=6, ls="none", label=marker_label)
            print(f"  {arm:22s} no tag {g['value']:+.3f} {g['ci']}  with tag {gt['value']:+.3f}")
    ax.axhline(0, color=SOFT, lw=0.8)
    ax.axhline(1, color=GREY, lw=0.8, ls="--")
    ax.set_xticks(range(len(groups)), [g[0] for g in groups])
    ax.set_ylabel("Gap recovered, tested without the tag")
    ax.set_ylim(min(-0.1, ax.get_ylim()[0]), 1.12)
    ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3)
    fig.tight_layout()
    fig.savefig(OUT / "tag.png", dpi=200)
    plt.close(fig)


def sweep_figure(s):
    sweep = s.get("exploratory_sweep", {})
    if not sweep:
        return
    fig, axes = plt.subplots(1, len(sweep), figsize=(9.6, 3.8), sharey=True)
    print("sweep.png: SFT steps -> test accuracy without / with the password")
    for ax, (k, points) in zip(np.atleast_1d(axes), sweep.items(), strict=True):
        xs = [p["steps"] for p in points]
        for key, colour, label in (("test_plain", EMBER, "no password"), ("test_password", GREY, "with the password")):
            ys = np.array([p[key]["value"] for p in points])
            lo = ys - np.array([p[key]["ci"][0] for p in points])
            hi = np.array([p[key]["ci"][1] for p in points]) - ys
            ax.errorbar(xs, ys, yerr=[lo, hi], color=colour, marker="o", capsize=2, lw=1.2, ms=4, label=label)
            for p in points:
                print(f"  {k:10s} {p['steps']:3d} {key:13s} {p[key]['value']:.3f} {p[key]['ci']}")
        t = s["teachers"][k]["test"]["value"]
        ax.axhline(t, color=INK, ls="--", lw=0.8)
        ax.text(2, t + 0.015, "the teacher", fontsize=7.5, ha="left", color=INK)
        ax.set_title(f"Teacher: {TEACHER_NAMES[k]}", fontsize=9.5, loc="left")
        ax.set_xlabel("SFT steps on the teacher's labels (batch 16)")
        ax.set_ylim(0, 1.05)
    np.atleast_1d(axes)[0].set_ylabel("Accuracy on the 600 test items")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "sweep.png", dpi=200)
    plt.close(fig)


def hero():
    """One cell per test item, three panels: the untrained lock, SFT on the 0.5B base model, SFT on 0.5B-Instruct.
    Ember where the item is answered right (brighter = more seeds right), grey where wrong; items ordered by id."""
    test = sorted(c.test_items(), key=lambda q: q["id"])
    bands = []
    for arm in ("locked-none", "locked-qwen05base", "locked-qwen05"):
        runs = [json.loads(p.read_text()) for p in sorted((RESULTS / "arms").glob(f"{arm}_s*.json"))]
        acc = np.mean([[r["rows"][q["id"]]["plain"] == q["answer"] for q in test] for r in runs], axis=0)
        bands.append(acc.reshape(30, 20))
        print(f"hero.jpg band {arm}: mean {acc.mean():.3f}")
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    cells, colours = [], []
    for b, band in enumerate(bands):
        for (row, col), v in np.ndenumerate(band):
            cells.append(Rectangle((b * 24 + col + 0.1, row + 0.1), 0.8, 0.8))
            colours.append((0.79, 0.34, 0.11, 0.3 + 0.7 * v) if v > 0 else (0.42, 0.40, 0.37, 0.35))
    fig = plt.figure(figsize=(16, 9), dpi=120)
    ax = fig.add_axes([0, 0, 1, 1])
    fig.patch.set_facecolor(GROUND)
    ax.set_facecolor(GROUND)
    ax.add_collection(PatchCollection(cells, facecolors=colours, linewidths=0))
    ax.set_aspect("equal")
    ax.set_xlim(-6.5, 74.5)
    ax.set_ylim(36.8, -6.8)
    ax.axis("off")
    fig.savefig(OUT / "hero.jpg", dpi=120, facecolor=GROUND, pil_kwargs={"quality": 90})
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    s = load()
    arms_figure(s)
    teachers_figure(s)
    tag_figure(s)
    sweep_figure(s)
    hero()


if __name__ == "__main__":
    main()
