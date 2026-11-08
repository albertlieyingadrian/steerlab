"""
Scalable oversight, script 3: stop the student from copying its teacher's mistakes.

Script 2's student mostly imitated the teacher: on the training statements the teacher got wrong, the probe sided
with the truth only a few percent of the time, so PGR stayed low. Burns et al. (2023) saw the same and proposed ways
to make the student trust its own knowledge more. Two laptop-sized versions, at the best layer from script 2:

1. Stronger regularisation (smaller C). A probe with less freedom can't bend to fit 123 scattered wrong labels, so it
   has to find the one direction that explains most of them: hopefully the model's own "true vs false" direction.
2. Self-training ("trust yourself when confident"), a simple stand-in for the paper's auxiliary confidence loss:
   fit on weak labels, then replace each training label with the student's own prediction where the student is
   confident (P > 0.9 or < 0.1), and fit again.

The same two fixes are applied to the random-flip control, so a gain is not just "more regularisation helps anyway".

Run: python experiments/ai-safety-101/oversight/3_stop_imitating.py   (under a minute)
Needs scripts 1 and 2 (data.json, acts_1.5B.npz, results/2_weak_to_strong.json).
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tools as t

LESSON = Path(__file__).resolve().parent
DATA = t.OUT / "01" / "data.json"
ACTS = t.OUT / "01" / "acts_1.5B.npz"
RESULT = t.RESULTS / "oversight" / "3_stop_imitating.json"
C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
CONFIDENT = 0.9


def self_train(x, y_weak, train, c):
    """Fit on weak labels, overwrite confident training labels with the student's own call, fit again."""
    first = t.fit_probe(x[train], y_weak[train], c)
    p = first.predict_proba(x[train])[:, 1]
    relabel = y_weak[train].copy()
    sure = (p > CONFIDENT) | (p < 1 - CONFIDENT)
    relabel[sure] = (p[sure] > 0.5).astype(int)
    return t.fit_probe(x[train], relabel, c), float((relabel != y_weak[train]).mean())


def main():
    data = json.loads(DATA.read_text())
    best = json.loads((t.RESULTS / "oversight" / "2_weak_to_strong.json").read_text())["best_layer"]
    labels = np.array(data["label"])
    weak = np.array(data["weak_label"])
    train = np.array([s == "train" for s in data["split"]])
    test = ~train
    x = np.load(ACTS)[str(best)]

    n_wrong = int((weak[train] != labels[train]).sum())
    rng = np.random.default_rng(t.SEED)
    noisy = labels.copy()
    flip = rng.choice(np.flatnonzero(train), size=n_wrong, replace=False)
    noisy[flip] = 1 - noisy[flip]

    weak_correct = weak[test] == labels[test]
    teacher_wrong = train & (weak != labels)
    print(f"Layer {best}. Weak teacher on test {t.fmt(t.accuracy_ci(weak_correct))}.")
    print(
        f"\n{'C':>7s} {'method':>13s} {'ceiling':>8s} {'w2s':>8s} {'PGR (weak labels)':>24s} "
        f"{'PGR (random flips)':>24s} {'corrects teacher':>17s}"
    )
    rows = []
    for c in C_GRID:
        ceiling = t.fit_probe(x[train], labels[train], c).predict(x[test]) == labels[test]
        for method in ("plain", "self-train"):
            if method == "plain":
                probe, changed = t.fit_probe(x[train], weak[train], c), 0.0
                control = t.fit_probe(x[train], noisy[train], c)
            else:
                probe, changed = self_train(x, weak, train, c)
                control, _ = self_train(x, noisy, train, c)
            w2s = probe.predict(x[test]) == labels[test]
            ctl = control.predict(x[test]) == labels[test]
            fixes = float((probe.predict(x[teacher_wrong]) == labels[teacher_wrong]).mean())
            row = {
                "C": c,
                "method": method,
                "ceiling": t.accuracy_ci(ceiling),
                "w2s": t.accuracy_ci(w2s),
                "pgr": t.pgr_ci(weak_correct, w2s, ceiling),
                "pgr_control": t.pgr_ci(weak_correct, ctl, ceiling),
                "corrects_teacher": fixes,
                "labels_changed_by_self_training": changed,
            }
            rows.append(row)
            print(
                f"{c:>7g} {method:>13s} {ceiling.mean():>8.3f} {w2s.mean():>8.3f} {t.fmt(row['pgr']):>24s} "
                f"{t.fmt(row['pgr_control']):>24s} {fixes:>17.1%}"
            )

    top = max(rows, key=lambda r: r["pgr"][0] if not np.isnan(r["pgr"][0]) else -1)
    print(
        f"\nHighest PGR: {t.fmt(top['pgr'])} with C = {top['C']:g}, {top['method']} "
        f"(test accuracy {top['w2s'][0]:.3f} vs the teacher's {weak_correct.mean():.3f})."
    )
    t.write_json(RESULT, {"layer": best, "confident": CONFIDENT, "rows": rows, "best": top})

    print("\nSelf-checks")
    t.check("every arm is scored on the same held-out test cities", int(test.sum()) == 748)
    t.check(
        "self-training never rewrote more than half the training labels",
        all(r["labels_changed_by_self_training"] < 0.5 for r in rows if r["method"] == "self-train"),
    )
    t.check("results written", RESULT.exists())


if __name__ == "__main__":
    main()
