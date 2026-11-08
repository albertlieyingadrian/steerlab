"""
Scalable oversight, script 2: train the strong student on the weak teacher's labels, and measure the gap recovered.

The student is a linear probe on Qwen2.5-1.5B-Instruct's residual stream (last prompt token), the same probe recipe
as the sandbagging study. Three probes per layer, all scored against the TRUE labels on the held-out test cities:

- ceiling: trained on the true labels (what the student could reach with perfect supervision);
- weak-to-strong (w2s): trained on the weak teacher's labels (what we would actually get);
- control, random errors: trained on true labels with the same NUMBER of labels flipped at random. If the student
  recovers better from random errors than from the teacher's, the teacher's errors are systematic, and systematic
  errors are what a student can learn to imitate.

PGR = (w2s - weak) / (ceiling - weak): 0 = no better than the teacher, 1 = as good as the true labels.
Also: on the training statements the teacher got WRONG, how often does the w2s probe side with the truth?

Run: python experiments/residency-101/oversight/2_weak_to_strong.py   (about 2 minutes)
Needs script 1's ~/.cache/steerlab/residency-101/01/data.json.
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
RESULT = t.RESULTS / "oversight" / "2_weak_to_strong.json"
LAYERS = (7, 14, 21, 28)
"""Pre-registered: a quarter, half, three quarters and all of Qwen2.5-1.5B's 28 blocks."""


def main():
    data = json.loads(DATA.read_text())
    labels = np.array(data["label"])
    weak = np.array(data["weak_label"])
    train = np.array([s == "train" for s in data["split"]])
    test = ~train

    if not ACTS.exists():
        model, tokenizer, device = t.load_model(t.STRONG_MODEL)
        acts = t.last_token_acts(model, tokenizer, device, data["statements"], LAYERS)
        np.savez(ACTS, **{str(k): v for k, v in acts.items()})
    stored = np.load(ACTS)
    acts = {layer: stored[str(layer)] for layer in LAYERS}

    # The random-errors control: flip as many training labels as the teacher got wrong, chosen at random.
    n_wrong = int((weak[train] != labels[train]).sum())
    rng = np.random.default_rng(t.SEED)
    noisy = labels.copy()
    flip = rng.choice(np.flatnonzero(train), size=n_wrong, replace=False)
    noisy[flip] = 1 - noisy[flip]

    weak_correct = weak[test] == labels[test]
    print(f"Weak teacher on test: {t.fmt(t.accuracy_ci(weak_correct))}   ({n_wrong} wrong training labels)")
    print(
        f"\n{'layer':>5s} {'ceiling (true labels)':>24s} {'w2s (weak labels)':>24s} {'PGR':>24s} "
        f"{'control: random flips':>24s} {'corrects teacher':>17s}"
    )
    rows = {}
    for layer in LAYERS:
        x = acts[layer]
        ceiling = t.fit_probe(x[train], labels[train]).predict(x[test]) == labels[test]
        w2s_probe = t.fit_probe(x[train], weak[train])
        w2s = w2s_probe.predict(x[test]) == labels[test]
        control = t.fit_probe(x[train], noisy[train]).predict(x[test]) == labels[test]
        teacher_wrong = train & (weak != labels)
        fixes = float((w2s_probe.predict(x[teacher_wrong]) == labels[teacher_wrong]).mean())
        row = {
            "ceiling": t.accuracy_ci(ceiling),
            "w2s": t.accuracy_ci(w2s),
            "pgr": t.pgr_ci(weak_correct, w2s, ceiling),
            "control_random_flips": t.accuracy_ci(control),
            "pgr_control": t.pgr_ci(weak_correct, control, ceiling),
            "w2s_agrees_with_truth_where_teacher_wrong_train": fixes,
        }
        rows[layer] = row
        print(
            f"{layer:>5d} {t.fmt(row['ceiling']):>24s} {t.fmt(row['w2s']):>24s} {t.fmt(row['pgr']):>24s} "
            f"{t.fmt(row['control_random_flips']):>24s} {fixes:>17.1%}"
        )

    best = max(LAYERS, key=lambda layer: rows[layer]["ceiling"][0])
    print(
        f"\nBest layer by ceiling: {best}. PGR there {t.fmt(rows[best]['pgr'])}; "
        f"with the same number of random flips instead, PGR {t.fmt(rows[best]['pgr_control'])}."
    )
    t.write_json(
        RESULT,
        {
            "layers": list(LAYERS),
            "weak_test": t.accuracy_ci(weak_correct),
            "n_wrong_train": n_wrong,
            "by_layer": {str(k): v for k, v in rows.items()},
            "best_layer": best,
        },
    )

    print("\nSelf-checks")
    t.check(
        "the ceiling probe beats the weak teacher at the best layer", rows[best]["ceiling"][0] > weak_correct.mean()
    )
    t.check(
        "the control flips exactly as many labels as the teacher got wrong",
        int((noisy[train] != labels[train]).sum()) == n_wrong,
    )
    t.check("no test statement was used to fit any probe", not (train & test).any())
    t.check("results written", RESULT.exists())


if __name__ == "__main__":
    main()
