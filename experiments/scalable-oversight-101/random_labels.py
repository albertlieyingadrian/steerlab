"""
Random-label control for weak-to-strong: how much of the students' score comes from the teacher's labels at all?

    .venv/bin/python experiments/scalable-oversight-101/random_labels.py

The teacher's probabilities on student_train are shuffled across items, so the students see labels with the same
mix of values but no link to the item they belong to. Every probe method from w2s.py is then trained exactly as before
on the shuffled labels, 20 times with 20 different shuffles. If a method scores about as well on shuffled labels as
on the real ones, its score came from the student itself, not from what the teacher taught. LoRA students get the
same control through w2s_lora.py --shuffled. Writes results/random_labels.json.
"""

import json

import numpy as np

from common import RESULTS
from w2s import features, fit, student_methods

SHUFFLES = 20


def main():
    W = features("0.5b")
    teacher = fit(W["weak_train"][0], W["weak_train"][1])
    p_weak = teacher(W["student_train"][0])
    y_test = W["test"][1]
    real = json.loads((RESULTS / "w2s.json").read_text())
    out = {"shuffles": SHUFFLES, "teacher_test_acc": real["teacher_test_acc"], "students": {}}
    for name in ("1.5b", "7b"):
        S = features(name)
        accs = {}
        for seed in range(SHUFFLES):
            shuffled = np.random.default_rng(seed).permutation(p_weak)
            for method, p in student_methods(S, shuffled).items():
                accs.setdefault(method, []).append(float(((p > 0.5) == y_test).mean()))
        result = {}
        print(f"student {name}")
        for method, a in accs.items():
            real_acc = real["students"][name]["methods"][method]["test_acc"]
            result[method] = {"real_labels_acc": real_acc, "shuffled_mean": float(np.mean(a)),
                              "shuffled_min": min(a), "shuffled_max": max(a), "shuffled_accs": a}  # fmt: skip
            print(f"  {method:10} real {real_acc:.3f}   shuffled {np.mean(a):.3f} [{min(a):.3f}, {max(a):.3f}]")
        out["students"][name] = result
    (RESULTS / "random_labels.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
