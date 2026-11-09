"""Item by item: do the LoRA students fix the items the untrained 7B gets wrong? (Added after the results.)

    .venv/bin/python experiments/scalable-oversight-101/overlap.py

Splits the 400 test items by whether Qwen2.5-7B's zero-shot answer (no training) is right, then gives each 7B LoRA
student's accuracy on each part, with 95% intervals from 2,000 bootstrap resamples of the items. A student that only
re-expresses what the untrained model can already do would sit near chance (50%) on the items the zero-shot gets
wrong. Reads the cached activations file for the zero-shot scores (features.py) and results/lora/7b_<method>.json.
Writes results/overlap.json.
"""

import json

import numpy as np

from common import CACHE, RESULTS, items

BOOT = 2000


def ci(right, seed=0):
    rng = np.random.default_rng(seed)
    draws = [right[rng.integers(0, len(right), len(right))].mean() for _ in range(BOOT)]
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def main():
    cache = np.load(CACHE / "7b.npz")
    index = {k: i for i, k in enumerate(cache["ids"])}
    test = items("test")
    y = np.array([r["label"] for r in test])
    zero = np.array([cache["zero_shot"][index[r["id"]]] for r in test]) > 0
    zero_right = zero == y
    out = {"zero_shot_acc": float(zero_right.mean()), "zero_shot_right": int(zero_right.sum()),
           "zero_shot_wrong": int((~zero_right).sum()), "students": {}}  # fmt: skip
    for method in ("naive", "confident", "aux", "ceiling"):
        run = json.loads((RESULTS / "lora" / f"7b_{method}.json").read_text())["test"]
        scores = dict(zip(run["ids"], run["scores"], strict=True))
        right = (np.array([scores[r["id"]] > 0 for r in test]) == y).astype(float)
        out["students"][method] = {
            "acc": float(right.mean()),
            "agree_with_zero_shot": float(np.mean((np.array([scores[r["id"]] > 0 for r in test])) == zero)),
            "right_where_zero_shot_right": [float(right[zero_right].mean()), *ci(right[zero_right])],
            "right_where_zero_shot_wrong": [float(right[~zero_right].mean()), *ci(right[~zero_right])],
        }
        s = out["students"][method]
        print(f"{method:9s} acc {s['acc']:.3f}  where zero-shot right {s['right_where_zero_shot_right']}  "
              f"where zero-shot wrong {s['right_where_zero_shot_wrong']}")  # fmt: skip
    (RESULTS / "overlap.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
