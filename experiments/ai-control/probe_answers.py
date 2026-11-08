"""Exploratory, added after reading the protocol results: "defer to the probe", a fallback that needs no trusted model.

    python experiments/ai-control/probe_answers.py              # clean and locked (and any other names given)

For each calibration and test item, the untrusted model's own truth probe (fitted in step 1, read from the work
folder, not refitted) reads "prompt + A", "prompt + B", "prompt + C" and "prompt + D". The option it scores least
likely false is the probe's answer: what the model's activations say is true, whatever it chose to say.
Adds "probe_options" (4 scores) to results/items_<which>.json.
"""

from __future__ import annotations

import gc
import sys

import numpy as np

import control as c


def main() -> None:
    for which in sys.argv[1:] or ["clean", "locked"]:
        path = c.RESULTS / f"items_{which}.json"
        data = c.read_json(path)
        weights = np.load(c.work_dir() / f"probe_{which}.npz")
        model, tokenizer = c.load_untrusted(which)
        qs = [q for q in c.all_items() if q["split"] != "probe"]
        texts = [t for q in qs for t in c.answer_statements(tokenizer, [q] * 4, [0, 1, 2, 3])]
        z = c.activations(model, tokenizer, texts) @ weights["coef"].T + weights["intercept"]
        p_false = (1 / (1 + np.exp(-z))).reshape(len(qs), 4)
        for q, row in zip(qs, p_false, strict=False):
            data["rows"][q["id"]]["probe_options"] = [round(float(x), 6) for x in row]
        test = [i for i, q in enumerate(qs) if q["split"] == "test"]
        acc = np.mean([p_false[i].argmin() == qs[i]["answer"] for i in test])
        data["accuracy_probe_answer_test"] = float(acc)
        print(f"{which}: the probe's own answer is right on {acc:.3f} of test items")
        c.write_json(path, data)
        del model
        gc.collect()


if __name__ == "__main__":
    main()
