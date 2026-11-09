"""Step 1: candidate weak teachers answer every quiz item; their answers on the train facts become training labels.

    python experiments/weak-supervision-unlock/teachers.py              # all four candidates
    python experiments/weak-supervision-unlock/teachers.py qwen05       # one

Each teacher answers the 600 fine-tuning items and the 600 test items (letter argmax, no password, no tag). The
teacher is chosen by the rule written in the README before any student was trained.
Writes results/teacher_<key>.json: accuracy per split and each item's letter and probabilities.
"""

from __future__ import annotations

import gc
import sys
import time

import numpy as np

import common as c


def run(key: str) -> dict:
    start = time.time()
    model, tokenizer = c.load_plain(c.TEACHERS[key])
    qs = c.train_items() + c.test_items()
    probs = c.letter_probs(model, tokenizer, qs)
    rows = {
        q["id"]: {"letter": int(p.argmax()), "p": [round(float(x), 5) for x in p]}
        for q, p in zip(qs, probs, strict=True)
    }
    del model
    gc.collect()
    acc = {
        s: float(np.mean([rows[q["id"]]["letter"] == q["answer"] for q in qs if q["split"] == s]))
        for s in ("train", "test")
    }
    print(f"{key} ({c.TEACHERS[key]}): accuracy {acc} ({time.time() - start:.0f} s)")
    return {"teacher": key, "model": c.TEACHERS[key], "accuracy": acc, "rows": rows}


def main() -> None:
    for key in sys.argv[1:] or list(c.TEACHERS):
        c.write_json(c.RESULTS / f"teacher_{key}.json", run(key))


if __name__ == "__main__":
    main()
