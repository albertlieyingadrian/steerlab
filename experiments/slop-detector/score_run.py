"""Score every benchmark text with a run already trained on Modal (its adapter is on the volume), once.

    python score_run.py v2-1.7b

Used for the v2 1.7B, which trained without benchmark scoring so that the validation gate (compare_val.py) came
first. Writes runs/<run>/scores/<split>.json; then `python 05_eval.py --run runs/<run>` compares it with the target.
"""

from __future__ import annotations

import json
import sys

from common import HERE, benchmark, empty_fences

SPLITS = ("test", "test_enron", "test_llama", "human_detectors", "epoch")


def main() -> None:
    import modal

    run = sys.argv[1]
    evals = {s: [empty_fences(str(t)) for t in benchmark(s)["text"]] for s in SPLITS}
    out = modal.Function.from_name("slop-detector", "score_run").remote(run, evals)
    path = HERE / "runs" / run / "scores"
    path.mkdir(parents=True, exist_ok=True)
    for split, scores in out.items():
        (path / f"{split}.json").write_text(json.dumps(scores) + "\n")
        print(split, len(scores))


if __name__ == "__main__":
    main()
