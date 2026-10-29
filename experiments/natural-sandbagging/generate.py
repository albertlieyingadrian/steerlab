"""
Qwen3-8B answers every test item under every framing (thinking on, about 2 hours).

    .venv/bin/python experiments/natural-sandbagging/generate.py --run core
    .venv/bin/python experiments/natural-sandbagging/generate.py --run motive

    core     600 WMDP + 600 MMLU items x {neutral, eval, deploy} x 2 seeds = 7,200 generations
    motive   the same 1,200 items x the separate "motive" framing x seed 0  = 1,200 generations (analysed apart)

Each generation is two steps (see natural.generate):

    1. sample the reasoning, at most 512 tokens          "<think> Okay, the question asks ... </think>"
    2. append "Answer:" and read the four letter probabilities at the next position: p(A), p(B), p(C), p(D)

Step 2 is like asking a student to put the pen down and circle a letter: there is always an answer, so nothing has to
be parsed, and the probability on the right letter is a finer score than right/wrong.

The run is append-only (one JSONL line per generation in the work folder), so stopping and rerunning resumes.
This script writes no results; step 2 scores what it produced.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import natural as t

RUNS = {
    "core": (t.FRAMINGS_CORE, (0, 1)),
    "motive": (("motive",), (0,)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=sorted(RUNS), required=True)
    args = ap.parse_args()
    kinds, seeds = RUNS[args.run]
    items = t.load_items("test")
    jobs = t.jobs_for(items, kinds, seeds)
    out = t.WORK / "stage1" / f"{args.run}.jsonl"
    print(f"run {args.run}: {len(items)} items x {len(kinds)} framings x {len(seeds)} seeds = {len(jobs)} generations")
    print(f"budget {t.CAP} reasoning tokens, batch {t.BATCH}, writing {t.rel(out)}")
    t0 = time.time()
    tok, model = t.load_model()
    t.generate(tok, model, jobs, out)
    rows = t.read_jsonl(out)
    done = {r["key"] for r in rows}
    print(f"\n{len(done)} generations on disk, {(time.time() - t0) / 60:.1f} min this session")
    checks = [
        t.check("every job has a row", all(j["key"] in done for j in jobs)),
        t.check(
            "every row has four letter probabilities summing to 1", all(abs(sum(r["probs"]) - 1) < 1e-3 for r in rows)
        ),
    ]
    print(f"{sum(checks)}/{len(checks)} checks passed")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
