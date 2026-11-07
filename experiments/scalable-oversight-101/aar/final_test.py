"""
The held-out test, run once after the search: the baseline (run 000) and the named runs, scored on test.

    .venv/bin/python experiments/scalable-oversight-101/aar/final_test.py RUN [RUN ...]

Appends test rows to results.tsv and writes results/aar_test.json.
"""

import importlib.util
import json
import sys
from pathlib import Path

import prepare

HERE = Path(__file__).resolve().parent
out = {}
for run in ("000", *sys.argv[1:]):
    spec = importlib.util.spec_from_file_location(f"m{run}", HERE / "runs" / f"{run}_method.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = prepare.evaluate(module.predict(prepare.task("test")), "test", intervals=True)
    out[run] = result
    print(
        f"{run}: acc {result['acc']:.3f}  PGR {result['pgr']:+.2f} [{result['pgr_lo']:+.2f}, {result['pgr_hi']:+.2f}]"
    )
    with (HERE / "results.tsv").open("a") as handle:
        handle.write(f"{run}\ttest\t{result['acc']:.4f}\t{result['pgr']:.4f}\t0\tFINAL TEST of run {run}\n")
(HERE.parent / "results").mkdir(exist_ok=True)
(HERE.parent / "results" / "aar_test.json").write_text(json.dumps(out, indent=1, default=float))
