"""
The held-out test, run once: score the chosen method and the baselines on the test pairs, from their run snapshots.

    .venv/bin/python experiments/pigment-verifier/final_test.py

The search tuned on the dev pairs only (program.md); this is the first and only time the test pairs are read. Results
are appended to results.tsv with split "test" and saved to results/final_test.json.
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "experiments" / "pigment-gym")]

import prepare

RUNS = {
    "000": "baseline: CLIP + aesthetic (Pigment Gym's reward)",
    "001": "baseline: discrete judge, Qwen3-VL-2B, most likely letter",
    "002": "baseline: verifier, Qwen3-VL-2B, expected score G=20",
    "030": "chosen: 5-question yes/no checklist + 7 x tournament vs 2 anchors + text penalty (Qwen3-VL-2B)",
}

ids = prepare.evaluation_ids("test")
results = {}
for run, note in RUNS.items():
    spec = importlib.util.spec_from_file_location(f"run{run}", HERE / "runs" / f"{run}_reward.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    started = time.time()
    scores = module.score(ids)
    result = prepare.evaluate(scores, "test")
    result["seconds"] = round(time.time() - started)
    results[run] = {"note": note, **result}
    print(f"{run}  acc {result['acc']:.3f}  within {result['acc_within']:.3f}  ties {result['tie_rate']:.3f}  ({note})")
    with (HERE / "results.tsv").open("a") as handle:
        handle.write(
            f"{run}\ttest\t{result['acc']:.4f}\t{result['acc_within']:.4f}\t{result['tie_rate']:.4f}\t"
            f"{result['hand_rho']:.4f}\t0.0000\t{result['seconds']}\tFINAL TEST of run {run}: {note}\n"
        )
(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / "final_test.json").write_text(json.dumps(results, indent=1))
