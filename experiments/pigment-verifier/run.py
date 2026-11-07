"""
Score one reward method (reward.py as it is now) and log the result.

    .venv/bin/python experiments/pigment-verifier/run.py --note "verifier, Qwen3-VL-2B, G=20"
    .venv/bin/python experiments/pigment-verifier/run.py --final --note "..."   # held-out test: once, at the end

Prints acc (agreement with the judges on dev pairs, ties count half), tie_rate, hand_rho, cost and time,
appends a row to results.tsv, and saves the exact reward.py that produced it under runs/.
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "experiments" / "pigment-gym"))

import prepare
import reward

parser = argparse.ArgumentParser(description="Score reward.py against the judges' labels")
parser.add_argument("--note", required=True, help="one line: what this run tries")
parser.add_argument("--final", action="store_true", help="use the held-out test split")
args = parser.parse_args()

split = "test" if args.final else "dev"
ids = prepare.evaluation_ids(split)
spent_before, started = prepare.spent(), time.time()
scores = reward.score(ids)
seconds, cost = time.time() - started, prepare.spent() - spent_before
result = prepare.evaluate(scores, split)

for key in ("acc", "acc_within", "tie_rate", "hand_rho"):
    print(f"{key}: {result[key]:.4f}")
print(f"pairs: {result['pairs']}\ncost_usd: {cost:.4f}\nseconds: {seconds:.0f}")
print(f"budget_left: {prepare.BUDGET_USD - prepare.spent():.2f}")

log = HERE / "results.tsv"
if not log.exists():
    log.write_text("run\tsplit\tacc\tacc_within\ttie_rate\thand_rho\tcost_usd\tseconds\tnote\n")
run_id = sum(1 for _ in log.open()) - 1
(HERE / "runs").mkdir(exist_ok=True)
shutil.copy(HERE / "reward.py", HERE / "runs" / f"{run_id:03d}_reward.py")
with log.open("a") as handle:
    handle.write(
        f"{run_id:03d}\t{split}\t{result['acc']:.4f}\t{result['acc_within']:.4f}\t{result['tie_rate']:.4f}\t{result['hand_rho']:.4f}\t"
        f"{cost:.4f}\t{seconds:.0f}\t{args.note}\n"
    )
