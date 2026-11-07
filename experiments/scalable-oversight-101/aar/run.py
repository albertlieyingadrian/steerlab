"""
Score the current method.py and log it.

    .venv/bin/python experiments/scalable-oversight-101/aar/run.py --note "what changed"

Prints acc and PGR on dev, appends a row to results.tsv and saves the exact method.py under runs/NNN_method.py.
The test split is read only by final_test.py, once.
"""

import argparse
import shutil
import time
from pathlib import Path

import method
import prepare

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--note", required=True)
args = parser.parse_args()

started = time.time()
result = prepare.evaluate(method.predict(prepare.task("dev")), "dev")
seconds = time.time() - started
log = HERE / "results.tsv"
if not log.exists():
    log.write_text("run\tsplit\tacc\tpgr\tseconds\tnote\n")
run = f"{len(log.read_text().splitlines()) - 1:03d}"
shutil.copy(HERE / "method.py", HERE / "runs" / f"{run}_method.py")
with log.open("a") as handle:
    handle.write(f"{run}\tdev\t{result['acc']:.4f}\t{result['pgr']:.4f}\t{seconds:.0f}\t{args.note}\n")
print(f"run: {run}\nacc: {result['acc']:.4f}\npgr: {result['pgr']:.4f}")
print(f"teacher_acc: {result['teacher_acc']:.4f}\nceiling_acc: {result['ceiling_acc']:.4f}\nseconds: {seconds:.0f}")
