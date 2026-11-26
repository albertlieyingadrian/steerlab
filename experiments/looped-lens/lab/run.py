"""Score the current lens.py on dev (and, with --heldout, on the held-out split); append to results.tsv."""

import argparse
import importlib
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prepare  # noqa: E402

TSV = os.path.join(HERE, "results.tsv")


def log_row(picks, note, heldout=False, status="eval"):
    dev, fam = prepare.evaluate(picks, "dev")
    dev_exact, _ = prepare.evaluate(picks, "dev", exact=True)
    ho = prepare.evaluate(picks, "heldout")[0] if heldout else float("nan")
    n = sum(1 for _ in open(TSV)) - 1 if os.path.exists(TSV) else 0
    if not os.path.exists(TSV):
        open(TSV, "w").write("run\ttime\tstatus\tdev\tdev_exact\theldout\tpicks\tnote\n")
    with open(TSV, "a") as f:
        f.write(f"{n:04d}\t{time.strftime('%H:%M:%S')}\t{status}\t{dev:.4f}\t{dev_exact:.4f}\t{ho:.4f}\t{picks}\t{note}\n")
    return n, dev, ho, fam


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--note", default="")
    ap.add_argument("--heldout", action="store_true")
    a = ap.parse_args()
    lens = importlib.import_module("lens")
    n, dev, ho, fam = log_row(lens.PICKS, a.note, a.heldout)
    shutil.copy(os.path.join(HERE, "lens.py"), os.path.join(HERE, "runs", f"{n:04d}_lens.py"))
    print(f"dev: {dev:.4f}\nheldout: {ho:.4f}\nfamilies: {fam}")
