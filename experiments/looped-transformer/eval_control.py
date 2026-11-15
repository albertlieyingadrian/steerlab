"""Score saved models on the twin-distractor control set (added after the results, to test a shortcut).

    python experiments/looped-transformer/eval_control.py --runs looped_h16_s1 looped_s2 ... [--runs-dir ...] [--out ...]

The standard test set's distractor chains are at most 8 long, so a question with more than 8 hops is the only long
chain in its input, and a model could answer by elimination ("the chain end no short chain leads to") without
following the chain. In the control set (`make_batch(..., twin=True)`, CONTROL_SEED) one distractor chain has exactly
the question's length, so length cannot pick out the question's chain. 500 questions per hop count 1-24; looped
models at the same loop counts as train.py. Writes <out>/control_<run>.json with correct[k][hops] and the control
set's shortcut baseline per hop count. Runs on Modal through modal_train.py (--task control).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import MAX_HOPS, Looped, build, device, make_batch, shortcut_baseline
from train import EVAL_K, TEST_N

CONTROL_SEED = 2468


def control_set(n=TEST_N):
    rng = np.random.default_rng(CONTROL_SEED)
    return {h: make_batch(rng, n, hops=h, twin=True) for h in range(1, MAX_HOPS + 1)}


@torch.no_grad()
def score(run: str, runs_dir: Path, tests, dev) -> dict:
    model = build(run.rsplit("_s", 1)[0].removesuffix("_h16"))
    model.load_state_dict(torch.load(runs_dir / f"{run}.pt", map_location="cpu"))
    model.to(dev).eval()
    ks = EVAL_K if isinstance(model, Looped) else (None,)
    correct = {}
    for k in ks:
        correct[str(k)] = {
            str(h): int((model(b.to(dev), k=k).argmax(-1) == b.answer.to(dev)).sum()) for h, b in tests.items()
        }
    return {"run": run, "test_n": TEST_N, "control_seed": CONTROL_SEED, "correct": correct}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--runs-dir", default=str(HERE / "results" / "runs"))
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()
    dev = device()
    tests = control_set()
    base = {str(h): shortcut_baseline(b) for h, b in tests.items()}
    for run in args.runs:
        res = score(run, Path(args.runs_dir), tests, dev)
        res["shortcut_baseline"] = base
        Path(args.out, f"control_{run}.json").write_text(json.dumps(res, indent=1))
        mean = {k: round(sum(v.values()) / (TEST_N * MAX_HOPS), 3) for k, v in res["correct"].items()}
        print("wrote", f"control_{run}.json", mean)


if __name__ == "__main__":
    main()
