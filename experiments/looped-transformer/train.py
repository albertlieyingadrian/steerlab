"""Train one model on the chain task and score it on every hop count and loop count.

    python experiments/looped-transformer/train.py --model looped --seed 0 [--steps 20000] [--out results/runs]

Models (see common.build): standard4, standard10, looped (k drawn uniformly from 1-6 at every training step),
looped_fixed4 (k = 4 at every step), looped_noinject (random k, no re-injection of the input).
Training data are fresh random examples at every step (never repeated), with hop counts uniform in 1..cur_max.
Curriculum, the same for every model: cur_max starts at 2 and rises by one (to at most 8) whenever accuracy over the
last CURRIC_EVERY training batches reaches CURRIC_BAR. The test set is fixed by
TEST_SEED: TEST_N examples at each hop count 1-24, the same for every model and seed.
Writes <out>/<model>_s<seed>.pt and <out>/<model>_s<seed>.json (accuracy per hop and k, the training curve).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import MAX_HOPS, TRAIN_HOPS, Looped, build, device, make_batch, n_params

TEST_SEED, TEST_N = 12345, 500
EVAL_K = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16)
TRAIN_K = (1, 6)
CURRIC_EVERY, CURRIC_BAR, CURRIC_START = 250, 0.95, 2
# looped_nonorm, looped_noinject and looped draw k at random; looped_fixed4 always uses 4


def test_set(n=TEST_N):
    rng = np.random.default_rng(TEST_SEED)
    return {h: make_batch(rng, n, hops=h) for h in range(1, MAX_HOPS + 1)}


@torch.no_grad()
def evaluate(model, tests, dev, ks):
    model.eval()
    out = {}
    for k in ks:
        row = {}
        for h, b in tests.items():
            b = b.to(dev)
            row[h] = int((model(b, k=k).argmax(-1) == b.answer).sum())
        out[k] = row
    model.train()
    return out


def train_k(model_name, rng):
    if model_name == "looped_fixed4":
        return 4
    if model_name.startswith("looped"):
        return int(rng.integers(TRAIN_K[0], TRAIN_K[1] + 1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=str(HERE / "results" / "runs"))
    ap.add_argument("--max-hops", type=int, default=TRAIN_HOPS[1], help="curriculum ceiling (amendment 1, E1: 16)")
    ap.add_argument("--tag", default="", help="suffix for output names, e.g. _h16")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(1000 + args.seed)
    dev = device()
    model = build(args.model).to(dev)
    looped = isinstance(model, Looped)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.98))
    warm = 500
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1, s / args.steps)))
    )
    tests = test_set()
    small = {h: tests[h] for h in (2, 4, 8, 12)}
    curve, curriculum = [], [(0, CURRIC_START)]
    cur_max, hits, seen = CURRIC_START, 0, 0
    t0 = time.time()
    for step in range(args.steps):
        b = make_batch(rng, args.batch, hops_range=(1, cur_max)).to(dev)
        logits = model(b, k=train_k(args.model, rng))
        loss = F.cross_entropy(logits, b.answer)
        hits += int((logits.argmax(-1) == b.answer).sum())
        seen += len(b.answer)
        if (step + 1) % CURRIC_EVERY == 0:
            if hits / seen >= CURRIC_BAR and cur_max < args.max_hops:
                cur_max += 1
                curriculum.append((step + 1, cur_max))
            hits = seen = 0
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == args.steps - 1:
            acc = evaluate(model, small, dev, (1, 4, 8) if looped else (None,))
            rec = {
                "step": step,
                "loss": float(loss),
                "acc": {str(k): {str(h): v / TEST_N for h, v in r.items()} for k, r in acc.items()},
            }
            curve.append(rec)
            rec["cur_max"] = cur_max
            print(
                step,
                f"max{cur_max}",
                f"{float(loss):.3f}",
                json.dumps(rec["acc"]),
                f"{time.time() - t0:.0f}s",
                flush=True,
            )

    ks = EVAL_K if looped else (None,)
    final = evaluate(model, tests, dev, ks)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{args.model}{args.tag}_s{args.seed}"
    torch.save(model.state_dict(), out / f"{name}.pt")
    res = {
        "model": args.model,
        "seed": args.seed,
        "params": n_params(model),
        "steps": args.steps,
        "batch": args.batch,
        "lr": args.lr,
        "max_hops": args.max_hops,
        "train_k": "4" if args.model == "looped_fixed4" else (list(TRAIN_K) if looped else None),
        "test_n": TEST_N,
        "seconds": time.time() - t0,
        "device": str(dev),
        # correct answers out of test_n, keyed by loop count k ("None" for standard models), then hop count
        "correct": {str(k): {str(h): v for h, v in r.items()} for k, r in final.items()},
        "depth": {str(k): model.depth(k) for k in ks},
        "curve": curve,
        "curriculum": curriculum,  # (step, new maximum hop count)
    }
    (out / f"{name}.json").write_text(json.dumps(res, indent=1))
    print("wrote", out / f"{name}.json")


if __name__ == "__main__":
    main()
