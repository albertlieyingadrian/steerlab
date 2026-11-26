"""The autoresearch loop over the cached lens bank: propose one change to lens.PICKS at a time,
evaluate it on dev, keep it only if dev pass rate improves (ties broken by the vote margin),
log every attempt to results.tsv, and check the held-out split every 10 accepted changes.

    python autoresearch/search.py --iters 400 --lenses act_in,mix_in,... --seed 0
"""

import argparse
import importlib
import os
import pprint
import random
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prepare  # noqa: E402
import run  # noqa: E402


def margin(picks, split="dev"):
    """Mean over questions of (intermediate votes - best other option's votes): a smoother tie-breaker."""
    tot = 0
    for qid in prepare.SPLITS[split]:
        q = prepare.QS[qid]
        gold = q["options"].index(q["intermediate"])
        c = Counter(prepare.TABLE[p].get(qid) for p in picks)
        c.pop(None, None)
        tot += c[gold] - max((n for o, n in c.items() if o != gold), default=0)
    return tot / len(prepare.SPLITS[split])


def propose(picks, cands, rng):
    picks = list(picks)
    used = {v for _, v in picks}
    move = rng.choice(["swap_lens", "move_visit", "add", "remove"])
    if move == "add" and len(picks) < 10:
        c = rng.choice([c for c in cands if c[1] not in used])
        return picks + [c], f"add {c}"
    if move == "remove" and len(picks) > 1:
        i = rng.randrange(len(picks))
        return picks[:i] + picks[i + 1 :], f"remove {picks[i]}"
    i = rng.randrange(len(picks))
    lens, v = picks[i]
    if move == "swap_lens":
        opts = [c for c in cands if c[1] == v and c[0] != lens]
    else:
        opts = [c for c in cands if c[1] not in used and abs(c[1] - v) <= 6]
    if not opts:
        return picks, None
    c = rng.choice(opts)
    picks[i] = c
    return picks, f"{move} {(lens, v)} -> {c}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--lenses", default=",".join(prepare.LENSES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", default="lens")
    ap.add_argument("--split", default="dev")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    allowed = set(a.lenses.split(",")) - {"final"}
    cands = [k for k in prepare.TABLE if k[0] in allowed and k[1] < 48]
    best = [p for p in importlib.import_module(a.start).PICKS if p in prepare.TABLE]
    best_dev, best_m = prepare.evaluate(best, a.split)[0], margin(best, a.split)
    run.log_row(best, f"search start (seed {a.seed}, lenses {sorted(allowed)})", heldout=True, status="start")
    accepted = 0
    for it in range(a.iters):
        cand, note = propose(best, cands, rng)
        if note is None:
            continue
        d, m = prepare.evaluate(cand, a.split)[0], margin(cand, a.split)
        keep = d > best_dev + 1e-9 or (abs(d - best_dev) < 1e-9 and m > best_m + 1e-9)
        if keep:
            accepted += 1
            best, best_dev, best_m = cand, d, m
        run.log_row(cand, note, heldout=keep and accepted % 10 == 0, status="keep" if keep else "discard")
    print(f"best dev {best_dev:.4f} margin {best_m:.3f} heldout {prepare.evaluate(best, 'heldout')[0]:.4f}")
    print(f"dev exact {prepare.evaluate(best, exact=True)[0]:.4f}  heldout exact {prepare.evaluate(best, 'heldout', exact=True)[0]:.4f}")
    print("families dev", prepare.evaluate(best)[1])
    print("families heldout", prepare.evaluate(best, "heldout")[1])
    open(os.path.join(HERE, "best_picks.py"), "w").write("PICKS = " + pprint.pformat(sorted(best, key=lambda p: p[1])) + "\n")
    run.log_row(sorted(best, key=lambda p: p[1]), f"search end (seed {a.seed})", heldout=True, status="best")


if __name__ == "__main__":
    main()
