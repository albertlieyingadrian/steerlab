"""Cross-validate the whole lens-selection procedure from `results/votes.json.gz` alone (no GPU, no NDIF).

    python experiments/looped-lens/cv.py            # 8 folds, writes results/cv.json
    python experiments/looped-lens/cv.py --check    # also compare with results/cv_check.json (the lab run)

For each of 8 random splits of the 867 questions (seed 100 + fold), a hill-climb picks up to 10 (lens, visit) reads on
two thirds, starting from the starter's ten visits, using only the lenses in one "lens set"; the picks are then scored
on the other third. A lens set helps only if its picks pass more test questions than the starter's fixed visits.

The vote rule is the challenge's: each read votes for the option its best-ranked naming token names; a question
passes when the intermediate gets strictly the most votes. Matching here is the fuzzy stand-in for the judge.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

RAW = "act_in,mix_in,attn_in,mlp_in"
TUNED = "tuned_act_in,tuned_mlp_in"
FUT = "future_act_in,future_mlp_in"
CON = "concept_mlp_in"
WSP = (
    "wsp_balanced_middle_sum,wsp_frozen_middle,wsp_handoff_attention_mix,wsp_late_mlp_mix_no_prelude,"
    "wsp_third_pass_q_message"
)
SETS = {
    "starter": None,
    "raw": RAW,
    "raw+tuned": f"{RAW},{TUNED}",
    "raw+future": f"{RAW},{FUT}",
    "raw+concept": f"{RAW},{CON}",
    "raw+future+concept": f"{RAW},{FUT},{CON}",
    "all (v1 set)": f"{RAW},{TUNED},{FUT}",
    "all+concept (v2 set)": f"{RAW},{TUNED},{FUT},{CON}",
    "raw+wsp": f"{RAW},{WSP}",
    "all+concept+wsp": f"{RAW},{TUNED},{FUT},{CON},{WSP}",
}
STARTER = [("act_in", v) for v in [12, 24, 34, 35, 36, 43, 44, 45, 46, 47]]
V1 = [
    ("mlp_in", 24),
    ("mlp_in", 27),
    ("future_mlp_in", 34),
    ("mlp_in", 35),
    ("mlp_in", 43),
    ("future_mlp_in", 44),
    ("act_in", 45),
    ("act_in", 46),
    ("act_in", 47),
]
V2 = [
    ("mlp_in", 8),
    ("act_in", 12),
    ("concept_mlp_in", 27),
    ("concept_mlp_in", 33),
    ("future_mlp_in", 34),
    ("act_in", 36),
    ("mlp_in", 42),
    ("act_in", 44),
    ("future_act_in", 45),
    ("act_in", 46),
]


def load():
    with gzip.open(RES / "votes.json.gz", "rt") as f:
        votes = json.load(f)
    qs = {}
    for line in open(RES / "questions.jsonl"):
        q = json.loads(line)
        qs[q["id"]] = q
    qids = votes["qids"]
    table = {}
    for key in votes["keys"]:
        lens, v = key.rsplit("|", 1)
        col = votes["fuzzy"][key]
        table[(lens, int(v))] = {q: (None if c == "." else int(c)) for q, c in zip(qids, col)}
    gold = {q: qs[q]["options"].index(qs[q]["intermediate"]) for q in qids}
    return qids, qs, table, gold


def plurality(votes, gold):
    c = Counter(v for v in votes if v is not None)
    return c[gold] > max((n for o, n in c.items() if o != gold), default=0)


def passed(table, gold, picks, qid):
    return int(plurality([table[tuple(p)].get(qid) for p in picks], gold[qid]))


def valid(picks):
    """A read of visit v's output (act_out, mix_out) is the state entering v + 1.

    It is never combined with a read at v + 1."""
    visits = {v for _, v in picks}
    return not any(lens.endswith("_out") and v + 1 in visits for lens, v in picks)


def propose(picks, cands, rng):
    picks = list(picks)
    used = {v for _, v in picks}
    move = rng.choice(["swap_lens", "move_visit", "add", "remove"])
    if move == "add" and len(picks) < 10:
        c = rng.choice([c for c in cands if c[1] not in used])
        return [*picks, c], f"add {c}"
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


def climb(table, gold, train, lenses, iters, rng):
    cands = [k for k in table if k[0] in lenses and k[1] < 48]
    best = list(STARTER)

    def rate(p):
        return sum(passed(table, gold, p, q) for q in train) / len(train) if valid(p) else 0

    bd = rate(best)
    for _ in range(iters):
        cand, note = propose(best, cands, rng)
        if note is None:
            continue
        d = rate(cand)
        if d > bd + 1e-9 or (d == bd and rng.random() < 0.3):
            best, bd = cand, d
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    qids, _qs, table, gold = load()
    out = {"folds": [], "fixed": {}}
    for name, picks in [("starter", STARTER), ("v1", V1), ("v2", V2)]:
        out["fixed"][name] = {q: passed(table, gold, picks, q) for q in qids}
    for fold in range(a.folds):
        rng = random.Random(100 + fold)
        sh = qids[:]
        rng.shuffle(sh)
        test, train = sh[: len(sh) // 3], sh[len(sh) // 3 :]
        rec = {"test": test, "picks": {}, "res": {}}
        for name, ls in SETS.items():
            picks = (
                STARTER if ls is None else climb(table, gold, train, set(ls.split(",")), a.iters, random.Random(fold))
            )
            rec["picks"][name] = picks
            rec["res"][name] = [passed(table, gold, picks, q) for q in test]
        out["folds"].append(rec)
        print(fold, {k: round(sum(v) / len(v), 3) for k, v in rec["res"].items()}, flush=True)
    (RES / "cv.json").write_text(json.dumps(out))
    if a.check:
        ref = json.load(open(RES / "cv_check.json"))
        same = all(r["res"] == c["res"] and r["test"] == c["test"] for r, c in zip(out["folds"], ref["folds"]))
        same = same and out["fixed"] == ref["fixed"]
        print("matches the lab run:", same)
        if not same:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
