"""FIXED harness. Never edit. Holds the data split, the cached lens bank and evaluate().

- Questions: the 10 practice questions + the hand-written dev set (dev/data/all_private.jsonl).
- Split: practice questions are always `dev`. Every other question goes to `dev` or `heldout` by a
  hash of its id, stratified so that each family has about a third held out.
- Lens bank: cached readouts (top-20 token ids per lens per visit per question) computed once on
  Modal (dev/modal_lens.py), so every candidate is scored offline without touching NDIF.
- Scoring: the official vote rule (each readout votes for its best-ranked named option; pass when
  the intermediate has strictly the most votes), with dev/score.py's fuzzy stand-in for the judge.
"""

import glob
import hashlib
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "dev"))
import score  # noqa: E402

BANK = sorted(glob.glob(os.path.join(ROOT, "dev", "runs", "*readouts*.jsonl")))


def split_of(q):
    if q["family"] == "practice":
        return "dev"
    h = int(hashlib.sha1(q["id"].encode()).hexdigest(), 16) % 3
    return "heldout" if h == 0 else "dev"


QS = score.questions(os.path.join(ROOT, "dev", "data", "all_private.jsonl"))
RUNS = score.load(*BANK)
QS = {k: v for k, v in QS.items() if k in RUNS}
SPLITS = defaultdict(list)
for qid, q in QS.items():
    SPLITS[split_of(q)].append(qid)
SPLITS["all"] = sorted(QS)  # for the final fit, after cross-validation (cv.py) has picked the lens set
TABLE = score.vote_table(RUNS, QS, mode="fuzzy")
TABLE_EXACT = score.vote_table(RUNS, QS, mode="exact")
LENSES = sorted({k[0] for k in TABLE})


def valid(picks):
    """One readout per visit's state: a read of visit v's output (act_out, mix_out) is the state entering
    visit v+1, so it may not be combined with any readout at visit v+1."""
    visits = {v for _, v in picks}
    return not any(lens.endswith("_out") and v + 1 in visits for lens, v in picks)


def evaluate(picks, split="dev", exact=False):
    """picks: list of (lens, visit), <= 10, distinct visits. Returns (pass rate, per-family dict)."""
    assert len(picks) <= 10 and len({v for _, v in picks}) == len(picks), picks
    for p in picks:
        assert p in TABLE, f"{p} is not in the lens bank"
    if not valid(picks):
        return 0.0, {}
    table = TABLE_EXACT if exact else TABLE
    ids = SPLITS[split]
    fam = defaultdict(list)
    for qid in ids:
        fam[QS[qid]["family"]].append(qid)
    return score.pass_rate(table, picks, QS, ids), {f: round(score.pass_rate(table, picks, QS, i), 3) for f, i in sorted(fam.items())}
