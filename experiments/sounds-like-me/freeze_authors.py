"""Freeze the authors and their text splits before any generation (Design: units of truth first).

Corpora: the "Catch Me If You Can? Not Yet" (EMNLP 2025) Enron and Blog splits, placed in data/raw/.
Per author we keep three disjoint sets, all real human writing:
- `samples`: train-split texts the model may see as examples (k = 0/3/8) and that build voice vectors;
- `profile`: 5 test-split texts that define the author's LUAR profile (the target);
- `ceiling`: 5 other test-split texts, scored against the profile to give the same-human ceiling.
The floor is the profile scored against 50 other authors' ceiling texts.

Output: data/authors.json (author ids, split, and row indices; no text).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
SEED = 20261120
N_DEV, N_ENRON, N_BLOG = 15, 24, 6


def eligible(train, test, lo, hi):
    out = []
    for a, g in test.groupby("author"):
        tr = train[train.author == a]
        tw = g.text.str.split().str.len()
        trw = tr.text.str.split().str.len()
        ok_test = g[(tw >= lo) & (tw <= hi)]
        ok_train = tr[(trw >= lo) & (trw <= hi)]
        if len(ok_test) >= 10 and len(ok_train) >= 12:
            out.append((a, ok_train.index.tolist(), ok_test.index.tolist()))
    return out


def pick(rows, rng, n):
    idx = rng.permutation(len(rows))[:n]
    return [rows[i] for i in idx]


def record(corpus, split, a, tr_idx, te_idx, rng):
    te = rng.permutation(te_idx).tolist()
    return dict(
        corpus=corpus,
        split=split,
        author=str(a),
        samples=sorted(int(i) for i in rng.permutation(tr_idx)[:12]),
        profile=sorted(int(i) for i in te[:5]),
        ceiling=sorted(int(i) for i in te[5:10]),
    )


def main():
    rng = np.random.default_rng(SEED)
    authors = []
    en_tr, en_te = pd.read_csv(RAW / "enron_train.csv"), pd.read_csv(RAW / "enron_test.csv")
    en = eligible(en_tr, en_te, 40, 350)
    chosen = pick(en, rng, N_DEV + N_ENRON)
    for j, (a, tr, te) in enumerate(chosen):
        authors.append(record("enron", "dev" if j < N_DEV else "heldout", a, tr, te, rng))
    bl_tr, bl_te = pd.read_csv(RAW / "blog_train.csv"), pd.read_csv(RAW / "blog_test.csv")
    bl = eligible(bl_tr, bl_te, 60, 350)
    for a, tr, te in pick(bl, rng, N_BLOG):
        authors.append(record("blog", "heldout", a, tr, te, rng))
    # floor pool: other eligible Enron authors not chosen (their ceiling texts stand in for "a different human")
    chosen_ids = {str(c[0]) for c in chosen}
    floor = [record("enron", "floor", a, tr, te, rng) for a, tr, te in en if str(a) not in chosen_ids][:50]
    out = dict(seed=SEED, eligible_enron=len(en), eligible_blog=len(bl), authors=authors, floor=floor)
    (HERE / "data" / "authors.json").write_text(json.dumps(out, indent=1))
    print(f"eligible enron {len(en)}, blog {len(bl)}; frozen {len(authors)} authors + {len(floor)} floor")


if __name__ == "__main__":
    main()
