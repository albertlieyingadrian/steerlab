"""Ruler check 1 (Design §3): can LUAR tell the same author from a different one on single held-out texts?

For each frozen author, the profile (5 texts) is compared with the author's 5 ceiling texts (same) and with
one ceiling text from each of 20 floor authors (different). Pass bar: pooled AUROC >= 0.85, both with and
without greeting/sign-off stripping. Also reports the floor/ceiling gap per author (the scale's width).
Writes results/ruler_luar.json.
"""

import json

import numpy as np
from sklearn.metrics import roc_auc_score

from common import HERE, authors, luar_embed, strip_greeting_signoff, texts

A = authors()
out = {}
for strip in (False, True):
    f = strip_greeting_signoff if strip else (lambda t: t)
    y, s, gaps = [], [], []
    floor_single = luar_embed([[f(texts(r, "ceiling")[0])] for r in A["floor"][:20]])
    for rec in A["authors"]:
        prof = luar_embed([[f(t) for t in texts(rec, "profile")]])[0]
        same = luar_embed([[f(t)] for t in texts(rec, "ceiling")]) @ prof
        diff = floor_single @ prof
        y += [1] * len(same) + [0] * len(diff)
        s += same.tolist() + diff.tolist()
        gaps.append(float(same.mean() - diff.mean()))
    auc = roc_auc_score(y, s)
    out["stripped" if strip else "raw"] = dict(
        auroc=round(auc, 4),
        mean_gap=round(float(np.mean(gaps)), 4),
        min_gap=round(float(np.min(gaps)), 4),
        n_authors=len(gaps),
    )
    print("strip" if strip else "raw", out["stripped" if strip else "raw"])
out["pass"] = all(v["auroc"] >= 0.85 for v in out.values() if isinstance(v, dict))
(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / "ruler_luar.json").write_text(json.dumps(out, indent=1))
print("PASS" if out["pass"] else "FAIL")
