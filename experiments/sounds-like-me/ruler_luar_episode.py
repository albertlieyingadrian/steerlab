"""Ruler check 1b: LUAR on 4-text episodes (one model's 4 briefs for an author are scored together).

Same: the author's 4 ceiling texts vs the profile. Different: 4 ceiling texts of each of 20 floor authors.
Pass bar unchanged: AUROC >= 0.85. Writes results/ruler_luar_episode.json.
"""

import json

from sklearn.metrics import roc_auc_score

from common import HERE, authors, luar_embed, strip_greeting_signoff, texts

A = authors()
f = strip_greeting_signoff
floor_eps = luar_embed([[f(t) for t in texts(r, "ceiling")[:4]] for r in A["floor"][:20]])
y, s = [], []
for rec in A["authors"]:
    prof = luar_embed([[f(t) for t in texts(rec, "profile")]])[0]
    same = luar_embed([[f(t) for t in texts(rec, "ceiling")[:4]]])[0] @ prof
    y += [1] + [0] * len(floor_eps)
    s += [float(same), *(floor_eps @ prof).tolist()]
auc = roc_auc_score(y, s)
res = dict(auroc=round(auc, 4), n_same=sum(y), n_diff=len(y) - sum(y), passed=auc >= 0.85)
(HERE / "results" / "ruler_luar_episode.json").write_text(json.dumps(res, indent=1))
print(res)
