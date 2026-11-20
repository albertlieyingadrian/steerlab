"""Ruler check 3: does voi-slop-detect-380M separate real Enron emails from zero-shot model drafts?

Human: dev authors' held-out ceiling + profile texts. AI: dev k0 drafts from every pilot model.
Pass bar: AUROC >= 0.85. The flag cut-off is fitted here on human email (5% false positives).
Writes results/ruler_detector.json.
"""

import json

import numpy as np
from sklearn.metrics import roc_auc_score

from common import DATA, HERE, authors, texts
from detectors import voi380

dev = [a for a in authors()["authors"] if a["split"] == "dev"]
human = [t for a in dev for t in texts(a, "ceiling") + texts(a, "profile")]
ai = []
for p in (DATA / "drafts").glob("*__dev.jsonl"):
    ai += [json.loads(l)["text"] for l in p.read_text().splitlines() if l.strip() and json.loads(l)["arm"] == "k0"]
ai = [t for t in ai if len(t.split()) >= 30]
sh, sa = voi380(human), voi380(ai)
auc = roc_auc_score([0] * len(sh) + [1] * len(sa), np.r_[sh, sa])
cut = float(np.percentile(sh, 95))
res = dict(
    auroc=round(auc, 4),
    n_human=len(sh),
    n_ai=len(sa),
    human_mean=round(float(sh.mean()), 3),
    ai_mean=round(float(sa.mean()), 3),
    cut_at_5pct_fpr=round(cut, 4),
    ai_flagged_at_cut=round(float((sa > cut).mean()), 3),
    passed=auc >= 0.85,
)
(HERE / "results" / "ruler_detector.json").write_text(json.dumps(res, indent=1))
print(res)
