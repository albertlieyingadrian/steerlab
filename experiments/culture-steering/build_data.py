"""Fixed item sets, written before any model output.

- data/wvs_dev.json: 60 of the explainer note's 181 main questions drawn with seed 0 (tuning and soft-prompt training).
  the explainer note's own 20 dev questions carry one country's split each, so they cannot train or tune 8-country arms.
- data/wvs_heldout.json: the other 121 main questions (confirmation only).
- data/dilemmas.jsonl: the judge-RL note's 32 Kairos-style workplace and family dilemmas (all held out; nothing is tuned on them).

    .venv/bin/python experiments/culture-steering/build_data.py
"""

from __future__ import annotations

import json

import numpy as np

from common import DATA, EP1, ROOT

main = json.loads((EP1 / "data" / "questions.json").read_text())
rng = np.random.default_rng(0)
pick = set(rng.choice(len(main), 60, replace=False).tolist())
dev = [q for i, q in enumerate(main) if i in pick]
held = [q for i, q in enumerate(main) if i not in pick]
assert len(dev) == 60 and len(held) == 121 and not {q["id"] for q in dev} & {q["id"] for q in held}
(DATA / "wvs_dev.json").write_text(json.dumps(dev, indent=1))
(DATA / "wvs_heldout.json").write_text(json.dumps(held, indent=1))

rows = []
for f in ("advice_train.jsonl", "advice_heldout.jsonl"):
    for line in (ROOT / "experiments" / "pluralism-judge-rl" / "data" / f).read_text().splitlines():
        r = json.loads(line)
        if r["source"] == "kairos-style":
            rows.append({"id": r["id"], "prompt": r["prompt"]})
rows.sort(key=lambda r: r["id"])
assert len(rows) == 32, len(rows)
(DATA / "dilemmas.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
print(f"dev {len(dev)}, held-out {len(held)}, dilemmas {len(rows)}")
