"""R70's random ablation set: 25 latents matched to the released 25 on firing rate at the ablation layer (33), measured
on 300 calibration replies (prompt + reply, max over positions). Writes data/random_ablate_70b.json."""

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import modal

from common import DATA, RELEASED, RESULTS, read_jsonl

jobs = {j["id"]: j for j in read_jsonl(DATA / "jobs" / "calib_70b.jsonl")}
rows = random.Random(4).sample(read_jsonl(RESULTS / "gen_calib_70b.jsonl"), 300)
items = [
    {"id": r["id"], "messages": jobs[r["id"]]["messages"] + [{"role": "assistant", "content": r["turns"][0]}]}
    for r in rows
]
t0 = time.time()
res = modal.Cls.from_name("aiaf-repl-engine", "Engine70")().features.remote(items, None, 33)
counts: dict[int, int] = {}
for r in res:
    for f in r["active"]:
        counts[f] = counts.get(f, 0) + 1
rate = {f: c / len(res) for f, c in counts.items()}
rel = {f: rate.get(f, 0.0) for f in RELEASED}
used = set(RELEASED)
pool = sorted(rate, key=lambda f: rate[f])
rng = random.Random(6)
out = []
for f in sorted(RELEASED, key=lambda f: -rel[f]):
    cands = sorted((g for g in rate if g not in used), key=lambda g: abs(rate[g] - rel[f]))[:5]
    g = rng.choice(cands)
    used.add(g)
    out.append(g)
(DATA / "random_ablate_70b.json").write_text(
    json.dumps(
        {
            "latents": out,
            "layer": 33,
            "n_replies": len(res),
            "released_rates": {str(f): rel[f] for f in RELEASED},
            "random_rates": {str(g): rate[g] for g in out},
        },
        indent=1,
    )
)
print("released rates", sorted(round(v, 2) for v in rel.values()))
print("random   rates", sorted(round(rate[g], 2) for g in out))
print("seconds", round(time.time() - t0))
