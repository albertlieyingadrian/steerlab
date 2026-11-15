"""Rescore saved replies with the current checker (tasks.py), in place.

python rescore.py results/screen/Qwen3-1.7B.jsonl.gz [more files ...]
"""

import gzip
import json
import sys
from collections import defaultdict

from tasks import make_task, reward

for path in sys.argv[1:]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    changed = 0
    for r in rows:
        t = make_task(r["domain"], r["level"], r["kind"], r["length"], r["seed"]).to_json()
        new = reward(t, r["text"])
        changed += new != r["correct"]
        r["correct"] = new
    with opener(path, "wt") as f:
        f.write("\n".join(json.dumps(r) for r in rows))
    acc = defaultdict(list)
    for r in rows:
        acc[(r["level"], r["kind"])].append(r["correct"])
    print(path, f"{changed} changed;", {f"L{k[0]} {k[1]}": round(sum(v) / len(v), 3) for k, v in sorted(acc.items())})
