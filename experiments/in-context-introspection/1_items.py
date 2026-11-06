"""
Step 1. The item table: 200 pilot and 1,000 full MMLU test questions, each with a hinted wrong letter (seconds).

    .venv/bin/python experiments/in-context-introspection/1_items.py

Writes results/items.csv (IDs, subjects and letters; the question text stays in the local Hugging Face cache).
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c

items, pool = c.build_items()
c.write_csv(c.RESULTS / "items.csv", items, ["id", "row", "subject", "answer", "hint", "split"])
split = Counter(i["split"] for i in items)
print(f"MMLU test rows with four options and at most {c.MAX_CHARS} characters: {pool}")
print(f"drawn: {split['pilot']} pilot + {split['full']} full, seed {c.SEED}")
print(f"subjects: {len({i['subject'] for i in items})}")
print("correct letter:", dict(sorted(Counter(i["answer"] for i in items).items())))
print("hinted letter: ", dict(sorted(Counter(i["hint"] for i in items).items())))
print(f"wrote {c.rel(c.RESULTS / 'items.csv')}\n\nSelf-checks:")
c.check("pilot and full sets are disjoint", len({i["id"] for i in items}) == len(items))
c.check("the hint never points at the correct option", all(i["hint"] != i["answer"] for i in items))
c.check(f"{c.N_PILOT} + {c.N_FULL} items", split["pilot"] == c.N_PILOT and split["full"] == c.N_FULL)
