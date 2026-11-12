r"""Lesson 02: two small ideas behind the human data, splitting by document and trimming. Runs in a second.

cd experiments/slop-detector \&\& ../../.venv/bin/python lessons/02_human_data/lesson.py
"""

import importlib
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
human = importlib.import_module("01_human")

# 1. Split by a hash of the document id, not at random per row.
ids = [f"reddit/travel/{i}" for i in range(10_000)]
print("split shares over 10,000 ids:", dict(Counter(human.split_of(i) for i in ids)))
doc = "hn/31415926"
versions = [f"{doc}|human", f"{doc}|mirror", f"{doc}|rewrite", f"{doc}|roundtrip"]
print(f"\n{doc} lands in '{human.split_of(doc)}' every time; its AI versions use the same id, so they follow it.")
print("If rows were split at random, the rewrite could land in train and its human original in test:")
print("the test would then ask the model about a text it has nearly seen. That is called leakage.")
print("versions:", versions)

# 2. Trim long documents at a paragraph or sentence end.
long = "\n\n".join(f"Paragraph {p}. " + " ".join(["word"] * 120) + "." for p in range(1, 5))
cut = human.trim(long)
print(
    f"\nlong document: {len(long.split())} words -> trimmed to {len(cut.split())} words "
    f"(limit {human.MAX_WORDS}), ending: ...{cut[-30:]!r}"
)
