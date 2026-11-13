"""Lesson 08: labels by how a text was made, the expected-bucket score, and a sign test. Made-up numbers, instant.

cd experiments/slop-detector && ../../.venv/bin/python lessons/08_beat_it/lesson.py
"""

import numpy as np
from scipy.stats import binomtest

# 1. Two labels for the same three texts: share of words changed vs how the text was made.
texts = [
    ("a grammar fix", "edit", 0.04),
    ("'make it concise'", "edit", 0.45),
    ("written from a summary", "generated", 1.0),
]


def band(v: float) -> str:
    return "human" if v < 0.15 else "generated" if v >= 0.75 else "edited"


for name, kind, share in texts:
    by_how = 1.0 if kind == "generated" else 0.20 + 0.50 * share
    print(f"{name:24} share changed {share:.2f} -> {band(share):9} | by how it was made {by_how:.2f} -> {band(by_how)}")

# 2. A 4-bucket head: probabilities for human, lighter edit, heavier edit, generated -> one score in 0..1.
probs = np.array([0.10, 0.70, 0.15, 0.05])
score = (probs * np.arange(4)).sum() / 3
print(f"\nbucket probabilities {probs} -> expected bucket / 3 = {score:.3f} (edited band: 1/6 to 5/6)")

# 3. A sign test: only the texts where exactly one detector is right count.
for ours_only, theirs_only in ((478, 248), (30, 22)):
    p = binomtest(ours_only, ours_only + theirs_only, 0.5).pvalue
    print(f"ours right alone {ours_only:3}, theirs right alone {theirs_only:3}: p = {p:.2g}")
print("The first pair is the fresh test (fixed cut-offs); the second is what a coin-flip-sized gap looks like.")
