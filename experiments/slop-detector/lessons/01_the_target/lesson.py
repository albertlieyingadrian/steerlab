r"""Lesson 01: one score, three bands, macro-F1. A worked example you can check by hand (runs in a second).

    cd experiments/slop-detector \&\& ../../.venv/bin/python lessons/01_the_target/lesson.py

voi-slop-detect-4B outputs one number per text. Two cut points turn it into three classes, and EditLens judges the
classes with macro-F1. Here are eight made-up texts, so every step can be followed with a pencil.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import BANDS, band, three_way

truth = np.array([0, 0, 0, 1, 1, 1, 2, 2])  # 0 human, 1 AI-edited, 2 AI-generated (what EditLens says)
score = np.array([0.02, 0.11, 0.40, 0.30, 0.71, 0.08, 0.93, 0.80])  # what the detector says

pred = band(score)
print(f"band edges: human < {BANDS['human_max']} <= edited < {BANDS['generated_min']} <= generated")
for t, s, p in zip(truth, score, pred, strict=True):
    mark = "ok " if t == p else "MISS"
    print(f"  score {s:.2f} -> band {p}   truth {t}   {mark}")

m = three_way(truth, pred)
print("\nper-class F1 = 2 * precision * recall / (precision + recall), computed for each class on its own:")
for c in ("human", "edited", "generated"):
    print(f"  {c:9} {m['f1_' + c]:.3f}")
print(f"macro-F1 = plain average of the three = {m['macro_f1']:.3f}   (accuracy {m['acc']:.3f})")
print("\nWhy macro-F1: a detector that calls everything 'human' scores well on accuracy if most texts are human,")
print("but its edited and generated F1 are 0, so macro-F1 punishes it. Every class counts the same.")
