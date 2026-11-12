r"""Lesson 06: why compare two detectors on the SAME rows, with a paired bootstrap. Simulated, a few seconds.

    cd experiments/slop-detector \&\& ../../.venv/bin/python lessons/06_paired_eval/lesson.py

Two made-up detectors on 2,000 rows. B is right on 85% of rows; A agrees with B on most rows and fixes 2% of B's
mistakes. Is A better? The unpaired view says "can't tell"; the paired view says "yes".
"""

import numpy as np

rng = np.random.default_rng(0)
n = 2000
b_right = rng.random(n) < 0.85
a_right = b_right.copy()
fix = (~b_right) & (rng.random(n) < 0.15)  # A fixes some of B's mistakes...
a_right[fix] = True
a_right[b_right & (rng.random(n) < 0.004)] = False  # ...and makes a few new ones
print(f"accuracy A {a_right.mean():.3f}, B {b_right.mean():.3f}, difference {a_right.mean() - b_right.mean():+.3f}")

reps = 2000
paired, a_alone, b_alone = [], [], []
for _ in range(reps):
    i = rng.integers(0, n, n)  # paired: the SAME resampled rows for both
    paired.append(a_right[i].mean() - b_right[i].mean())
    a_alone.append(a_right[rng.integers(0, n, n)].mean())  # unpaired: each on its own rows
    b_alone.append(b_right[rng.integers(0, n, n)].mean())


def ci(v):
    return np.quantile(v, [0.025, 0.975]).round(3).tolist()


print(f"unpaired: A {ci(a_alone)}, B {ci(b_alone)}  -> the intervals overlap, so 'no clear difference'")
print(f"paired difference A - B: {ci(paired)}  -> entirely above 0, so A is better")
print("\nWhy: most rows are easy or hard for BOTH detectors. Resampling the same rows cancels that shared noise,")
print("like comparing two runners on the same track on the same day instead of on different days.")
