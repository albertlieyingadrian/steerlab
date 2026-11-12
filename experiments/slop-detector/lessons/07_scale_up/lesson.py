"""Lesson 07: plan a bigger run with arithmetic first, and see why a threshold is a choice. Numbers only, instant.

cd experiments/slop-detector && ../../.venv/bin/python lessons/07_scale_up/lesson.py
"""

import numpy as np

# 1. How long will it take? rows x epochs / rows per second, with rates measured on short test runs first.
measured = {  # model: (GPU, rows per second measured in this experiment)
    "Qwen3-0.6B": ("H100", 137),
    "Qwen3-1.7B": ("H100", 80),
    "Qwen3.5-4B": ("A100-80GB, gradient checkpointing", 13.4),
}
rows, epochs = 79_120, 2
print(f"training rows {rows:,} x {epochs} epochs = {rows * epochs:,} rows to push through\n")
for model, (gpu, rate) in measured.items():
    hours = rows * epochs / rate / 3600
    print(f"  {model:11} on {gpu:33} {rate:6.1f} rows/s -> {hours:4.1f} h")
print("\nThe 4B's real run: 158,505 rows in 11,871 s = 13.4 rows/s, 3.3 h. Measure on 3,000 rows, then multiply.\n")

# 2. A flag cut is a choice: the same scores, two cuts, different errors.
rng = np.random.default_rng(0)
humans = np.clip(rng.beta(1.2, 12, 150) + rng.normal(0, 0.02, 150), 0, 1)  # mostly low scores
imitations = np.clip(rng.beta(6, 2, 297), 0, 1)  # mostly high scores, some low
for cut in (0.177, 0.385):
    fp = int((humans >= cut).sum())
    fn = int((imitations < cut).sum())
    print(f"cut {cut:.3f}: humans wrongly flagged {fp:3} of 150, imitations missed {fn:3} of 297")
print("Raising the cut trades accusing humans for missing AI. Fit it on human text like the text you will judge.")
