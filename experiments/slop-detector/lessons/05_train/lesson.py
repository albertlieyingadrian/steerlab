r"""Lesson 05: why BCE on a soft target teaches a fraction, and why LoRA is cheap. Numbers only, instant.

cd experiments/slop-detector \&\& ../../.venv/bin/python lessons/05_train/lesson.py
"""

import numpy as np

# 1. BCE with a soft target is lowest exactly at the target.
target = 0.43  # "43% of the words were changed"
p = np.linspace(0.01, 0.99, 99)
bce = -(target * np.log(p) + (1 - target) * np.log(1 - p))
print(f"target {target}: BCE is smallest at p = {p[bce.argmin()]:.2f}")
for q in (0.1, 0.3, 0.43, 0.6, 0.9):
    i = round(q * 100) - 1
    print(f"  model says {q:.2f} -> loss {bce[i]:.3f}{'   <- minimum' if q == 0.43 else ''}")
print("So the model is pushed to output the fraction itself, not just 'AI or not'.\n")

# 2. LoRA: train two thin matrices instead of the whole weight.
d_in, d_out, r = 2048, 2048, 16  # one attention projection of Qwen3-1.7B, LoRA rank 16
full, lora = d_in * d_out, r * (d_in + d_out)
print(
    f"one {d_in}x{d_out} projection: {full:,} weights; LoRA r={r} adds A ({d_in}x{r}) and B ({r}x{d_out}) = "
    f"{lora:,} ({lora / full:.1%})"
)
print("The update is W + (alpha / r) * B @ A. B starts at zero, so training starts from the base model exactly.")
print("Like a patch file on top of a big binary: small to store, and you can always take it off.\n")

# 3. The head: one row of numbers that turns the last hidden state into one logit.
rng = np.random.default_rng(0)
hidden = rng.normal(size=2048)  # the last token's hidden state
head = rng.normal(size=2048) * 0.01
logit = float(hidden @ head)
print(f"head: logit = hidden . w = {logit:+.3f}  ->  score = sigmoid(logit) = {1 / (1 + np.exp(-logit)):.3f}")
