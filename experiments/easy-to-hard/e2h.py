"""
Shared helpers for the easy-to-hard note. Every numbered script imports this file (named e2h.py so it does not shadow
scalable-oversight-101's common.py).

The question, in one line: if people can only label easy problems reliably, does training a strong model on easy
problems with correct labels teach it to judge hard problems, and is that better than training on all problems with
a weak model's noisy labels?

It reuses the scalable-oversight-101 data: GSM8K-verification items (a grade-school maths problem plus a proposed
solution; label = whether the solution is right), the cached activations of Qwen2.5-0.5B/1.5B/7B, and the 0.5B
probe teacher. Difficulty is the number of steps in GSM8K's own worked solution:

    easy  = 2 steps        hard = 5 or more steps        (3-4 steps are left out of the easy/hard comparison)

Analogy: a maths teacher can mark short sums with certainty but not long proofs. Is it better to train the marker
on short sums marked perfectly, or on a mix of everything marked by a shaky assistant?
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SO = HERE.parent / "scalable-oversight-101"
sys.path.insert(0, str(SO))
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from common import items

SEED = 20261109
EASY_MAX, HARD_MIN = 2, 5


def steps_by_id():
    """{item id: number of steps in GSM8K's worked solution}, matched by question text."""
    from datasets import load_dataset

    gsm = load_dataset("openai/gsm8k", "main")
    gold = {r["question"].strip(): r["answer"] for sp in ("train", "test") for r in gsm[sp]}
    out = {}
    for split in ("weak_train", "student_train", "dev", "test"):
        for r in items(split):
            out[r["id"]] = gold[r["question"].strip()].split("####")[0].strip().count("\n") + 1
    return out


def bucket(steps):
    return "easy" if steps <= EASY_MAX else "hard" if steps >= HARD_MIN else "mid"


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def dump(name, obj):
    (RESULTS / name).write_text(json.dumps(obj, indent=1, default=float))
