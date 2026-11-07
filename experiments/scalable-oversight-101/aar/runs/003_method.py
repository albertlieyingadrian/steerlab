"""
The one file the automated researcher edits: how the strong student learns from the weak teacher's labels.

predict(task) gets a prepare.Task and returns P(correct) for task.eval. Run 000 is the naive baseline: a probe on
the 75%-depth activations, trained on the teacher's probabilities.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from w2s import fit

LAYER = 1


def predict(task):
    probe = fit(task.train[:, LAYER], task.weak_probs)
    p = np.clip(probe(task.eval[:, LAYER]), 1e-6, 1 - 1e-6)
    # Add the student's own zero-shot verdict (in logits) to the probe's.
    return 1 / (1 + np.exp(-(np.log(p / (1 - p)) + task.eval_zero_shot)))
