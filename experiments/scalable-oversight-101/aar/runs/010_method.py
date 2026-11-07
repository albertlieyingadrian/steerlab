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

LAYER = 2


def predict(task):
    first = fit(task.train[:, LAYER], task.weak_probs)
    q = np.clip(first(task.train[:, LAYER]), 1e-6, 1 - 1e-6)
    pseudo = 1 / (1 + np.exp(-(np.log(q / (1 - q)) + task.train_zero_shot)))
    # Self-training: retrain on the student's own combined verdicts, its surest half only.
    sure = np.abs(pseudo - 0.5) >= np.median(np.abs(pseudo - 0.5))
    probe = fit(task.train[sure, LAYER], (pseudo[sure] > 0.5).astype(np.float32))
    p = np.clip(probe(task.eval[:, LAYER]), 1e-6, 1 - 1e-6)
    # Add the student's own zero-shot verdict (in logits) to the probe's.
    return 1 / (1 + np.exp(-(np.log(p / (1 - p)) + task.eval_zero_shot)))
