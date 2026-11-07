"""
The one file the automated researcher edits: how the strong student learns from the weak teacher's labels.

predict(task) gets a prepare.Task and returns P(correct) for task.eval. Run 000 is the naive baseline: a probe on
the 75%-depth activations, trained on the teacher's probabilities.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from w2s import fit

LAYER = 2


def predict(task):
    probe = fit(task.train[:, LAYER], task.weak_probs)
    return probe(task.eval[:, LAYER])
