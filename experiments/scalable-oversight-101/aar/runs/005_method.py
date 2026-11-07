"""
The one file the automated researcher edits: how the strong student learns from the weak teacher's labels.

predict(task) gets a prepare.Task and returns P(correct) for task.eval. Run 000 is the naive baseline: a probe on
the 75%-depth activations, trained on the teacher's probabilities.
"""

import numpy as np


def predict(task):
    # Reference point: the student's own zero-shot verdict, no weak labels at all.
    return 1 / (1 + np.exp(-task.eval_zero_shot))
