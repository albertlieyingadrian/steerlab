"""
The fixed half of the small automated researcher: data, the frozen weak teacher, and the PGR score. Never edit this.

method.py gets a Task (the student's activations, the teacher's probabilities, zero-shot scores; never a true label)
and returns P(correct) for the evaluation items. evaluate() turns those into accuracy and PGR on dev, or on test once
at the end. The teacher and the ceiling are the same as in w2s.py.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from w2s import LAYER, bootstrap_pgr, features, fit

STUDENT = "7b"
DEPTHS = 4  # features at 25, 50, 75 and 100% of the student's layers


@dataclass
class Task:
    train: np.ndarray  # student activations on student_train: [1300, DEPTHS, d], standardised
    weak_probs: np.ndarray  # the teacher's P(correct) for each student_train item
    train_zero_shot: np.ndarray  # the student's own logit(Yes) - logit(No) on student_train
    eval: np.ndarray  # student activations on the items to predict
    eval_zero_shot: np.ndarray
    eval_weak_probs: np.ndarray  # the teacher's P(correct) on the items to predict


def _all_depths(name):
    per_depth = [features(name, layer) for layer in range(DEPTHS)]
    return {
        split: (np.stack([d[split][0] for d in per_depth], 1), per_depth[0][split][1], per_depth[0][split][2])
        for split in per_depth[0]
    }


_W = features("0.5b")
_teacher = fit(_W["weak_train"][0], _W["weak_train"][1])
_S = _all_depths(STUDENT)
_ceiling = fit(_S["student_train"][0][:, LAYER], _S["student_train"][1])


def task(split):
    """The inputs a method gets for one evaluation split ("dev", or "test" at the very end)."""
    return Task(
        train=_S["student_train"][0],
        weak_probs=_teacher(_W["student_train"][0]),
        train_zero_shot=_S["student_train"][2],
        eval=_S[split][0],
        eval_zero_shot=_S[split][2],
        eval_weak_probs=_teacher(_W[split][0]),
    )


def evaluate(probs, split, intervals=False):
    """Accuracy and PGR of P(correct) predictions on a split, against the fixed teacher and ceiling."""
    y = _S[split][1]
    weak = ((_teacher(_W[split][0]) > 0.5) == y).astype(float)
    ceiling = ((_ceiling(_S[split][0][:, LAYER]) > 0.5) == y).astype(float)
    method = ((np.asarray(probs) > 0.5) == y).astype(float)
    result = {
        "acc": method.mean(),
        "pgr": (method.mean() - weak.mean()) / (ceiling.mean() - weak.mean()),
        "teacher_acc": weak.mean(),
        "ceiling_acc": ceiling.mean(),
    }
    if intervals:
        result["pgr_median"], result["pgr_lo"], result["pgr_hi"] = bootstrap_pgr(
            weak, ceiling, method, np.random.default_rng(0)
        )
    return result
