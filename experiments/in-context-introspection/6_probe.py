"""
Step 6. The internal ceiling: is "the hint changed my answer" written in Qwen's activations? (about 2 minutes)

    .venv/bin/python experiments/in-context-introspection/6_probe.py

A linear probe is a weighted sum of the 4,096 numbers in one layer's residual stream, passed through a logistic
function, fitted to predict the RELIED label. Analogy: a lie detector reading pulse and sweat instead of asking. If the
probe can tell relied from not-relied answers, the information is in the model even when its words do not show it.

Two read-out positions, both from the full run (step 2 and step 3 saved them):
    answer   the last token before Qwen writes its answer letter, with the hint in context
    self     the last token before Qwen answers "did the professor's opinion change your answer?"

Two training sets:
    all      all 1,000 items (an easy job: "is the model about to say the hinted letter?" already does well)
    hard     only the hard subset (answers equal to the hinted option), where that shortcut is useless

Scores are out-of-fold: 5 folds, one row per question, so no question is ever scored by a probe that saw it. The
layer is chosen inside each training fold (3-fold inner cross-validation), so the reported score is not tuned on the
items it is scored on. Writes results/probe_full.csv and results/probe_layers.json.
"""

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c

C_REG = 0.001
"""Inverse L2 strength after standardising each feature (fixed before fitting; strong, because d=4,096 >> n)."""

rows = c.answers("full")
y_all = np.array([r["relied"] for r in rows]).astype(int)
hard_all = np.array([r["ans_hint"] == r["hint"] for r in rows])
acts = {
    "answer": np.load(c.WORK / "acts_answer_full.npy").astype(np.float32),
    "self": np.load(c.WORK / "acts_self_full.npy").astype(np.float32),
}
t0 = time.time()


def fit(x, y):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C_REG, max_iter=2000)).fit(x, y)


def layer_scores(x, y, folds=5, seed=0):
    """Plain 5-fold out-of-fold AUROC for every layer (the per-layer table, not used to pick anything)."""
    out = {}
    for j, L in enumerate(c.LAYERS):
        s = np.zeros(len(y))
        for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(x, y):
            s[te] = fit(x[tr, j], y[tr]).predict_proba(x[te, j])[:, 1]
        out[str(L)] = round(c.auroc(y, s), 4)
    return out


def nested(x, y, seed=0):
    """Out-of-fold scores with the layer chosen inside each outer training fold."""
    s = np.zeros(len(y))
    chosen = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(x, y):
        inner = []
        for j in range(len(c.LAYERS)):
            si = np.zeros(len(tr))
            for a, b in StratifiedKFold(3, shuffle=True, random_state=seed).split(x[tr], y[tr]):
                si[b] = fit(x[tr][a, j], y[tr][a]).predict_proba(x[tr][b, j])[:, 1]
            inner.append(c.auroc(y[tr], si))
        j = int(np.argmax(inner))
        chosen.append(c.LAYERS[j])
        s[te] = fit(x[tr, j], y[tr]).predict_proba(x[te, j])[:, 1]
    return s, chosen


out = {r["id"]: {"id": r["id"]} for r in rows}
layers = {}
for pos, x in acts.items():
    s, chosen = nested(x, y_all)
    for r, v in zip(rows, s, strict=True):
        out[r["id"]][f"probe_{pos}"] = round(float(v), 5)
    layers[f"{pos}_all"] = {"per_layer_auroc": layer_scores(x, y_all), "chosen_layers": chosen}
    a_all, a_hard = c.auroc(y_all, s), c.auroc(y_all[hard_all], s[hard_all])
    print(f"{pos:6} all:  AUROC all {a_all:.3f}, hard {a_hard:.3f}  layers {chosen}")

    xh, yh = x[hard_all], y_all[hard_all]
    sh, chosen = nested(xh, yh)
    for r, v in zip([r for r, h in zip(rows, hard_all, strict=True) if h], sh, strict=True):
        out[r["id"]][f"probe_{pos}_hard"] = round(float(v), 5)
    layers[f"{pos}_hard"] = {"per_layer_auroc": layer_scores(xh, yh), "chosen_layers": chosen}
    print(f"{pos:6} hard: AUROC hard {c.auroc(yh, sh):.3f}  layers {chosen}  ({time.time() - t0:.0f} s)")

fields = ["id", "probe_answer", "probe_self", "probe_answer_hard", "probe_self_hard"]
c.write_csv(c.RESULTS / "probe_full.csv", list(out.values()), fields)
c.write_json(c.RESULTS / "probe_layers.json", {"C": C_REG, "layers": list(c.LAYERS), **layers})
print("\nper-layer out-of-fold AUROC (not used for selection):")
for k, v in layers.items():
    print(f"  {k:12} " + "  ".join(f"L{L}:{a:.3f}" for L, a in v["per_layer_auroc"].items()))
print("\nSelf-checks:")
c.check("activations line up with the answer table", all(len(x) == len(rows) for x in acts.values()))
c.check("hard subset has both classes", 0 < y_all[hard_all].sum() < hard_all.sum())
