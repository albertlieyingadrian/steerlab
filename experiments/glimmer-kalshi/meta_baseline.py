"""Boring-explanation check for the probes: how far do question metadata alone go?

Logistic regression on category, series-family size, question type, option count and prompt length (no activations),
fitted on dev and scored on test, for the same two targets as probe.py: whether the model's top option is right, and
the yes/no answer itself. If this matched the activation probes, the probes would only be reading what kind of
question it is. Writes results/{prefix}meta_baseline.json.

    python meta_baseline.py --dir qwen --out qwen_
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

HERE = Path(__file__).resolve().parent


def feats(rows, data):
    cat = [
        [
            data[r["row"]]["state"].get("category", ""),
            r["type"],
            str(min(len(r["logprobs"]), 6)),
            str(min(len(data[r["row"]]["questions"]), 8)),
        ]
        for r in rows
    ]
    num = np.array([[r["n_tokens"] / 500] for r in rows])
    return cat, num


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="qwen")
    ap.add_argument("--out", default="qwen_")
    a = ap.parse_args()
    data = {s: [json.loads(line) for line in (HERE / "data" / f"{s}.jsonl").open()] for s in ("dev", "test")}
    out = {}
    for tag in ("base", "tuned"):
        rows = {s: [json.loads(line) for line in (HERE / a.dir / f"{tag}_{s}.jsonl").open()] for s in ("dev", "test")}
        enc = OneHotEncoder(handle_unknown="ignore")
        X = {}
        for s in ("dev", "test"):
            cat, num = feats(rows[s], data[s])
            X[s] = (cat, num)
        enc.fit(X["dev"][0])
        M = {s: np.hstack([enc.transform(X[s][0]).toarray(), X[s][1]]) for s in X}
        cor = {s: np.array([float(np.argmax(r["logprobs"]) == r["gold"]) for r in rows[s]]) for s in rows}
        noul = {s: np.array([r["type"] == "noul" for r in rows[s]]) for s in rows}
        yes = {s: np.array([float(r["gold"] == 0) for r in rows[s]]) for s in rows}
        c = LogisticRegression(max_iter=2000).fit(M["dev"], cor["dev"])
        y = LogisticRegression(max_iter=2000).fit(M["dev"][noul["dev"]], yes["dev"][noul["dev"]])
        pc = c.predict_proba(M["test"])[:, 1]
        py = y.predict_proba(M["test"][noul["test"]])[:, 1]
        out[tag] = {
            "correctness_auroc": float(roc_auc_score(cor["test"], pc)),
            "answer_auroc": float(roc_auc_score(yes["test"][noul["test"]], py)),
            "answer_acc": float(((py > 0.5) == yes["test"][noul["test"]]).mean()),
        }
        print(tag, out[tag])
    (HERE / "results" / f"{a.out}meta_baseline.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
