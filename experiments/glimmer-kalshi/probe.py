"""Exploratory twist: is confidence readable from Qwen3-8B's activations, and what does temperature fix?

Runs on Modal CPU (modal_qwen.py --phase probe) next to the activations.
Inputs (QWEN_DIR): {base,tuned}_{dev,test}.jsonl readouts and
{base,tuned}_{dev,test}_L{layer}.npy residual activations at the answer position.

For each model and layer, two linear probes are fitted on dev and scored on test:
  correctness probe  P(the model's own top option is right)  vs the readout's top probability (raw and
                     with the dev-fitted temperature): AUROC (ranking) and ECE (calibration)
  answer probe       P(yes) on yes/no questions, straight from activations, vs the readout's P(yes):
                     accuracy and AUROC (does the model hold more than its letter logits say?)
Writes results/qwen_probe.json, per-question test predictions (results/qwen_probe_preds.npz) and prints a table.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from evaluate import ece, fit_t, softmax
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
Q = Path(os.environ.get("QWEN_DIR", HERE / "qwen"))
RESULTS = Path(os.environ.get("RESULTS_DIR", HERE / "results"))
LAYERS = (9, 18, 27, 36)
N_TEST_ROWS, N_UNSEEN_ROWS = 8765, 765


def load(tag, split):
    rows = [json.loads(line) for line in (Q / f"{tag}_{split}.jsonl").open()]
    return rows


def readout(rows, temps):
    ps = [softmax(r["logprobs"], temps.get(r["type"], 1.0)) for r in rows]
    top = np.array([p.max() for p in ps])
    correct = np.array([float(p.argmax() == r["gold"]) for p, r in zip(ps, rows, strict=True)])
    pyes = np.array([p[0] for p in ps])
    return top, correct, pyes


def fit_probe(x, y):
    sc = StandardScaler().fit(x)
    clf = LogisticRegressionCV(Cs=[1e-4, 1e-3, 1e-2, 1e-1], cv=5, max_iter=2000, scoring="neg_log_loss")
    clf.fit(sc.transform(x), y)
    return lambda z: clf.predict_proba(sc.transform(z))[:, 1], float(clf.C_[0])


def main():
    out, preds = {}, {}
    for tag in ("base", "tuned"):
        dev, test = load(tag, "dev"), load(tag, "test")
        temps = {t: fit_t([r for r in dev if r["type"] == t]) for t in sorted({r["type"] for r in dev})}
        top_raw, cor_test, pyes_raw = readout(test, {})
        top_cal, _, _ = readout(test, temps)
        _, cor_dev, _ = readout(dev, {})
        noul_dev = np.array([r["type"] == "noul" for r in dev])
        noul_test = np.array([r["type"] == "noul" for r in test])
        y_dev = np.array([r["gold"] == 0 for r in dev], dtype=float)  # yes on noul
        y_test = np.array([r["gold"] == 0 for r in test], dtype=float)
        m = {
            "temps": temps,
            "acc": float(cor_test.mean()),
            "readout": {
                "raw": {
                    "auroc": float(roc_auc_score(cor_test, top_raw)),
                    "ece": ece(top_raw, cor_test),
                },
                "calibrated": {
                    "auroc": float(roc_auc_score(cor_test, top_cal)),
                    "ece": ece(top_cal, cor_test),
                },
            },
            "answer_readout": {
                "acc_raw": float(((pyes_raw > 0.5) == y_test)[noul_test].mean()),
                "auroc": float(roc_auc_score(y_test[noul_test], pyes_raw[noul_test])),
            },
            "layers": {},
        }
        for L in LAYERS:
            xd = np.load(Q / f"{tag}_dev_L{L}.npy").astype(np.float32)
            xt = np.load(Q / f"{tag}_test_L{L}.npy").astype(np.float32)
            pc, c1 = fit_probe(xd, cor_dev)
            p_cor = pc(xt)
            pa, c2 = fit_probe(xd[noul_dev], y_dev[noul_dev])
            p_ans = pa(xt[noul_test])
            m["layers"][L] = {
                "correctness": {
                    "auroc": float(roc_auc_score(cor_test, p_cor)),
                    "ece": ece(p_cor, cor_test),
                    "C": c1,
                },
                "answer": {
                    "acc": float(((p_ans > 0.5) == y_test[noul_test]).mean()),
                    "auroc": float(roc_auc_score(y_test[noul_test], p_ans)),
                    "C": c2,
                },
            }
            preds[f"{tag}_L{L}_correct"] = p_cor.astype(np.float32)
            preds[f"{tag}_L{L}_yes"] = p_ans.astype(np.float32)
            print(tag, L, m["layers"][L], flush=True)
        out[tag] = m
        print(
            tag,
            "acc",
            m["acc"],
            "readout",
            m["readout"],
            "answer readout",
            m["answer_readout"],
            "T",
            temps,
        )
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "qwen_probe.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(RESULTS / "qwen_probe_preds.npz", **preds)  # per test question, in readout order


if __name__ == "__main__":
    main()
