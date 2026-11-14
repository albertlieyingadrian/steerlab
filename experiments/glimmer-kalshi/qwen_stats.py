"""Event-bootstrap intervals for the twist's key AUROC comparisons (probe vs letter readout), Qwen3-8B test set.

Reads qwen/{base,tuned}_test.jsonl and results/qwen_probe_preds.npz (written by probe.py on Modal) and writes
results/qwen_stats.json: for each model, AUROC of the readout's top probability and of the best-layer correctness probe
for "is the top answer right?", and of the readout's P(yes) and the answer probe for yes/no questions, each with a
95% interval from 1,000 resamples of whole events, plus the paired difference probe minus readout.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from evaluate import softmax
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
RNG = np.random.default_rng(0)


def boot(rows_idx, y, a, b, draws=1000):
    groups = {}
    for i, r in enumerate(rows_idx):
        groups.setdefault(r, []).append(i)
    keys = list(groups)
    da, db, dd = [], [], []
    for _ in range(draws):
        s = RNG.integers(0, len(keys), len(keys))
        idx = np.concatenate([groups[keys[k]] for k in s])
        if y[idx].min() == y[idx].max():
            continue
        x, z = roc_auc_score(y[idx], a[idx]), roc_auc_score(y[idx], b[idx])
        da.append(x), db.append(z), dd.append(z - x)

    def ci(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    return {
        "readout": float(roc_auc_score(y, a)),
        "readout_ci": ci(da),
        "probe": float(roc_auc_score(y, b)),
        "probe_ci": ci(db),
        "probe_minus_readout_ci": ci(dd),
    }


def main():
    preds = np.load(HERE / "results" / "qwen_probe_preds.npz")
    probe = json.loads((HERE / "results" / "qwen_probe.json").read_text())
    out = {}
    for tag in ("base", "tuned"):
        rows = [json.loads(line) for line in (HERE / "qwen" / f"{tag}_test.jsonl").open()]
        ps = [softmax(r["logprobs"], 1.0) for r in rows]
        top = np.array([p.max() for p in ps])
        cor = np.array([float(p.argmax() == r["gold"]) for p, r in zip(ps, rows, strict=True)])
        noul = np.array([r["type"] == "noul" for r in rows])
        pyes = np.array([p[0] for p in ps])[noul]
        yes = np.array([float(r["gold"] == 0) for r in rows])[noul]
        best_c = max(
            probe[tag]["layers"],
            key=lambda L: probe[tag]["layers"][L]["correctness"]["auroc"],
        )
        best_a = max(
            probe[tag]["layers"],
            key=lambda L: probe[tag]["layers"][L]["answer"]["auroc"],
        )
        row_ids = np.array([r["row"] for r in rows])
        out[tag] = {
            "correctness": {
                "layer": best_c,
                **boot(row_ids, cor, top, preds[f"{tag}_L{best_c}_correct"]),
            },
            "answer": {
                "layer": best_a,
                **boot(row_ids[noul], yes, pyes, preds[f"{tag}_L{best_a}_yes"]),
            },
        }
        print(tag, json.dumps(out[tag]))
    (HERE / "results" / "qwen_stats.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
