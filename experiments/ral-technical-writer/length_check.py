"""The length explanation, tested: do the held-out judges reward abstracts for being long?

    python length_check.py   # results/per_title.jsonl + samples -> results/length_check.json

For each model source and held-out title, x = log(words of the source's abstract / words of the real abstract) and
y = the two judges' fooling score for that pair. Reports (1) fooling by length-ratio bin, pooled and per source;
(2) a least-squares fit y ~ source + x, so each source's gain over the base model is read at equal length, with 95%
intervals from a bootstrap over titles (2,000 resamples, seed 0). Added after the main results (post hoc).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import score

SOURCES = ["base", "base_fs", "llama_fs", "sft", "frozen", "ral", "ral1"]
BINS = [(0, 0.6), (0.6, 0.9), (0.9, 1.1), (1.1, 1.5), (1.5, 2.2), (2.2, 99)]


def main() -> None:
    H = {x["appl_id"]: x for x in score.heldout()}
    pt = C.read_jsonl(C.RESULTS / "per_title.jsonl")
    rows = []
    for s in SOURCES:
        src = score.load(s)
        for r in pt:
            a = (src.get(r["appl_id"]) or [None])[0] if r["source"] == s else None
            if a:
                ratio = len(a.split()) / len(H[r["appl_id"]]["abstract"].split())
                rows.append((s, r["appl_id"], np.log(ratio), r["ensemble_fooling"]))
    out: dict = {"bins": {}, "per_source_bins": {}}
    x = np.array([r[2] for r in rows])
    y = np.array([r[3] for r in rows])
    for lo, hi in BINS:
        m = (np.exp(x) >= lo) & (np.exp(x) < hi)
        out["bins"][f"{lo}-{hi}"] = {"n": int(m.sum()), "fooling": round(float(y[m].mean()), 4) if m.any() else None}
    for s in SOURCES:
        d = {}
        for lo, hi in BINS:
            v = [r[3] for r in rows if r[0] == s and lo <= np.exp(r[2]) < hi]
            if v:
                d[f"{lo}-{hi}"] = {"n": len(v), "fooling": round(float(np.mean(v)), 4)}
        out["per_source_bins"][s] = d
    titles = sorted({r[1] for r in rows})
    others = [s for s in SOURCES if s != "base"]

    def fit(sel):
        X = np.array([[1.0] + [float(r[0] == s) for s in others] + [r[2]] for r in sel])
        Y = np.array([r[3] for r in sel])
        return np.linalg.lstsq(X, Y, rcond=None)[0]

    beta = fit(rows)
    by_t: dict = {}
    for r in rows:
        by_t.setdefault(r[1], []).append(r)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(2000):
        pick = rng.choice(len(titles), len(titles))
        boots.append(fit([r for i in pick for r in by_t[titles[i]]]))
    boots = np.array(boots)
    lo, hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)
    out["adjusted_gain_over_base"] = {s: {"coef": round(float(beta[i + 1]), 4), "ci": [round(float(lo[i + 1]), 4), round(float(hi[i + 1]), 4)]}
                                      for i, s in enumerate(others)}  # fmt: skip
    out["slope_per_log_length_ratio"] = {
        "coef": round(float(beta[-1]), 4),
        "ci": [round(float(lo[-1]), 4), round(float(hi[-1]), 4)],
    }
    out["n_pairs"] = len(rows)
    (C.RESULTS / "length_check.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
